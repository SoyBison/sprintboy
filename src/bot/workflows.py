"""
Deterministic workflows for requests whose shape is known.

A discography request used to go through the chat model, which spent a minute
or more calling the same handful of tools in the same order. Here code makes
those calls and decision models answer the two judgement questions (which
words name the artist, and how much of their catalogue is wanted). A workflow
returns None whenever it cannot handle a message confidently, and the agent
then runs exactly as before.
"""

import asyncio
import logging
import re
import string
import time
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Awaitable, Callable

from thefuzz import fuzz

from bot.agent import AgentResult, Step
from bot.decide import decide
from bot.llm import Message, assistant
from bot.orpheus import AccountStats, OrpheusClient, OrpheusError, plan_purchase
from bot.netcode import BTCategory, LastFMClient, QBittorrentClient
from bot.config import Config
from bot.questions import (
    ACCOUNT_NAME,
    DISCOGRAPHY_QUESTIONS_NAME,
    RECOMMEND_FIT_NAME,
    RECOMMEND_SEED_NAME,
    SPECIFIC_EXTRACT_NAME,
    SPECIFIC_PICK_NAME,
    account_questions,
    discography_questions,
    recommend_fit_questions,
    recommend_seed_questions,
    specific_extract_questions,
    specific_pick_questions,
)
from bot.releases import Release, _norm, best_versions, parse_release, quality_key
from bot.routing import route_state
from bot.tags import tag_vocabulary, vocabulary_matches
from bot.tools import (
    AlbumRef,
    TorrentContext,
    _add_one,
    _canonicalise,
    _library_lookup,
    _maybe_title,
    _ownership,
    _same_release,
    _title_matches,
    download_many,
)

logger = logging.getLogger(__name__)

MIN_ARTIST_CONFIDENCE = 0.6
MIN_SCOPE_CONFIDENCE = 0.5
# A maybe-owned album is only added when the model is fairly sure it is a different one.
SAME_RELEASE_SKIP = 0.25
MAX_SPANS = 250
MAX_OWNED_LISTED = 15

STOPWORDS = {
    "get", "me", "the", "rest", "of", "my", "discography", "collection",
    "everything", "by", "all", "albums", "newest", "latest", "album", "grab",
    "fill", "in", "update", "to", "include", "his", "her", "their", "new",
    "please", "pls", "a", "an", "and", "from",
}
_PUNCTUATION = string.punctuation + "“”‘’«»"
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)
_MEDIA = re.compile(r"\b(sacd|cd|web|vinyl)\b", re.IGNORECASE)
_MEDIA_NAMES = {"sacd": "SACD", "cd": "CD", "web": "WEB", "vinyl": "Vinyl"}
_EXCLUDED_KINDS = {"Compilation", "DJ Mix", "Bootleg", "Interview", "Mixtape"}
_SCOPE_NOUNS = {"albums": "albums", "everything": "releases", "newest": "newest release"}
_SCOPE_SINGULAR = {"albums": "album", "everything": "release", "newest": "newest release"}


def candidate_spans(
    text: str, max_words: int = 6, stopwords: set[str] = STOPWORDS
) -> list[str]:
    """Word n-grams of the message that could be an artist name."""
    words = []
    for raw in text.split():
        word = _POSSESSIVE.sub("", raw.strip(_PUNCTUATION)).strip(_PUNCTUATION)
        if word:
            words.append(word)
    spans: dict[str, str] = {}
    for start in range(len(words)):
        for end in range(start + 1, min(start + max_words, len(words)) + 1):
            chunk = words[start:end]
            if chunk[0].casefold() in stopwords or chunk[-1].casefold() in stopwords:
                continue
            span = " ".join(chunk)
            spans.setdefault(span.casefold(), span)
    return list(spans.values())[:MAX_SPANS]


@dataclass
class Choice:
    label: str  # "Weather Report - Night Passage"
    artist: str
    title: str


@dataclass
class Pending:
    """An unresolved "did you mean": what they asked for and the likeliest releases."""

    wanted: str  # as they wrote it
    options: list[Choice]  # 2-3, best first


@dataclass
class Confirmation:
    """Something that costs money: only run `action` after the owner presses Confirm."""

    prompt: str
    action: Callable[[], Awaitable[str]]


@dataclass
class WorkflowResult:
    reply: str
    steps: list[Step]
    pending: list[Pending] = field(default_factory=list)
    confirm: Confirmation | None = None

    def to_agent_result(self, messages: list[Message]) -> AgentResult:
        return AgentResult(
            messages=[*messages, assistant(self.reply)],
            steps=self.steps,
            stopped="workflow",
            pending=self.pending,
            confirm=self.confirm,
        )


def _media_from(text: str) -> str | None:
    match = _MEDIA.search(text)
    return _MEDIA_NAMES[match.group(1).lower()] if match else None


def _wanted(
    ranked: list[Release], names: set[str], scope: str, artist: str, media: str | None
) -> list[Release]:
    """The releases by this artist that the scope asks for."""

    def by_artist(r: Release) -> bool:
        credit = r.artist.casefold()
        if credit in names:
            return True
        return scope == "everything" and fuzz.partial_ratio(artist.casefold(), credit) >= 95

    kept = [r for r in ranked if by_artist(r) and (media == "Vinyl" or not r.vinyl)]
    if scope == "albums":
        return [r for r in kept if r.kind == "Album"]
    if scope == "everything":
        return [r for r in kept if r.kind not in _EXCLUDED_KINDS]
    kept = [r for r in kept if r.kind in ("Album", "EP")]
    if not kept:
        return []
    newest = max(r.year for r in kept)
    kept = [r for r in kept if r.year == newest]
    kept.sort(key=lambda r: r.kind != "Album")
    return kept[:1]


def _describe(r: Release) -> str:
    detail = f"{r.year}, {r.media}" + (", 24bit" if "24bit" in r.encoding.lower() else "")
    return f"{r.title} ({detail})"


def _reason(result: str) -> str:
    if "DRY_RUN" in result:
        return "dry run is on"
    reason = result.removeprefix("NOT ADDED:").strip()
    reason = re.sub(r"\s*Tell the user[^.]*\.", "", reason).strip()
    return reason or "unknown reason"


def _owned_line(titles: list[str]) -> str:
    shown = titles[:MAX_OWNED_LISTED]
    line = "You already have: " + ", ".join(shown)
    if len(titles) > len(shown):
        line += f" and {len(titles) - len(shown)} more"
    return line


def _unique_titles(titles: list[str]) -> list[str]:
    return list({_norm(t): t for t in reversed(titles)}.values())[::-1]


async def discography(
    text: str, context: TorrentContext, run_id: str | None = None, max_add: int = 20
) -> WorkflowResult | None:
    """Add the albums by one artist that the user does not own, or None to defer to the agent."""
    steps: list[Step] = []

    def record(stage: str, started: float, args: dict, result: str) -> None:
        steps.append(
            Step(
                kind="tool",
                name=f"discography/{stage}",
                seconds=time.perf_counter() - started,
                args=args,
                result=result,
            )
        )

    # a. Who and how much.
    started = time.perf_counter()
    spans = candidate_spans(text)
    if not spans:
        return None
    decision = await decide(
        DISCOGRAPHY_QUESTIONS_NAME,
        {"message": text},
        discography_questions(spans),
        run_id=run_id,
    )
    if decision is None:
        return None
    span, span_p = decision.choice("artist")
    scope, scope_p = decision.choice("scope")
    # Grabbing every single and live record by mistake costs far more than
    # grabbing only the albums, so going wider needs a confident answer.
    if scope_p < MIN_SCOPE_CONFIDENCE or (scope == "everything" and scope_p < 0.85):
        scope = "albums"
    record(
        "decide", started, {"spans": len(spans)},
        f"artist={span!r} ({span_p:.2f}) scope={scope} ({scope_p:.2f})",
    )
    if span == "none" or span_p < MIN_ARTIST_CONFIDENCE:
        return None

    # b. Canonical spelling.
    started = time.perf_counter()
    release = await _canonicalise(span, None)
    artist = release.artist
    candidates = release.artist_candidates or [artist]
    names = {c.casefold() for c in candidates} | {artist.casefold()}
    record("canonicalise", started, {"artist": span}, f"{artist} {candidates}")

    # c. One tracker search.
    started = time.perf_counter()
    media = _media_from(text)
    async with QBittorrentClient() as qclient:
        response = await qclient.search(artist, BTCategory.Music)
    found = [r for r in (response.results or []) if "FLAC" in r.fileName]
    for result in found:
        context.search_results[result.fileName] = result
    ranked = best_versions([r.fileName for r in found], media)
    record("search", started, {"query": artist}, f"{len(found)} FLAC results")

    # d. What the scope asks for.
    wanted = _wanted(ranked, names, scope, artist, media)
    if not wanted:
        return None

    # e. What they already have.
    started = time.perf_counter()
    library = await _library_lookup(candidates)
    if any(titles is None for titles in library.values()):
        record("library", started, {"artists": candidates}, "library check failed")
        return WorkflowResult(
            reply=(
                f"I couldn't check your library for {artist} just now, so I didn't add "
                f"anything. Try again in a moment."
            ),
            steps=steps,
        )
    owned_list = [t for titles in library.values() for t in titles or []]
    record("library", started, {"artists": candidates}, f"{len(owned_list)} albums owned")

    started = time.perf_counter()
    owned: list[str] = []
    maybes: list[tuple[Release, str]] = []
    missing: list[Release] = []
    for r in wanted:
        mark = _ownership(owned_list, r.title)
        if mark == " [OWNED]":
            owned.append(r.title)
        elif mark.startswith(" [maybe owned as"):
            maybes.append((r, _maybe_title(owned_list, r.title) or ""))
        else:
            missing.append(r)
    probs = await asyncio.gather(
        *[_same_release(f"{artist} - {r.title}", f"{artist} - {t}") for r, t in maybes]
    )
    skipped: list[str] = []
    for (r, t), p in zip(maybes, probs):
        if p is None or p >= SAME_RELEASE_SKIP:
            skipped.append(f"{r.title} (as '{t}')")
        else:
            missing.append(r)
    record(
        "ownership", started, {"maybes": len(maybes)},
        f"owned={len(owned)} skipped={len(skipped)} missing={len(missing)}",
    )

    # f. One version per album, oldest first, capped.
    best: dict[str, Release] = {}
    for r in missing:
        current = best.get(_norm(r.title))
        if current is None or quality_key(r, media) > quality_key(current, media):
            best[_norm(r.title)] = r
    to_add = sorted(best.values(), key=lambda r: (r.year, r.title))
    later = to_add[max_add:]
    to_add = to_add[:max_add]

    # g. Add.
    added: list[Release] = []
    failed: list[tuple[Release, str]] = []
    if to_add:
        started = time.perf_counter()
        context.torrent_types.add(BTCategory.Music)
        outcomes = await asyncio.gather(
            *[_add_one(r.name, BTCategory.Music, context, corrected_name=r.name) for r in to_add],
            return_exceptions=True,
        )
        for r, outcome in zip(to_add, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning(f"Adding {r.name} failed: {outcome}")
                failed.append((r, f"{type(outcome).__name__}: {outcome}"))
            elif outcome.startswith("Added"):
                added.append(r)
            else:
                failed.append((r, _reason(outcome)))
        record(
            "add", started, {"names": [r.name for r in to_add]},
            f"added={len(added)} failed={len(failed)}",
        )

    # h. Reply.
    noun = _SCOPE_NOUNS[scope]
    parts = []
    if added:
        count = f"1 {_SCOPE_SINGULAR[scope]}" if len(added) == 1 else f"{len(added)} {noun}"
        parts.append(
            f"Added {count} by {artist}:\n" + "\n".join(f"- {_describe(r)}" for r in added)
        )
    if failed:
        parts.append(
            "Couldn't add:\n" + "\n".join(f"- {r.title} - {why}" for r, why in failed)
        )
    if owned:
        parts.append(_owned_line(_unique_titles(owned)))
    if skipped:
        parts.append("Skipped because you may already have them: " + ", ".join(skipped))
    if later:
        parts.append(f"{len(later)} more not added yet; ask again to get them.")
    if not to_add and not failed:
        what = "the newest release" if scope == "newest" else f"every {_SCOPE_SINGULAR[scope]}"
        if skipped:
            # Not a claim of owning everything: some were only probably owned.
            parts.insert(0, f"Nothing new to add by {artist}.")
        else:
            parts.insert(0, f"You already have {what} by {artist} that's on the tracker.")
    return WorkflowResult(reply="\n\n".join(parts), steps=steps)


MIN_SEED_CONFIDENCE = 0.5
MIN_SAME_ARTIST = 0.6
MIN_NEW_TO_THEM = 0.5
MAX_OPTIONS = 250
MAX_CANDIDATES = 60
MAX_ROUNDS = 4
LOOKUP_CONCURRENCY = 4
# Below this a score answer means "does not fit".
MIN_FIT = 1.0

_NOT_AN_ALBUM = (
    "greatest hits", "best of", "collection", "anthology", "remix", "live at",
    "(live", " - single", "deluxe sampler",
)
# Words of the request itself. A span containing one is a phrase of the
# request ("couple albums like Mordechai but heavier"), not a name, and
# offering those makes the choice between them close to a coin toss.
REQUEST_WORDS = {
    "find", "give", "throw", "show", "gimme", "recommend", "suggest", "added",
    "some", "couple", "few", "something", "any", "more", "other", "another",
    "album", "like", "but", "than", "heavier", "lighter", "softer", "slower",
    "faster", "older", "newer", "similar", "that", "this", "it", "at",
    "to", "without", "is", "are", "was", "i", "you", "be",
}
MAX_SEED_WORDS = 4
_WANTS_NEW = re.compile(r"\bnew\b", re.IGNORECASE)
_ADDED = re.compile(r"^Added '(.*)', confirmed")


@dataclass
class Candidate:
    artist: str
    album: str
    known: bool = False

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.album}"

    @property
    def ref(self) -> AlbumRef:
        return AlbumRef(artist=self.artist, title=self.album)


def _looks_like_album(name: str) -> bool:
    lowered = name.casefold().strip()
    return not (any(word in lowered for word in _NOT_AN_ALBUM) or lowered.endswith(" ep"))


def _name_spans(text: str, stopwords: set[str] = STOPWORDS) -> list[str]:
    """Spans of the text that could be an artist, album or tag name."""
    return [
        span
        for span in candidate_spans(text, max_words=MAX_SEED_WORDS, stopwords=stopwords)
        if not REQUEST_WORDS & {w.casefold() for w in span.split()}
    ]


def _seed_options(text: str, state: dict, vocab: list[str]) -> list[str]:
    """Message spans, spans of the conversation it refers to, and tags that may fit."""
    earlier = state.get("earlier", [])
    recent = [
        next((t["text"] for t in reversed(earlier) if t["from"] == who), "")
        for who in ("bot", "user")
    ]
    tags = vocabulary_matches(text, vocab)
    spans = _name_spans(text)[: max(MAX_OPTIONS - len(tags), 0)]
    spoken = [span for entry in recent for span in _name_spans(entry)]
    spoken = spoken[: max(MAX_OPTIONS - len(spans) - len(tags), 0)]
    options: dict[str, str] = {}
    for span in [*spans, *spoken, *tags]:
        options.setdefault(span.casefold(), span)
    return list(options.values())[:MAX_OPTIONS]


def _score(decision, key: str) -> float | None:
    if decision is None:
        return None
    try:
        return float(decision.answers[key]["score"])
    except (KeyError, TypeError, ValueError):
        return None


async def _gather_limited(coros: list, default) -> list:
    """Run lookups four at a time; one that fails counts as `default`."""
    semaphore = asyncio.Semaphore(LOOKUP_CONCURRENCY)

    async def one(coro):
        async with semaphore:
            try:
                return await coro
            except Exception as e:
                logger.warning(f"Last.fm lookup failed: {e}")
                return default

    return await asyncio.gather(*[one(c) for c in coros])


async def _candidates(
    lastfm: LastFMClient, seed: str, seed_type: str, same_artist: bool
) -> tuple[list[Candidate], str, str | None]:
    """Candidate albums from Last.fm, popular first: (candidates, seed artist or tag, seed album)."""
    pairs: list[tuple[str, str]] = []
    exclude_album: str | None = None
    artist = seed
    if seed_type == "tag":
        albums, artists = await asyncio.gather(
            lastfm.get_tag_top_albums(seed, limit=50), lastfm.get_tag_top_artists(seed, limit=15)
        )
        by_artist = await _gather_limited(
            [lastfm.get_artist_albums(a.name, limit=2) for a in artists], []
        )
        direct = [(a.artist, a.name) for a in albums if a.artist]
        indirect = [
            (a.artist or similar.name, a.name)
            for similar, found in zip(artists, by_artist)
            for a in found
        ]
        pairs = [
            pair
            for two in zip_longest(direct, indirect)
            for pair in two
            if pair is not None
        ]
        return _dedupe(pairs), seed, None

    if seed_type == "album":
        found = await lastfm.search_album(seed)
        if found is None or not found.artist or fuzz.partial_ratio(seed.casefold(), found.name.casefold()) < 80:
            return [], seed, None
        artist, exclude_album = found.artist, found.name
    else:
        correction = await lastfm.correct_artist(seed)
        artist = correction.name if correction else seed

    if same_artist:
        albums = await lastfm.get_artist_albums(artist, limit=20)
        pairs = [(artist, a.name) for a in albums]
    else:
        similar = await lastfm.get_similar_artists(artist, limit=15)
        seeds = {artist.casefold(), seed.casefold()}
        names = [a.name for a in similar if a.name.casefold() not in seeds]
        found_albums = await _gather_limited(
            [lastfm.get_artist_albums(name, limit=3) for name in names], []
        )
        pairs = [(name, a.name) for name, albums in zip(names, found_albums) for a in albums]
    candidates = _dedupe(pairs)
    if exclude_album:
        keep = _norm(exclude_album)
        candidates = [
            c for c in candidates
            if not (c.artist.casefold() == artist.casefold() and _norm(c.album) == keep)
        ]
    return candidates, artist, exclude_album


def _dedupe(pairs: list[tuple[str, str]]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for artist, album in pairs:
        key = (artist.casefold(), _norm(album))
        if key in seen or not _norm(album) or not _looks_like_album(album):
            continue
        seen.add(key)
        out.append(Candidate(artist, album))
    return out[:MAX_CANDIDATES]


def _is_dry_run(line: str) -> bool:
    return "DRY_RUN" in line


def _listed(candidate: Candidate, line: str) -> str:
    """"Album (year) by Artist", with the year when the torrent name says it."""
    match = _ADDED.match(line)
    release = parse_release(match.group(1)) if match else None
    year = f" ({release.year})" if release else ""
    return f"{candidate.album}{year} by {candidate.artist}"


async def recommend(
    text: str,
    context: TorrentContext,
    *,
    count: int = 3,
    earlier: list[dict] | None = None,
    run_id: str | None = None,
) -> WorkflowResult | None:
    """Add `count` albums that fit an open-ended music request, or None to defer to the agent."""
    steps: list[Step] = []

    def record(stage: str, started: float, args: dict, result: str) -> None:
        steps.append(
            Step(
                kind="tool",
                name=f"recommend/{stage}",
                seconds=time.perf_counter() - started,
                args=args,
                result=result,
            )
        )

    # a. What to base it on.
    started = time.perf_counter()
    state = route_state(text, earlier)
    options = _seed_options(text, state, await tag_vocabulary())
    decision = await decide(
        RECOMMEND_SEED_NAME, state, recommend_seed_questions(options), run_id=run_id
    )
    if decision is None:
        return None
    seed, seed_p = decision.choice("seed")
    seed_type, type_p = decision.choice("seed_type")
    same_artist = decision.noul("same_artist")
    new_to_them = decision.noul("new_to_them")
    record(
        "seed", started, {"options": len(options)},
        f"seed={seed!r} ({seed_p:.2f}) type={seed_type} ({type_p:.2f}) "
        f"same_artist={same_artist:.2f} new_to_them={new_to_them:.2f}",
    )
    if seed == "none" or seed_p < MIN_SEED_CONFIDENCE or seed_type == "none":
        return None
    same_artist = same_artist >= MIN_SAME_ARTIST and seed_type in ("artist", "album")
    want_new = (new_to_them >= MIN_NEW_TO_THEM or bool(_WANTS_NEW.search(text))) and not same_artist

    # b. Candidates from Last.fm.
    started = time.perf_counter()
    async with LastFMClient() as lastfm:
        candidates, subject, _ = await _candidates(lastfm, seed, seed_type, same_artist)
    record(
        "candidates", started, {"seed": seed, "type": seed_type},
        f"{len(candidates)} candidates: " + "; ".join(c.label for c in candidates[:10]),
    )
    if not candidates:
        return None

    # c. Drop what they own; prefer artists they have never listened to.
    started = time.perf_counter()
    library = await _library_lookup([c.artist for c in candidates])
    remaining: list[Candidate] = []
    for c in candidates:
        owned = library.get(c.artist)
        if owned is not None and _ownership(owned, c.album) != "":
            continue
        c.known = bool(owned)
        remaining.append(c)
    if want_new:
        fresh = [c for c in remaining if not c.known]
        if len(fresh) >= count * 2:
            remaining = fresh
    record(
        "library", started, {"artists": len(library), "want_new": want_new},
        f"{len(remaining)} of {len(candidates)} candidates left",
    )
    if not remaining:
        return None

    # d. Rank by how well each fits the request.
    started = time.perf_counter()
    keys = [f"c{i}" for i in range(len(remaining))]
    fit_state: dict = {"request": text}
    if "earlier" in state:
        fit_state["earlier"] = state["earlier"]
    fit_state["candidates"] = {k: c.label for k, c in zip(keys, remaining)}
    fit = await decide(
        RECOMMEND_FIT_NAME, fit_state, recommend_fit_questions(keys), run_id=run_id
    )
    scores = {k: _score(fit, k) for k in keys}
    order = sorted(
        range(len(remaining)),
        key=lambda i: (
            want_new and remaining[i].known,
            -(scores[keys[i]] if scores[keys[i]] is not None else 0.0),
            i,
        ),
    )
    ranked = [remaining[i] for i in order]
    if fit is not None:
        fitting = [c for i, c in zip(order, ranked) if (scores[keys[i]] or 0.0) >= MIN_FIT]
        if len(fitting) >= count:
            ranked = fitting
    record(
        "rank", started, {"candidates": len(remaining)},
        "; ".join(
            f"{remaining[i].label} = "
            + (f"{scores[keys[i]]:.2f}" if scores[keys[i]] is not None else "?")
            for i in order[:10]
        )
        if fit is not None
        else "no ranking; Last.fm order",
    )

    # e. One album per artist, unless they asked for one artist's albums.
    picks: list[Candidate] = []
    used: set[str] = set()
    for c in ranked:
        if not same_artist and c.artist.casefold() in used:
            continue
        used.add(c.artist.casefold())
        picks.append(c)

    # f. Download in rounds, replacing what is not available.
    media = _media_from(text)
    added: list[str] = []
    not_found: list[str] = []
    other_failed: list[str] = []
    dry_run: list[str] = []
    index = 0
    for round_number in range(1, MAX_ROUNDS + 1):
        batch = picks[index : index + count - len(added)]
        if not batch:
            break
        index += len(batch)
        started = time.perf_counter()
        lines = await download_many([c.ref for c in batch], context, media=media)
        for c, line in zip(batch, lines):
            name = f"{c.album} by {c.artist}"
            if line.startswith("Added"):
                added.append(_listed(c, line))
            elif _is_dry_run(line):
                dry_run.append(name)
            elif line.startswith("NOT FOUND"):
                not_found.append(name)
            elif line.startswith("NOT ADDED"):
                other_failed.append(f"{name} ({_reason(line)})")
        record(
            f"download {round_number}", started, {"albums": [c.label for c in batch]},
            " | ".join(lines),
        )
        if len(added) >= count or dry_run:
            break

    # g. Reply.
    if dry_run and not added:
        shown = ", ".join(dry_run)
        return WorkflowResult(
            reply=f"Nothing added: dry run is on. Would have added: {shown}.", steps=steps
        )
    if not added:
        why = []
        if not_found:
            why.append("no FLAC torrent for " + ", ".join(not_found[:5]))
        if other_failed:
            why.append("couldn't add " + ", ".join(other_failed[:3]))
        return WorkflowResult(
            reply="Nothing added: " + "; ".join(why or ["nothing available"]) + ".", steps=steps
        )
    if same_artist:
        phrase = f"by {subject}"
    elif seed_type == "tag":
        phrase = f"for {seed}"
    else:
        phrase = f"like {subject if seed_type == 'artist' else seed}"
    noun = "album" if len(added) == 1 else "albums"
    parts = [f"Added {len(added)} {noun} {phrase}:\n" + "\n".join(f"- {a}" for a in added)]
    if len(added) < count:
        why = []
        if not_found:
            why.append("no FLAC torrent for " + ", ".join(not_found[:5]))
        if other_failed:
            why.append("couldn't add " + ", ".join(other_failed[:3]))
        parts.append(f"Only found {len(added)} of {count}: " + "; ".join(why or ["ran out of candidates"]) + ".")
    return WorkflowResult(reply="\n".join(parts), steps=steps)


# ---------------------------------------------------------------------------
# Specific releases: "get me Mordechai by Khruangbin", "download Loveless and Souvlaki".
# ---------------------------------------------------------------------------

MAX_SPECIFIC_SPANS = 120
MAX_SPECIFIC_ALBUMS = 5
MAX_PICK_CANDIDATES = 12
MIN_ALBUM_SPAN = 0.5
MIN_NAMED = 0.5
# A longer span (with an edition word, say) beats a shorter one inside it unless
# the shorter is clearly the better answer.
LONGER_SPAN_MARGIN = 0.1
SURE_PICK = 0.75
UNSURE_PICK_TOTAL = 0.5
MIN_OPTION_P = 0.1
MAX_OPTIONS_OFFERED = 3
MIN_OFFER_SIMILARITY = 45
TRACK_ALBUMS = 3


@dataclass
class _Release:
    artist: str
    title: str
    year: int | None = None
    # Evidence for the picker, e.g. "has the track 'Fast City'" or the artist's
    # Last.fm listeners: with no artist named, "in rainbowz" means Radiohead, not
    # a 2k-listener act whose title happens to be spelled that way.
    note: str = ""
    listeners: int | None = None
    track: str | None = None  # a song they named that is on this release

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title}" + (f" ({self.year})" if self.year else "")

    @property
    def choice(self) -> Choice:
        return Choice(label=f"{self.artist} - {self.title}", artist=self.artist, title=self.title)


@dataclass
class _Wanted:
    span: str  # what they asked for, as written
    artist: str | None
    releases: list[_Release]
    track: str | None = None
    by_title: bool = True  # False when found through a song: titles say nothing about the span
    written: str = ""  # the span with neighbouring title words, see _as_written


def _pick_view(r: "_Release") -> dict:
    view: dict = {"artist": r.artist, "title": r.title}
    if r.year:
        view["year"] = r.year
    if r.listeners:
        view["artist_listeners"] = r.listeners
    if r.track:
        view["has_track"] = r.track
    return view


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _similarity(span: str, title: str) -> int:
    a, b = span.casefold(), title.casefold()
    return (fuzz.ratio(a, b) + fuzz.token_set_ratio(a, b)) // 2


def _add_release(found: dict[tuple[str, str], _Release], artist: str | None, title: str, year: int | None = None) -> None:
    if not artist or not _norm(title):
        return
    key = (artist.casefold(), _norm(title))
    existing = found.get(key)
    if existing is None:
        found[key] = _Release(artist, title, year)
    elif existing.year is None and year:
        existing.year = year


# Titles often begin with these ("In Rainbows", "The Bends"); for artist and tag
# names they are noise, but here a span starting with one must stay possible.
TITLE_STARTERS = {"in", "the", "a", "an"}


def _title_spans(text: str) -> list[str]:
    return [*_name_spans(text), *_name_spans(text, STOPWORDS - TITLE_STARTERS)]


def _specific_spans(text: str, state: dict) -> list[str]:
    """Spans of the message, then of the last two turns it may refer to."""
    spans = _title_spans(text)
    for entry in state.get("earlier", [])[-2:]:
        spans.extend(_title_spans(entry["text"]))
    unique: dict[str, str] = {}
    for span in spans:
        unique.setdefault(span.casefold(), span)
    return list(unique.values())[:MAX_SPECIFIC_SPANS]


# A longer span this plausible is kept as what they wrote, for the picker.
MIN_WRITTEN_SPAN = 0.2


def _select_albums(spans: list[str], decision) -> list[tuple[str, float]]:
    """Spans judged to be albums, one per mention, as (span, probability) in message order."""
    scored = [(i, span, decision.noul(f"album_{i}")) for i, span in enumerate(spans)]
    chosen = [(i, span, p) for i, span, p in scored if p >= MIN_ALBUM_SPAN]
    dropped: set[int] = set()
    for i, short, p_short in chosen:
        for j, long, p_long in chosen:
            if len(long) > len(short) and short.casefold() in long.casefold():
                dropped.add(i if p_long >= p_short - LONGER_SPAN_MARGIN else j)
    kept = [(i, span, p) for i, span, p in chosen if i not in dropped]
    kept = sorted(kept, key=lambda item: -item[2])[:MAX_SPECIFIC_ALBUMS]
    return [(span, p) for _, span, p in sorted(kept)]


def _as_written(span: str, spans: list[str], decision) -> str:
    """The span with the words around it that might belong to the title.

    For "get me in rainbowz" the extract call prefers "rainbowz", but handed
    "rainbowz" the picker matches the title literally and chooses an act
    called RaINBOWZ; "in rainbowz" points it at Radiohead. The longest span
    containing this one that is still plausibly a title is what they wrote.
    """
    best = span
    for i, other in enumerate(spans):
        if (
            len(other) > len(best)
            and span.casefold() in other.casefold()
            and decision.noul(f"album_{i}") >= MIN_WRITTEN_SPAN
        ):
            best = other
    return best


async def _safe(semaphore: asyncio.Semaphore, coro, default, what: str):
    async with semaphore:
        try:
            return await coro
        except Exception as e:
            logger.warning(f"{what} failed: {e}")
            return default


async def _album_releases(
    span: str,
    artist: str | None,
    context: TorrentContext,
    lastfm: LastFMClient,
    semaphore: asyncio.Semaphore,
) -> list[_Release]:
    """Releases that the words of a request could mean, from Last.fm and the tracker."""
    canonical = artist
    if artist:
        correction = await _safe(semaphore, lastfm.correct_artist(artist), None, "Last.fm correction")
        canonical = correction.name if correction else artist

    async def artist_side() -> list[_Release]:
        if not canonical:
            return []
        found: dict[tuple[str, str], _Release] = {}
        info, albums = await asyncio.gather(
            _safe(semaphore, lastfm.get_album_info(canonical, span), None, "Last.fm album info"),
            _safe(semaphore, lastfm.get_artist_albums(canonical, limit=50), [], "Last.fm artist albums"),
        )
        if info is not None:
            _add_release(found, info.artist or canonical, info.name)
        fuzzy = [
            (max(fuzz.token_set_ratio(span, a.name), fuzz.partial_ratio(span.casefold(), a.name.casefold())), a)
            for a in albums
            if fuzz.token_set_ratio(span, a.name) >= 70
            or fuzz.partial_ratio(span.casefold(), a.name.casefold()) >= 80
        ]
        for _, a in sorted(fuzzy, key=lambda item: -item[0])[:5]:
            _add_release(found, a.artist or canonical, a.name)
        return list(found.values())

    async def search_side() -> list[_Release]:
        found: dict[tuple[str, str], _Release] = {}
        albums = await _safe(semaphore, lastfm.search_albums(span, limit=8), [], "Last.fm album search")
        albums = [a for a in albums if a.artist]
        if canonical:
            names = {canonical.casefold(), (artist or canonical).casefold()}
            matching = [
                a for a in albums
                if any(fuzz.partial_ratio(n, a.artist.casefold()) >= 80 for n in names)
            ]
            # The artist may be what they misremembered, so keep two others.
            albums = matching + [a for a in albums if a not in matching][:2]
        for a in albums:
            _add_release(found, a.artist, a.name)
        return list(found.values())

    async def tracker_side() -> list[_Release]:
        query = f"{canonical} {span}" if canonical else span

        async def search():
            async with QBittorrentClient() as qclient:
                return await qclient.search(query, BTCategory.Music)

        response = await _safe(semaphore, search(), None, "Tracker search")
        if response is None:
            return []
        flac = [r for r in (response.results or []) if "FLAC" in r.fileName]
        for r in flac:
            context.search_results[r.fileName] = r
        parsed = best_versions([r.fileName for r in flac])
        parsed.sort(key=lambda r: -_similarity(span, r.title))
        found: dict[tuple[str, str], _Release] = {}
        for r in parsed[:5]:
            _add_release(found, r.artist, r.title, r.year)
        return list(found.values())

    sides = await asyncio.gather(artist_side(), search_side(), tracker_side())
    found: dict[tuple[str, str], _Release] = {}
    for side in sides:
        for r in side:
            _add_release(found, r.artist, r.title, r.year)
    # Most like the request first, so the cap drops the least likely.
    ranked = sorted(found.values(), key=lambda r: -_similarity(span, r.title))
    return ranked[:MAX_PICK_CANDIDATES]


async def _track_releases(
    track: str, artist: str | None, lastfm: LastFMClient, semaphore: asyncio.Semaphore
) -> list[_Release]:
    """The albums that songs matching `track` appear on."""
    songs = await _safe(semaphore, lastfm.search_tracks(track, artist, limit=5), [], "Last.fm track search")
    albums = await asyncio.gather(
        *[
            _safe(semaphore, lastfm.get_track_album(a, t), None, "Last.fm track info")
            for a, t in songs[:TRACK_ALBUMS]
        ]
    )
    found: dict[tuple[str, str], _Release] = {}
    for (song_artist, _), album in zip(songs, albums):
        if album is not None:
            _add_release(found, album.artist or song_artist, album.name)
    releases = list(found.values())
    for release in releases:
        release.note = f"has a track called '{track}'"
        release.track = track
    return releases


def _top_options(probabilities: dict, keys: list[str]) -> list[tuple[str, float]]:
    scored = []
    for key in keys:
        try:
            scored.append((key, float(probabilities.get(key, 0.0))))
        except (TypeError, ValueError):
            scored.append((key, 0.0))
    return sorted(scored, key=lambda item: -item[1])


def _by_similarity(want: _Wanted, likelihood: dict[int, float] | None = None) -> list[_Release]:
    """Candidates closest to the request in spelling, the model's pick breaking ties."""
    if not want.by_title:
        return list(want.releases)
    likelihood = likelihood or {}
    index = {id(r): i for i, r in enumerate(want.releases)}
    ranked = sorted(
        want.releases,
        key=lambda r: (-_similarity(want.span, r.title), -likelihood.get(index[id(r)], 0.0)),
    )
    return [r for r in ranked if _similarity(want.span, r.title) >= MIN_OFFER_SIMILARITY]


def _offer(want: _Wanted, options: list[_Release]) -> Pending | None:
    """A did-you-mean, never with fewer than two options."""
    if len(options) >= 2:
        return Pending(want.span, [r.choice for r in options[:MAX_OPTIONS_OFFERED]])
    return None


async def _pick(
    want: _Wanted, text: str, earlier: list[dict] | None, run_id: str | None
) -> tuple[_Release | Pending | None, str]:
    """Settle one wanted release: resolved, pending or None (not found), plus a log line."""
    if not want.releases:
        return None, "no candidates"
    keys = [f"r{i}" for i in range(len(want.releases))]
    state = route_state(text, earlier) | {
        "wanted": {
            "title": want.written or want.span,
            "artist": want.artist,
            "track": want.track,
        },
        # Structured, not "Artist - Title [notes]": Jev weighs a listeners
        # number it can see far more than one buried in a label (0.82 vs 0.14
        # for In Rainbows over a 14-listener "Rainbowz").
        "candidates": {k: _pick_view(r) for k, r in zip(keys, want.releases)},
    }
    decision = await decide(SPECIFIC_PICK_NAME, state, specific_pick_questions(keys), run_id=run_id)
    if decision is None:
        matching = [r for r in want.releases if _title_matches(want.span, r.title)]
        if len(matching) == 1 and want.by_title:
            return matching[0], "no decision; the one title that matches"
        return _offer(want, _by_similarity(want)), "no decision; fuzzy"

    pick, p = decision.choice("pick")
    ranked = _top_options(decision.answers["pick"].get("probabilities", {}), keys)
    shown = ", ".join(f"{want.releases[keys.index(k)].label}={q:.2f}" for k, q in ranked[:3])
    log = f"pick={pick} ({p:.2f}); {shown}"
    if pick != "none" and pick in keys and p >= SURE_PICK:
        return want.releases[keys.index(pick)], log
    top = ranked[:MAX_OPTIONS_OFFERED]
    if sum(q for _, q in top) >= UNSURE_PICK_TOTAL:
        plausible = [(k, q) for k, q in top if q >= MIN_OPTION_P]
        if len(plausible) >= 2:
            return Pending(want.span, [want.releases[keys.index(k)].choice for k, _ in plausible]), log
        if plausible and plausible[0][1] >= MIN_NAMED:
            return want.releases[keys.index(plausible[0][0])], log
        return None, log
    likelihood = {keys.index(k): q for k, q in ranked}
    return _offer(want, _by_similarity(want, likelihood)), log


def _label_of(line: str, ref: AlbumRef | None) -> str:
    if ref is not None:
        return f"{ref.artist} - {ref.title}"
    match = re.search(r"'(.*?)'(?=[\s.,]|$)", line)
    name = match.group(1) if match else line
    release = parse_release(name)
    return f"{release.artist} - {release.title}" if release else name.split(" [")[0]


def describe_lines(lines: list[str], refs: list[AlbumRef] | None = None) -> str:
    """Plain wording for download_many's result lines; `refs` are the albums, in order."""
    added: list[str] = []
    other: list[str] = []
    for i, line in enumerate(lines):
        ref = refs[i] if refs is not None and i < len(refs) else None
        if line.startswith("Added"):
            added.append(_label_of(line, ref))
            continue
        label = _label_of(line, ref)
        if line.startswith("OWNED"):
            other.append(f"You already have {label}.")
        elif line.startswith("NOT FOUND"):
            other.append(f"Couldn't find a FLAC of {label} on the tracker.")
        elif line.startswith("NOT ADDED"):
            other.append(f"Couldn't add {label}: {_reason(line)}")
        else:
            other.append(line)
    parts = []
    if len(added) == 1:
        parts.append(f"Added {added[0]}.")
    elif added:
        parts.append("Added:\n" + "\n".join(f"- {a}" for a in added))
    return "\n".join([*parts, *other])


async def specific(
    text: str,
    context: TorrentContext,
    *,
    earlier: list[dict] | None = None,
    run_id: str | None = None,
) -> WorkflowResult | None:
    """Add the particular releases a message names, or ask "did you mean", or None to defer."""
    steps: list[Step] = []

    def record(stage: str, started: float, args: dict, result: str) -> None:
        steps.append(
            Step(
                kind="tool",
                name=f"specific/{stage}",
                seconds=time.perf_counter() - started,
                args=args,
                result=result,
            )
        )

    # a. What is wanted, from which artist.
    started = time.perf_counter()
    state = route_state(text, earlier)
    spans = _specific_spans(text, state)
    if not spans:
        return None
    decision = await decide(
        SPECIFIC_EXTRACT_NAME,
        state | {"spans": {f"album_{i}": span for i, span in enumerate(spans)}},
        specific_extract_questions(spans),
        run_id=run_id,
    )
    if decision is None:
        return None

    # b. Albums, artist and song.
    albums = _select_albums(spans, decision)
    artist_choice, artist_p = decision.choice("artist")
    track_choice, track_p = decision.choice("track")
    artist = artist_choice if artist_choice != "none" and artist_p >= MIN_NAMED else None
    track = track_choice if track_choice != "none" and track_p >= MIN_NAMED else None
    record(
        "extract", started, {"spans": len(spans)},
        f"albums={[(s, round(p, 2)) for s, p in albums]} artist={artist!r} ({artist_p:.2f}) "
        f"track={track!r} ({track_p:.2f})",
    )
    # The artist's own name is not an album; a self-titled album asked for
    # alone is the one case this loses, and the agent still gets it.
    if artist and (track or len(albums) > 1):
        albums = [(s, p) for s, p in albums if s.casefold() != artist.casefold()]
    if not albums and not track:
        return None

    # c. Candidates for each wanted release.
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(LOOKUP_CONCURRENCY)
    wanted: list[_Wanted] = []
    async with LastFMClient() as lastfm:
        if albums:
            # The same words named as artist and as album: one of them is wrong, so
            # search the album without trusting the artist.
            credited = [
                None if artist and span.casefold() == artist.casefold() else artist
                for span, _ in albums
            ]
            found = await asyncio.gather(
                *[
                    _album_releases(span, by, context, lastfm, semaphore)
                    for (span, _), by in zip(albums, credited)
                ]
            )
            wanted = [
                _Wanted(span, by, releases, written=_as_written(span, spans, decision))
                for (span, _), by, releases in zip(albums, credited, found)
            ]
        else:
            releases = await _track_releases(track, artist, lastfm, semaphore)
            wanted = [_Wanted(f"the album with {track}", artist, releases, track, by_title=False)]
    record(
        "candidates", started, {"wanted": [w.span for w in wanted]},
        " | ".join(f"{w.span}: " + "; ".join(r.label for r in w.releases) for w in wanted),
    )

    # Popularity, so a misspelling resolves to the release people mean.
    artists = list(dict.fromkeys(r.artist for w in wanted for r in w.releases))
    if artists:
        started = time.perf_counter()
        async with LastFMClient() as lastfm:
            infos = await asyncio.gather(
                *[_safe(semaphore, lastfm.get_artist_info(a), None, "Last.fm artist info") for a in artists]
            )
        listeners = {a: i.listeners for a, i in zip(artists, infos) if i and i.listeners}
        for w in wanted:
            for r in w.releases:
                if r.artist in listeners:
                    r.listeners = listeners[r.artist]
                    popularity = f"artist has {_compact(listeners[r.artist])} Last.fm listeners"
                    r.note = f"{r.note}; {popularity}" if r.note else popularity
        record("popularity", started, {"artists": artists}, str(listeners))

    # d. Settle each one.
    started = time.perf_counter()
    outcomes = await asyncio.gather(*[_pick(w, text, earlier, run_id) for w in wanted])
    record(
        "pick", started, {"wanted": [w.span for w in wanted]},
        " | ".join(f"{w.span}: {log}" for w, (_, log) in zip(wanted, outcomes)),
    )
    resolved: list[_Release] = []
    pending: list[Pending] = []
    missing: list[str] = []
    for w, (outcome, _) in zip(wanted, outcomes):
        if isinstance(outcome, _Release):
            resolved.append(outcome)
        elif isinstance(outcome, Pending):
            pending.append(outcome)
        else:
            missing.append(w.span)
    if not resolved and not pending:
        return None

    # e. Download what is settled.
    parts: list[str] = []
    if resolved:
        started = time.perf_counter()
        refs = [AlbumRef(artist=r.artist, title=r.title) for r in resolved]
        lines = await download_many(refs, context, media=_media_from(text))
        record("download", started, {"albums": [r.label for r in resolved]}, " | ".join(lines))
        parts.append(describe_lines(lines, refs))

    # f. Reply.
    parts.extend(f'Couldn\'t find anything called "{span}".' for span in missing)
    parts.extend(f'Did you mean one of these for "{p.wanted}"?' for p in pending)
    return WorkflowResult(reply="\n".join(parts), steps=steps, pending=pending)


# --- Orpheus account ---------------------------------------------------------

MIN_MAX_CONFIDENCE = 0.6
_TOKEN_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a couple": 2, "couple": 2, "a few": 3, "few": 3,
}
_TOKEN_COUNT = re.compile(
    r"\b(\d{1,3}|" + "|".join(sorted((re.escape(w) for w in _TOKEN_WORDS), key=len, reverse=True))
    + r")\s+(?:of\s+)?(?:free\s*-?\s*leech\s+)?(?:tokens?|fl)\b",
    re.IGNORECASE,
)
SHOP_URL = "https://orpheus.network/bonus.php"


def _tokens(n: int) -> str:
    return f"{n} freeleech token" + ("" if n == 1 else "s")


def token_count(text: str) -> int | None:
    """The number of tokens the message names, if it does."""
    match = _TOKEN_COUNT.search(text)
    if not match:
        return None
    word = match.group(1).lower()
    return int(word) if word.isdigit() else _TOKEN_WORDS[word]


def status_reply(stats: AccountStats) -> str:
    per_day = round(stats.bonus_per_hour * 24)
    text = (
        f"You have {_tokens(stats.tokens)} and {stats.bonus_points:,} bonus points "
        f"(+{stats.bonus_per_hour:.0f}/hour, about {per_day:,}/day). "
        f"Ratio {stats.ratio:.2f} (you need {stats.required_ratio:.2f})."
    )
    if stats.ratio < stats.required_ratio + 0.05:
        text += "\n⚠️ That's close to ratio watch: use tokens on big downloads."
    return text


def _orpheus_failure(e: Exception) -> str:
    message = str(e)
    prefix = "Couldn't reach Orpheus: "
    return message if message.startswith(prefix) else prefix + message


async def account(text: str, *, run_id: str | None = None) -> WorkflowResult | None:
    """Answer an Orpheus account question, or offer a token purchase to confirm."""
    started = time.perf_counter()
    decision = await decide(ACCOUNT_NAME, {"message": text}, account_questions(), run_id=run_id)
    if decision is not None:
        intent, _ = decision.choice("intent")
        wants_max = decision.noul("max") >= MIN_MAX_CONFIDENCE
    else:
        intent = "buy" if "buy" in text.lower() else "status"
        wants_max = "as many" in text.lower()
    count = token_count(text)
    if count is None and not wants_max:
        count = 1

    def done(reply: str, confirm: Confirmation | None = None) -> WorkflowResult:
        step = Step(
            "tool",
            "account",
            time.perf_counter() - started,
            args={"intent": intent, "count": count},
            result=reply,
        )
        return WorkflowResult(reply=reply, steps=[step], confirm=confirm)

    try:
        async with OrpheusClient() as client:
            stats, authkey = await client.account()
            status = status_reply(stats)
            if intent != "buy":
                return done(status)
            if not Config.ORPHEUS_SESSION_COOKIE:
                return done(
                    f"{status}\n\nI can't buy from here without your Orpheus session cookie "
                    f"(ORPHEUS_SESSION_COOKIE). You can buy them at {SHOP_URL}"
                )
            items = await client.token_shop()
    except OrpheusError as e:
        return done(_orpheus_failure(e))

    plan = plan_purchase(items, count, stats.bonus_points)
    if not plan:
        cheapest = plan_purchase(items, count, 10**12)
        if count is None:
            priced = [i for i in items if i.tokens > 0]
            if not priced:
                return done(f"{status}\n\nThe shop has no freeleech tokens on sale right now.")
            item = min(priced, key=lambda i: i.price)
            return done(
                f"{status}\n\nYou can't afford any tokens yet: the cheapest is "
                f"{item.price:,} points for {item.tokens}."
            )
        if not cheapest:
            return done(
                f"{status}\n\nThe shop can't make up exactly {_tokens(count)} right now."
            )
        price = sum(i.price for i in cheapest)
        return done(
            f"{status}\n\nYou can't afford {_tokens(count)}: the cheapest is "
            f"{price:,} points for {count}."
        )

    total_tokens = sum(i.tokens for i in plan)
    total_price = sum(i.price for i in plan)
    left = stats.bonus_points - total_price
    prompt = (
        f"Buy {_tokens(total_tokens)} for {total_price:,} bonus points? "
        f"You'd have {left:,} left."
    )

    async def action() -> str:
        bought = 0
        try:
            async with OrpheusClient() as buyer:
                for item in plan:
                    await buyer.buy(item.label, authkey)
                    bought += item.tokens
        except OrpheusError as e:
            if bought == 0:
                return _orpheus_failure(e)
            return f"Bought {bought} of {total_tokens} before Orpheus said: {e}"
        try:
            async with OrpheusClient() as reader:
                after, _ = await reader.account()
        except OrpheusError:
            return f"Bought {bought} token{'' if bought == 1 else 's'}."
        return (
            f"Bought {bought} token{'' if bought == 1 else 's'}. You now have "
            f"{after.tokens} tokens and {after.bonus_points:,} bonus points."
        )

    return done(f"{status}\n\n{prompt}", Confirmation(prompt=prompt, action=action))
