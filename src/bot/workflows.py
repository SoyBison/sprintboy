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
from dataclasses import dataclass
from itertools import zip_longest

from thefuzz import fuzz

from bot.agent import AgentResult, Step
from bot.decide import decide
from bot.llm import Message, assistant
from bot.netcode import BTCategory, LastFMClient, QBittorrentClient
from bot.questions import (
    DISCOGRAPHY_QUESTIONS_NAME,
    RECOMMEND_FIT_NAME,
    RECOMMEND_SEED_NAME,
    discography_questions,
    recommend_fit_questions,
    recommend_seed_questions,
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


def candidate_spans(text: str, max_words: int = 6) -> list[str]:
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
            if chunk[0].casefold() in STOPWORDS or chunk[-1].casefold() in STOPWORDS:
                continue
            span = " ".join(chunk)
            spans.setdefault(span.casefold(), span)
    return list(spans.values())[:MAX_SPANS]


@dataclass
class WorkflowResult:
    reply: str
    steps: list[Step]

    def to_agent_result(self, messages: list[Message]) -> AgentResult:
        return AgentResult(
            messages=[*messages, assistant(self.reply)],
            steps=self.steps,
            stopped="workflow",
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


def _name_spans(text: str) -> list[str]:
    """Spans of the text that could be an artist, album or tag name."""
    return [
        span
        for span in candidate_spans(text, max_words=MAX_SEED_WORDS)
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
