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

from thefuzz import fuzz

from bot.agent import AgentResult, Step
from bot.decide import decide
from bot.llm import Message, assistant
from bot.netcode import BTCategory, QBittorrentClient
from bot.questions import DISCOGRAPHY_QUESTIONS_NAME, discography_questions
from bot.releases import Release, _norm, best_versions, quality_key
from bot.tools import (
    TorrentContext,
    _add_one,
    _canonicalise,
    _library_lookup,
    _maybe_title,
    _ownership,
    _same_release,
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
