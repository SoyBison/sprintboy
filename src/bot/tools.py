import asyncio
import re
import logging
from typing import Union

import aiohttp
from langchain.tools import ToolRuntime, tool
from bot.netcode import (
    BTCategory,
    CanonicalRelease,
    LastFMClient,
    LastFMError,
    QBittorrentClient,
    SearchResult,
    PlexAPIClient,
    PLEX_CONTENT_TYPES,
    resolve_release,
)
from bot.decide import decide
from bot.releases import best_versions, parse_release, quality_key
from bot.questions import SAME_RELEASE_QUESTIONS, SAME_RELEASE_QUESTIONS_NAME
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from thefuzz import fuzz, process
from typing import Literal

# Whole-title similarity at or above this means the user already owns the album.
ALBUM_MATCH_THRESHOLD = 90
# One title containing the other at or above this is a maybe: a deluxe edition,
# a reissue, or a different album that happens to share a prefix. The agent is
# given these to judge rather than being told they are a collision.
ALBUM_CONTAINED_THRESHOLD = 90
# Decision-model probability bounds for "these two titles are the same release".
SAME_RELEASE_YES = 0.85
SAME_RELEASE_NO = 0.25
# Displayed result lines per search query (all results are still stored).
MAX_RESULTS_PER_QUERY = 25


@dataclass
class TorrentContext:
    """A simple class to keep track of known torrents on trackers and qBittorrent."""

    search_results: dict[str, SearchResult]
    internal_torrents: dict[str, Union[str, None]]
    torrent_types: set[BTCategory]


def _str_to_list(value):
    """Small models often send a bare string where a list is expected."""
    return [value] if isinstance(value, str) else value


class TorrentAddQuery(BaseModel):
    names: list[str]
    category: BTCategory

    @field_validator("names", mode="before")
    @classmethod
    def _wrap_names(cls, value):
        return _str_to_list(value)


class TorrentSearchQuery(BaseModel):
    queries: list[str]
    category: BTCategory
    media: Literal["CD", "SACD", "WEB", "Vinyl"] | None = None

    @field_validator("queries", mode="before")
    @classmethod
    def _wrap_queries(cls, value):
        return _str_to_list(value)


def _media_label(r, media: str | None) -> str:
    if media and r.media.casefold() != media.casefold():
        return f" (no {media} version)"
    if r.vinyl and not media:
        return " (vinyl only)"
    return ""


@tool(args_schema=TorrentSearchQuery)
async def search_for_torrent(
    queries: list[str],
    category: BTCategory,
    runtime: ToolRuntime[TorrentContext],
    media: str | None = None,
) -> str:
    """
    Perform one or more search queries on qBittorrent and return the results per query.
    You can and should pass several queries at once (for example one per album, as
    "Artist Album") instead of calling this tool repeatedly.
    This is not like google.
    It only returns results that match the query in a fuzzy REGEX.
    Do not include words like "discography" or "album" in your search, this tool only returns single album torrents, or single movies, or single episodes.
    It is best to only include album titles and artist names in your query.
    This tool can find Movies, Music, and TV shows.
    Do not include words like "BluRay" or "720p" or "1080p" in your query.
    Only set media when the user asks for a source (CD, SACD, WEB or Vinyl); by
    default the best version is picked and vinyl is never chosen.
    """
    runtime.context.torrent_types.add(category)
    # qBittorrent rejects more than 5 concurrent search jobs.
    semaphore = asyncio.Semaphore(3)

    async def _search_one(query: str) -> list[SearchResult]:
        async with semaphore:
            async with QBittorrentClient() as qclient:
                response = await qclient.search(query, category)
        found = response.results or []
        # filter out results that are not flacs
        if category == BTCategory.Music:
            found = [result for result in found if "FLAC" in result.fileName]
        return found

    outcomes = await asyncio.gather(
        *[_search_one(query) for query in queries], return_exceptions=True
    )

    sections = []
    for query, outcome in zip(queries, outcomes):
        if isinstance(outcome, BaseException):
            logging.warning(f"Search for '{query}' failed: {outcome}")
            sections.append(f"Search for '{query}' failed: {outcome}")
            continue
        if not outcome:
            sections.append(f"No results for '{query}'.")
            continue
        # Store results in runtime for later use
        for result in outcome:
            runtime.context.search_results[result.fileName] = result
        if category == BTCategory.Music:
            names = [result.fileName for result in outcome]
            ranked = best_versions(names, media)
            all_lines = [
                r.name + _media_label(r, media)
                for r in ranked
            ]
            parsed = {r.name for r in ranked}
            all_lines += [n for n in names if parse_release(n) is None and n not in parsed]
            lines = all_lines[:MAX_RESULTS_PER_QUERY]
            hidden = len(all_lines) - len(lines)
        else:
            lines = [result.fileName for result in outcome[:MAX_RESULTS_PER_QUERY]]
            hidden = len(outcome) - len(lines)
        if hidden > 0:
            lines.append(f"({hidden} more not shown; search more specifically)")
        sections.append(f"Results for '{query}':\n" + "\n".join(lines))
    if runtime.context.search_results:
        # Small models read a search result as the end of the job and reply
        # "Added ..." without ever adding; saying so here is cheaper than a retry.
        sections.append(
            "Nothing is downloading yet. Call add_torrent once with every name you "
            "picked, exactly as written above."
        )
    return "\n\n".join(sections)


# How long to wait for a submitted torrent to appear in qBittorrent before
# treating the add as failed. qBittorrent answers "Ok." as soon as it has taken
# the URL and fetches it afterwards, so this is the earliest moment at which
# "added" can be said truthfully.
ADD_CONFIRM_TIMEOUT = 20.0


async def _add_one(name: str, category: BTCategory, context: TorrentContext, corrected_name: str | None = None) -> str:
    if corrected_name is None:
        top_result = process.extractOne(name, context.search_results.keys())
        corrected_name = top_result[0]
        score = top_result[1]
        if score < 80:
            return (
                f"NOT ADDED: '{name}' is not one of the search results. Add a name "
                f"exactly as search_for_torrent returned it."
            )

    url = context.search_results[corrected_name].fileUrl

    async with QBittorrentClient() as qclient:
        memory_code = await qclient.add_torrent(url, category)
        if memory_code is None:
            return (
                f"NOT ADDED: QBITTORRENT_DRY_RUN is on, so '{corrected_name}' was "
                f"never submitted. Tell the user nothing was downloaded."
            )
        # An add whose URL qBittorrent cannot fetch -- an expired Jackett key, a
        # dead indexer -- is reported exactly like a successful one, so confirm
        # the torrent exists before calling it added.
        try:
            info = await qclient.get_torrent_info(
                memory_code, timeout=ADD_CONFIRM_TIMEOUT
            )
        except TimeoutError as e:
            logging.error(f"Add of '{corrected_name}' was accepted but never appeared: {e}")
            return (
                f"NOT ADDED: qBittorrent accepted '{corrected_name}' but the torrent "
                f"never appeared, so nothing is downloading. Tell the user it failed. "
                f"Details: {e}"
            )
        context.internal_torrents[corrected_name] = memory_code
    return f"Added '{info.name}', confirmed present in qBittorrent."


@tool(args_schema=TorrentAddQuery)
async def add_torrent(
    names: list[str], category: BTCategory, runtime: ToolRuntime[TorrentContext]
) -> str:
    """Add torrents to qBittorrent using names retrieved from a previous search. Fuzzy search.

    Pass every torrent you chose in one call.
    Only tell the user you downloaded something if this tool answers "Added" for it.
    Any line starting with "NOT ADDED" means that item is not downloading and you
    must say that instead.
    """
    context = runtime.context
    context.torrent_types.add(category)
    if not context.search_results:
        return (
            "NOT ADDED: there are no search results to add from. Call "
            "search_for_torrent first, then add one of the names it returned."
        )

    # Resolve up front so two names resolving to the same torrent add it once.
    jobs: list[tuple[int, str]] = []
    seen: set[str] = set()
    lines: dict[int, str] = {}
    for index, name in enumerate(names):
        top_result = process.extractOne(name, context.search_results.keys())
        if top_result is None or top_result[1] < 80:
            lines[index] = (
                f"NOT ADDED: '{name}' is not one of the search results. Add a name "
                f"exactly as search_for_torrent returned it."
            )
            continue
        corrected = top_result[0]
        if corrected in seen:
            lines[index] = f"NOT ADDED: '{name}' resolves to '{corrected}', already being added in this call."
            continue
        seen.add(corrected)
        jobs.append((index, corrected))

    outcomes = await asyncio.gather(
        *[_add_one(corrected, category, context, corrected) for _, corrected in jobs],
        return_exceptions=True,
    )
    for (index, corrected), outcome in zip(jobs, outcomes):
        if isinstance(outcome, BaseException):
            logging.warning(f"Add of '{corrected}' failed: {outcome}")
            outcome = f"NOT ADDED: adding '{corrected}' failed: {outcome}"
        lines[index] = outcome
    return "\n".join(lines[i] for i in range(len(names)))


class PlexAlbumQuery(BaseModel):
    title: str | None
    artist: str


def _plex_metadata(results: dict) -> list[dict]:
    """Pull the Metadata list out of a Plex response, which omits it when empty."""
    container = results.get("MediaContainer")
    if not isinstance(container, dict):
        return []
    metadata = container.get("Metadata")
    return [item for item in metadata if isinstance(item, dict)] if isinstance(metadata, list) else []


async def _canonicalise(artist: str, title: str | None) -> CanonicalRelease:
    """Resolve names through Last.fm, falling back to the names as given.

    Canonicalisation makes the Plex check stricter, so a Last.fm outage or a
    missing API key must not stop the check from running at all.
    """
    try:
        return await resolve_release(artist, title)
    except (LastFMError, aiohttp.ClientError) as e:
        logging.info(f"Last.fm unavailable, checking Plex with names as given: {e}")
        return CanonicalRelease(
            artist=artist,
            artist_candidates=[artist],
            album=title,
            album_candidates=[title] if title else [],
        )


def _album_scores(candidates: list[str], name: str) -> tuple[int, int]:
    """Score a library album title against the titles we are looking for.

    Returns (whole-title similarity, containment). Whole-title similarity ignores
    word order and punctuation so "MM.. FOOD" matches "MM..FOOD", while
    containment catches "Shore (Deluxe Edition)" against "Shore" -- and also
    false friends like "Metallica" against "Metallica Through the Never", which
    is why containment alone is never treated as a collision.
    """
    lowered = name.lower()
    whole = max(fuzz.token_sort_ratio(c.lower(), lowered) for c in candidates)
    contained = max(fuzz.partial_ratio(c.lower(), lowered) for c in candidates)
    return whole, contained


async def _same_release(candidate: str, owned: str) -> float | None:
    """Probability the two releases are the same, or None if no judgement."""
    try:
        decision = await decide(
            SAME_RELEASE_QUESTIONS_NAME,
            {"candidate": candidate, "owned": owned},
            SAME_RELEASE_QUESTIONS,
        )
        if decision is None:
            return None
        return decision.noul("same_release")
    except Exception as e:
        logging.warning(f"same_release decision failed: {e}")
        return None


async def _check_album(artist: str, title: str | None) -> str:
    album_type = PLEX_CONTENT_TYPES["album"]
    release = await _canonicalise(artist, title)
    logging.debug(
        f"Checking Plex for artists {release.artist_candidates} "
        f"albums {release.album_candidates}"
    )

    async with PlexAPIClient() as plex:
        responses = await asyncio.gather(
            *[
                plex.get_all_library_items(
                    {"type": album_type, "artist.title": candidate}
                )
                for candidate in release.artist_candidates
            ],
            return_exceptions=True,
        )

    # Keyed by ratingKey so the same album found under two artist spellings is
    # only reported once.
    owned: dict[str, tuple[str, str]] = {}
    failures: list[BaseException] = []
    for response in responses:
        if isinstance(response, BaseException):
            logging.warning(f"Plex album lookup failed: {response}")
            failures.append(response)
            continue
        for item in _plex_metadata(response):
            album_title = str(item.get("title", ""))
            credited = str(item.get("parentTitle", ""))
            key = str(item.get("ratingKey") or f"{credited}:{album_title}")
            owned[key] = (album_title, credited)

    if failures and len(failures) == len(responses):
        # Plex answers an expired token with a 401, and reporting that as an
        # empty library is how the same album gets downloaded twice.
        return (
            f"COULD NOT CHECK: every Plex lookup for {release.artist} failed "
            f"({failures[0]}). Do not assume the user does or does not own this "
            f"album; say the library check failed and carry on."
        )

    note = ""
    if release.corrected:
        note = f"(Last.fm corrected the artist name to '{release.artist}'.) "

    if not owned:
        return f"{note}The user has no albums by {release.artist}."

    albums = sorted(set(owned.values()))
    if not release.album:
        listing = "\n".join(f"- {name} by {credited}" for name, credited in albums)
        return (
            f"{note}The user already has these albums by {release.artist}:\n{listing}"
        )

    candidates = release.album_candidates or [release.album]
    collisions: list[tuple[int, str, str]] = []
    possibles: list[tuple[int, str, str]] = []
    for name, credited in albums:
        whole, contained = _album_scores(candidates, name)
        if whole >= ALBUM_MATCH_THRESHOLD:
            collisions.append((whole, name, credited))
        elif contained >= ALBUM_CONTAINED_THRESHOLD:
            possibles.append((contained, name, credited))

    suffix: dict[tuple[str, str], str] = {}
    if possibles:
        probs = await asyncio.gather(
            *[
                _same_release(
                    f"{release.artist} - {release.album}", f"{credited} - {name}"
                )
                for _, name, credited in possibles
            ]
        )
        kept: list[tuple[int, str, str]] = []
        for entry, p in zip(possibles, probs):
            key = (entry[1], entry[2])
            if p is None:
                kept.append(entry)
            elif p >= SAME_RELEASE_YES:
                collisions.append(entry)
                suffix[key] = f" (judged the same release, p={p:.2f})"
            elif p <= SAME_RELEASE_NO:
                continue
            else:
                kept.append(entry)
                suffix[key] = f" (same-release probability {p:.2f})"
        possibles = kept

    if collisions:
        collisions.sort(reverse=True)
        listing = "\n".join(
            f"- {name} by {credited}{suffix.get((name, credited), '')}"
            for _, name, credited in collisions
        )
        return (
            f"{note}COLLISION: the user already owns '{release.album}' by "
            f"{release.artist}. Do NOT download it again. Matching albums:\n{listing}"
        )

    if possibles:
        possibles.sort(reverse=True)
        listing = "\n".join(
            f"- {name} by {credited}{suffix.get((name, credited), '')}"
            for _, name, credited in possibles
        )
        return (
            f"{note}The user does not have an exact match for '{release.album}' by "
            f"{release.artist}, but these are close enough that they may be the same "
            f"release under a different edition name. Decide whether downloading "
            f"would duplicate what they have:\n{listing}"
        )

    listing = "\n".join(f"- {name}" for name, _ in albums)
    return (
        f"{note}The user does NOT have '{release.album}' by {release.artist}, so it is "
        f"safe to download. They do already have these other albums by that artist:\n"
        f"{listing}"
    )


@tool(args_schema=PlexAlbumQuery)
async def check_for_album(artist: str, title: str | None = None) -> str:
    """
    Check whether the user already owns an album. Always call this before downloading music.
    Artist and album names are canonicalised through Last.fm and then every album the user
    has by that artist is compared fuzzily, so aliases, punctuation and deluxe editions are
    all caught. You do not need to retry this tool with alternative spellings yourself.
    Omit the title to list everything the user has by that artist.
    """
    return await _check_album(artist, title)


class AlbumRef(BaseModel):
    artist: str
    title: str


class PlexAlbumsQuery(BaseModel):
    albums: list[AlbumRef]


@tool(args_schema=PlexAlbumsQuery)
async def check_albums(albums: list[AlbumRef]) -> str:
    """
    Check whether the user already owns several albums at once. Check every candidate
    album in one call; prefer this over check_for_album when you have more than one album.
    Names are canonicalised through Last.fm and compared fuzzily, like check_for_album.
    """
    refs = [AlbumRef.model_validate(a) if isinstance(a, dict) else a for a in albums]
    results = await asyncio.gather(
        *[_check_album(ref.artist, ref.title) for ref in refs],
        return_exceptions=True,
    )
    sections = []
    for ref, result in zip(refs, results):
        if isinstance(result, BaseException):
            result = f"COULD NOT CHECK: {result}"
        sections.append(f"### {ref.artist} - {ref.title}\n{result}")
    return "\n\n".join(sections)


class PlexSongQuery(BaseModel):
    title: str
    album: str | None
    artist: str | None


@tool(args_schema=PlexSongQuery)
async def get_song_id(
    artist: str, title: str | None = None, album: str | None = None
) -> str:
    """
    This tool is used to get the song id for a given song, providing the artist and title.
    """
    song_type = PLEX_CONTENT_TYPES["song"]
    async with PlexAPIClient() as plex:
        results = await plex.get_all_library_items(
            {
                "type": song_type,
                "artist.title": artist,
                "title": title,
                "album.title": album,
            }
        )
    if "MediaContainer" not in results:
        return "The User does not have any albums that match the query."
    if "Metadata" not in results["MediaContainer"]:
        return "The User does not have any albums that match the query."
    if len(results["MediaContainer"]["Metadata"]) == 0:
        return (
            f"The User does not have any albums that match the query: {artist} {title}"
        )
    song_id = results["MediaContainer"]["Metadata"][0]["key"]
    return song_id


class PlexMovieQuery(BaseModel):
    title: str
    year: int | None


@tool(args_schema=PlexMovieQuery)
async def check_for_movie(title: str, year: int | None = None) -> str:
    """
    This tool is used to check if the user already has a movie by a given title. You can also not specify a year and it will return all movies by the title.
    """
    movie_type = PLEX_CONTENT_TYPES["movie"]
    logging.debug(f"Checking for movie: {title} {year}")
    async with PlexAPIClient() as plex:
        if year:
            results = await plex.get_all_library_items(
                {"type": movie_type, "title": title, "year": year}
            )
        else:
            results = await plex.get_all_library_items(
                {"type": movie_type, "title": title}
            )
    if "MediaContainer" not in results:
        return "The User does not have any movies that match the query."
    if "Metadata" not in results["MediaContainer"]:
        return "The User does not have any movies that match the query."
    if len(results["MediaContainer"]["Metadata"]) == 0:
        return f"The User does not have any movies that match the query: {title} {year}"
    logging.debug(f"Got results: {results}")
    response_text = "The User already has the following movies:\n" + "\n".join(
        [
            f"{result['title']} ({result['year']})"
            for result in results["MediaContainer"]["Metadata"]
        ]
    )
    return response_text


class LastFMArtistQuery(BaseModel):
    artist: str


class LastFMSimilarQuery(BaseModel):
    artist: str
    limit: int = 15


class LastFMAlbumsQuery(BaseModel):
    artists: list[str]
    limit: int = 15

    @field_validator("artists", mode="before")
    @classmethod
    def _wrap_artists(cls, value):
        return _str_to_list(value)


class LastFMTagQuery(BaseModel):
    tag: str
    limit: int = 30


class LastFMResolveQuery(BaseModel):
    artist: str
    album: str | None = None


async def _library_albums(plex, artist: str) -> list[str] | None:
    """Album titles the user has by exactly this artist name, or None if Plex failed."""
    try:
        results = await plex.get_all_library_items(
            {"type": PLEX_CONTENT_TYPES["album"], "artist.title": artist}
        )
        return [str(item.get("title", "")) for item in _plex_metadata(results)]
    except Exception as e:
        logging.warning(f"Plex library lookup for {artist} failed: {e}")
        return None


def _maybe_title(owned: list[str] | None, album: str) -> str | None:
    """The owned title that is a maybe for album (None if exact, none or failed)."""
    if owned is None:
        return None
    maybe = None
    for title in owned:
        whole, contained = _album_scores([album], title)
        if whole >= ALBUM_MATCH_THRESHOLD:
            return None
        if maybe is None and contained >= ALBUM_CONTAINED_THRESHOLD:
            maybe = title
    return maybe


def _ownership(owned: list[str] | None, album: str) -> str:
    if owned is None:
        return " [library check failed]"
    maybe = None
    for title in owned:
        whole, contained = _album_scores([album], title)
        if whole >= ALBUM_MATCH_THRESHOLD:
            return " [OWNED]"
        if maybe is None and contained >= ALBUM_CONTAINED_THRESHOLD:
            maybe = title
    return f" [maybe owned as '{maybe}']" if maybe is not None else ""


async def _resolve_maybes(
    artist: str, albums: list[str], owned: list[str] | None
) -> dict[str, str]:
    """Final ownership mark per album, asking the model only for maybes."""
    marks = {album: _ownership(owned, album) for album in albums}
    pending = [a for a, m in marks.items() if m.startswith(" [maybe owned as '")]
    titles = {a: _maybe_title(owned, a) for a in pending}
    probs = await asyncio.gather(
        *[
            _same_release(f"{artist} - {a}", f"{artist} - {titles[a]}")
            for a in pending
        ]
    )
    for album, p in zip(pending, probs):
        if p is None:
            continue
        if p >= SAME_RELEASE_YES:
            marks[album] = " [OWNED]"
        elif p <= SAME_RELEASE_NO:
            marks[album] = ""
    return marks


def _artist_ownership(owned: list[str] | None) -> str:
    if owned is None:
        return " [library check failed]"
    return f" [in library: {len(owned)} albums]" if owned else " [new to you]"


async def _library_lookup(artists: list[str]) -> dict[str, list[str] | None]:
    """Owned album titles per distinct artist; all None if Plex is unreachable."""
    distinct = list(dict.fromkeys(artists))
    if not distinct:
        return {}
    try:
        async with PlexAPIClient() as plex:
            owned = await asyncio.gather(
                *[_library_albums(plex, name) for name in distinct]
            )
        return dict(zip(distinct, owned))
    except Exception as e:
        logging.warning(f"Plex unavailable for library annotation: {e}")
        return {name: None for name in distinct}


@tool(args_schema=LastFMArtistQuery)
async def lastfm_artist_info(artist: str) -> str:
    """
    Look up an artist on Last.fm: their canonical name, genre tags, listener count and a
    short biography. Use this to find out what an artist actually sounds like and which
    tags to browse next, instead of relying on what you remember about them.
    """
    async with LastFMClient() as lastfm:
        info = await lastfm.get_artist_info(artist)
    lines = [f"Artist: {info.name}"]
    if info.listeners:
        lines.append(f"Listeners: {info.listeners:,}")
    if info.tags:
        lines.append(f"Tags: {', '.join(info.tags)}")
    if info.similar:
        lines.append(f"Similar artists: {', '.join(info.similar)}")
    if info.summary:
        lines.append(f"Biography: {info.summary}")
    return "\n".join(lines)


@tool(args_schema=LastFMSimilarQuery)
async def lastfm_similar_artists(artist: str, limit: int = 15) -> str:
    """
    Find artists Last.fm considers similar to a given artist, most similar first, with a
    0-1 similarity score. Use this for "more like this" and "introduce me to new music"
    requests rather than guessing at similar artists yourself.
    Artists are marked with library ownership ([in library: N albums] / [new to you]), so
    you can skip ones the user already has without calling check_albums separately.
    """
    async with LastFMClient() as lastfm:
        similar = await lastfm.get_similar_artists(artist, limit=limit)
    if not similar:
        return f"Last.fm has no similar artists for {artist}."
    owned = await _library_lookup([item.name for item in similar])
    listing = "\n".join(
        f"- {item.name}"
        + (f" (similarity {item.match:.2f})" if item.match else "")
        + _artist_ownership(owned.get(item.name))
        for item in similar
    )
    return f"Artists similar to {artist}:\n{listing}"


async def _artist_albums_section(artist: str, limit: int) -> str:
    async with LastFMClient() as lastfm:
        albums = await lastfm.get_artist_albums(artist, limit=limit)
    if not albums:
        return f"Last.fm has no albums listed for {artist}."
    resolved = albums[0].artist or artist
    owned = (await _library_lookup([resolved]))[resolved]
    marks = await _resolve_maybes(resolved, [a.name for a in albums], owned)
    listing = "\n".join(
        f"- {album.name}"
        + (f" ({album.playcount:,} plays)" if album.playcount else "")
        + marks[album.name]
        for album in albums
    )
    new = sum(1 for album in albums if marks[album.name] == "")
    return (
        f"Albums by {resolved} on Last.fm, most played first:\n{listing}\n"
        f"{new} of these are not in the library."
    )


@tool(args_schema=LastFMAlbumsQuery)
async def lastfm_artist_albums(artists: list[str], limit: int = 15) -> str:
    """
    List albums from Last.fm for one or more artists, most listened first. Pass every
    artist you are considering in one call. These are real releases with real spellings,
    so use this to build a discography before searching for torrents instead of recalling
    album names from memory. Raise the limit for a fuller discography, but be aware the
    tail of the list drifts into singles, live records and compilations.
    Albums are marked [OWNED] / [maybe owned as ...] when the user's library has them, so
    pick unmarked ones directly without calling check_albums separately.
    """
    results = await asyncio.gather(
        *[_artist_albums_section(artist, limit) for artist in artists],
        return_exceptions=True,
    )
    return "\n\n".join(
        f"Last.fm lookup for {artist} failed: {result}"
        if isinstance(result, BaseException)
        else result
        for artist, result in zip(artists, results)
    )


@tool(args_schema=LastFMTagQuery)
async def lastfm_browse_tag(tag: str, limit: int = 30) -> str:
    """
    Browse a Last.fm genre or style tag (for example "future jazz", "shoegaze",
    "hauntology") and get back the top artists and top albums carrying that tag. Use this
    when the user asks for a style rather than a specific artist.
    Results are marked with library ownership (albums [OWNED], artists [in library: N
    albums] / [new to you]) so you can skip owned ones without calling check_albums separately.
    """
    async with LastFMClient() as lastfm:
        artists, albums = await asyncio.gather(
            lastfm.get_tag_top_artists(tag, limit=limit),
            lastfm.get_tag_top_albums(tag, limit=limit),
        )
    if not artists and not albums:
        return f"Last.fm has nothing tagged '{tag}'. Try a broader or differently spelled tag."
    owned = await _library_lookup(
        [item.name for item in artists]
        + [album.artist for album in albums if album.artist]
    )
    sections = []
    if artists:
        sections.append(
            f"Top artists tagged '{tag}':\n"
            + "\n".join(
                f"- {item.name}" + _artist_ownership(owned.get(item.name))
                for item in artists
            )
        )
    if albums:
        by_artist: dict[str, list[str]] = {}
        for album in albums:
            if album.artist:
                by_artist.setdefault(album.artist, []).append(album.name)
        resolved = await asyncio.gather(
            *[
                _resolve_maybes(a, names, owned.get(a))
                for a, names in by_artist.items()
            ]
        )
        marks = dict(zip(by_artist, resolved))
        sections.append(
            f"Top albums tagged '{tag}':\n"
            + "\n".join(
                f"- {album.name}"
                + (f" by {album.artist}" if album.artist else "")
                + (
                    marks[album.artist][album.name]
                    if album.artist
                    else ""
                )
                for album in albums
            )
        )
    return "\n\n".join(sections)


@tool(args_schema=LastFMResolveQuery)
async def lastfm_resolve(artist: str, album: str | None = None) -> str:
    """
    Resolve a possibly misspelled or aliased artist/album name to the spelling Last.fm uses,
    along with its MusicBrainz id. Call this before searching for torrents so the search
    query uses a real release name. check_for_album already does this internally, so you do
    not need to call this first just to check the library.
    """
    release = await resolve_release(artist, album)
    lines = [f"Canonical artist: {release.artist}"]
    if release.corrected:
        lines.append(f"(Corrected from '{artist}'.)")
    if release.artist_mbid:
        lines.append(f"Artist MusicBrainz id: {release.artist_mbid}")
    if release.album:
        lines.append(f"Canonical album: {release.album}")
    if release.album_mbid:
        lines.append(f"Album MusicBrainz id: {release.album_mbid}")
    lines.append(f"Use this for torrent searches: {release.artist} {release.album or ''}".strip())
    return "\n".join(lines)


class DownloadAlbumsQuery(BaseModel):
    albums: list[AlbumRef]
    media: Literal["CD", "SACD", "WEB", "Vinyl"] | None = None


_EDITION_WORDS = {
    "deluxe", "edition", "expanded", "remaster", "remastered", "anniversary",
    "bonus", "version", "special", "complete", "reissue", "collectors",
    "collector's", "super", "anniv", "th", "the",
}
_WORD = re.compile(r"[a-z0-9']+")


def _title_matches(wanted: str, found: str) -> bool:
    """Whether a torrent's title is the album asked for, or an edition of it.

    "Mordechai" contains in "Mordechai Remixes" too, and a remix album is not
    what was asked for, so extra words only pass when they are edition words
    ("Deluxe Edition", "2017 Remaster").
    """
    a, b = wanted.casefold(), found.casefold()
    if re.sub(r"[\W_]+", "", a) == re.sub(r"[\W_]+", "", b):
        return True
    want = set(_WORD.findall(a))
    extra = {w for w in set(_WORD.findall(b)) - want if not w.isdigit()}
    # "The Universe Smiles Upon You ii" is a different album however close.
    if not want or not extra <= _EDITION_WORDS:
        return False
    return fuzz.token_set_ratio(a, b) >= 90


@tool(args_schema=DownloadAlbumsQuery)
async def download_albums(
    albums: list[AlbumRef],
    runtime: ToolRuntime[TorrentContext],
    media: str | None = None,
) -> str:
    """Get albums in one step: checks the library, searches, picks the best version (SACD and perfect CD rips first, never vinyl unless asked; set media only when the user asks for CD, SACD, WEB or Vinyl) and adds it. Pass every album you chose in one call, using Last.fm spellings. Only items reported as 'Added' are downloading."""
    context = runtime.context
    context.torrent_types.add(BTCategory.Music)
    refs = [AlbumRef.model_validate(a) if isinstance(a, dict) else a for a in albums]
    semaphore = asyncio.Semaphore(3)
    claimed: set[str] = set()

    async def _one(ref: AlbumRef) -> str:
        artist, title = ref.artist, ref.title
        label = f"'{artist} - {title}'"
        owned = await _check_album(artist, title)
        if "COLLISION:" in owned:
            return f"OWNED: {label} is already in the library; skipped."

        async with semaphore:
            async with QBittorrentClient() as qclient:
                response = await qclient.search(f"{artist} {title}", BTCategory.Music)
        found = [r for r in (response.results or []) if "FLAC" in r.fileName]
        for result in found:
            context.search_results[result.fileName] = result
        names = [r.fileName for r in found]

        ranked = best_versions(names, media)
        matches = [
            r
            for r in ranked
            if _title_matches(title, r.title)
            and fuzz.partial_ratio(artist.casefold(), r.artist.casefold()) >= 90
        ]
        if not matches:
            line = f"NOT FOUND: no FLAC torrent for {label}."
            if ranked:
                line += " Closest results: " + "; ".join(r.name for r in ranked[:3])
            return line
        best = max(
            matches,
            key=lambda r: (
                r.kind == "Album",
                fuzz.ratio(title.casefold(), r.title.casefold()),
                quality_key(r, media),
            ),
        )
        if media and best.media.casefold() != media.casefold():
            return (
                f"NOT ADDED: there is no {media} version of {label}. Best available: "
                f"{best.name}. Ask again without a media preference to get it."
            )
        if best.vinyl and not media:
            return (
                f"NOT ADDED: only a vinyl rip of {label} exists ({best.name}). "
                f"Ask for vinyl to get it."
            )
        if best.name in claimed:
            return "NOT ADDED: duplicate of an earlier album in this call."
        claimed.add(best.name)
        return await _add_one(best.name, BTCategory.Music, context, corrected_name=best.name)

    outcomes = await asyncio.gather(*[_one(r) for r in refs], return_exceptions=True)
    lines = []
    for ref, outcome in zip(refs, outcomes):
        if isinstance(outcome, BaseException):
            logging.warning(f"download_albums for {ref} failed: {outcome}")
            outcome = f"NOT ADDED: '{ref.artist} - {ref.title}' failed: {outcome}"
        lines.append(outcome)
    return "\n".join(lines)
