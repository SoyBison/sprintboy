import asyncio
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
from dataclasses import dataclass
from pydantic import BaseModel
from thefuzz import fuzz, process

# Whole-title similarity at or above this means the user already owns the album.
ALBUM_MATCH_THRESHOLD = 90
# One title containing the other at or above this is a maybe: a deluxe edition,
# a reissue, or a different album that happens to share a prefix. The agent is
# given these to judge rather than being told they are a collision.
ALBUM_CONTAINED_THRESHOLD = 90


@dataclass
class TorrentContext:
    """A simple class to keep track of known torrents on trackers and qBittorrent."""

    search_results: dict[str, SearchResult]
    internal_torrents: dict[str, Union[str, None]]
    torrent_types: set[BTCategory]


class TorrentAddQuery(BaseModel):
    name: str
    category: BTCategory


class TorrentSearchQuery(BaseModel):
    query: str
    category: BTCategory


@tool(args_schema=TorrentSearchQuery)
async def search_for_torrent(
    query: str, category: BTCategory, runtime: ToolRuntime[TorrentContext]
) -> str:
    """
    Perform a search query on qBittorrent and return the results.
    This is not like google.
    It only returns results that match the query in a fuzzy REGEX.
    Do not include words like "discography" or "album" in your search, this tool only returns single album torrents, or single movies, or single episodes.
    It is best to only include album titles and artist names in your query.
    This tool can find Movies, Music, and TV shows.
    Do not include words like "BluRay" or "720p" or "1080p" in your query.
    """
    runtime.context.torrent_types.add(category)
    async with QBittorrentClient() as qclient:
        results = await qclient.search(query, category)
        if not results.results:
            return "No results found."
        # filter out results that are not flacs
        if category == BTCategory.Music:
            results = [
                result for result in results.results if ("FLAC" in result.fileName)
            ]
        else:
            results = results.results

        # Store results in runtime for later use
        for result in results:
            runtime.context.search_results[result.fileName] = result
        summary = "\n".join(result.fileName for result in results)
    return f"Search results:\n{summary}"


@tool(args_schema=TorrentAddQuery)
async def add_torrent(
    name: str, category: BTCategory, runtime: ToolRuntime[TorrentContext]
) -> str:
    """Add a torrent to qBittorrent using a name retrieved from a previous search. Fuzzy search."""
    runtime.context.torrent_types.add(category)
    top_result = process.extractOne(name, runtime.context.search_results.keys())
    corrected_name = top_result[0]
    score = top_result[1]
    if score < 80:
        raise FileNotFoundError(
            f"Torrent with name '{name}' not found in known torrents."
        )

    url = runtime.context.search_results[corrected_name].fileUrl

    async with QBittorrentClient() as qclient:
        memory_code = await qclient.add_torrent(url, category)
        runtime.context.internal_torrents[corrected_name] = memory_code
        return "Torrent added successfully."


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


@tool(args_schema=PlexAlbumQuery)
async def check_for_album(artist: str, title: str | None = None) -> str:
    """
    Check whether the user already owns an album. Always call this before downloading music.
    Artist and album names are canonicalised through Last.fm and then every album the user
    has by that artist is compared fuzzily, so aliases, punctuation and deluxe editions are
    all caught. You do not need to retry this tool with alternative spellings yourself.
    Omit the title to list everything the user has by that artist.
    """
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
    for response in responses:
        if isinstance(response, BaseException):
            logging.warning(f"Plex album lookup failed: {response}")
            continue
        for item in _plex_metadata(response):
            album_title = str(item.get("title", ""))
            credited = str(item.get("parentTitle", ""))
            key = str(item.get("ratingKey") or f"{credited}:{album_title}")
            owned[key] = (album_title, credited)

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

    if collisions:
        collisions.sort(reverse=True)
        listing = "\n".join(
            f"- {name} by {credited}" for _, name, credited in collisions
        )
        return (
            f"{note}COLLISION: the user already owns '{release.album}' by "
            f"{release.artist}. Do NOT download it again. Matching albums:\n{listing}"
        )

    if possibles:
        possibles.sort(reverse=True)
        listing = "\n".join(
            f"- {name} by {credited}" for _, name, credited in possibles
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
    artist: str
    limit: int = 15


class LastFMTagQuery(BaseModel):
    tag: str
    limit: int = 15


class LastFMResolveQuery(BaseModel):
    artist: str
    album: str | None = None


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
    """
    async with LastFMClient() as lastfm:
        similar = await lastfm.get_similar_artists(artist, limit=limit)
    if not similar:
        return f"Last.fm has no similar artists for {artist}."
    listing = "\n".join(
        f"- {item.name}" + (f" (similarity {item.match:.2f})" if item.match else "")
        for item in similar
    )
    return f"Artists similar to {artist}:\n{listing}"


@tool(args_schema=LastFMAlbumsQuery)
async def lastfm_artist_albums(artist: str, limit: int = 15) -> str:
    """
    List an artist's albums from Last.fm, most listened first. These are real releases with
    real spellings, so use this to build a discography before searching for torrents instead
    of recalling album names from memory. Raise the limit for a fuller discography, but be
    aware the tail of the list drifts into singles, live records and compilations.
    """
    async with LastFMClient() as lastfm:
        albums = await lastfm.get_artist_albums(artist, limit=limit)
    if not albums:
        return f"Last.fm has no albums listed for {artist}."
    listing = "\n".join(
        f"- {album.name}"
        + (f" ({album.playcount:,} plays)" if album.playcount else "")
        for album in albums
    )
    return f"Albums by {albums[0].artist or artist} on Last.fm, most played first:\n{listing}"


@tool(args_schema=LastFMTagQuery)
async def lastfm_browse_tag(tag: str, limit: int = 15) -> str:
    """
    Browse a Last.fm genre or style tag (for example "future jazz", "shoegaze",
    "hauntology") and get back the top artists and top albums carrying that tag. Use this
    when the user asks for a style rather than a specific artist.
    """
    async with LastFMClient() as lastfm:
        artists, albums = await asyncio.gather(
            lastfm.get_tag_top_artists(tag, limit=limit),
            lastfm.get_tag_top_albums(tag, limit=limit),
        )
    if not artists and not albums:
        return f"Last.fm has nothing tagged '{tag}'. Try a broader or differently spelled tag."
    sections = []
    if artists:
        sections.append(
            f"Top artists tagged '{tag}':\n"
            + "\n".join(f"- {item.name}" for item in artists)
        )
    if albums:
        sections.append(
            f"Top albums tagged '{tag}':\n"
            + "\n".join(
                f"- {album.name}" + (f" by {album.artist}" if album.artist else "")
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
