import logging
from typing import Union
from langchain.tools import BaseTool, ToolRuntime, tool
from bot.netcode import (
    BTCategory,
    QBittorrentClient,
    SearchResult,
    PlexAPIClient,
    PLEX_CONTENT_TYPES,
)
from dataclasses import dataclass
from pydantic import BaseModel
from thefuzz import process


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


@tool(args_schema=PlexAlbumQuery)
async def check_for_album(artist: str, title: str | None = None) -> str:
    """
    This tool is used to check if the user already has an album by a given artist. You can also not specify a title and it will return all albums by the artist. Remember that this is an exact match based service, and so you may want to also search aliases and alternative spellings to be sure, if that makes sense.
    """
    album_type = PLEX_CONTENT_TYPES["album"]
    logging.debug(f"Checking for album: {artist} {title}")
    async with PlexAPIClient() as plex:
        if title:
            results = await plex.get_all_library_items(
                {"type": album_type, "artist.title": artist, "title": title}
            )
        else:
            results = await plex.get_all_library_items(
                {"type": album_type, "artist.title": artist}
            )
    if "MediaContainer" not in results:
        return "The User does not have any albums that match the query."
    if "Metadata" not in results["MediaContainer"]:
        return "The User does not have any albums that match the query."
    if len(results["MediaContainer"]["Metadata"]) == 0:
        return (
            f"The User does not have any albums that match the query: {artist} {title}"
        )
    logging.debug(f"Got results: {results}")
    response_text = "The User already has the following albums:\n" + "\n".join(
        [
            f"{result['title']} by {result['parentTitle']}"
            for result in results["MediaContainer"]["Metadata"]
        ]
    )

    return response_text


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
