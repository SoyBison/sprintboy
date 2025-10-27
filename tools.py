import logging
from langchain.tools import BaseTool, ToolRuntime, tool
from netcode import QBittorrentClient, SearchResult, PlexAPIClient, PLEX_CONTENT_TYPES
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class KnownTorrents:
    """A simple class to keep track of known torrents."""

    torrents: dict[str, SearchResult]


@tool
async def search_for_torrents(query: str, runtime: ToolRuntime[KnownTorrents]) -> str:
    """Perform a search query on qBittorrent and return the results. This is not like google. It only returns results that match the query in a fuzzy REGEX. Do not include words like "discography" in your search, this only returns single album torrents, or single movies, or single episodes."""
    async with QBittorrentClient() as qclient:
        results = await qclient.search(query)
        if not results.results:
            return "No results found."
        # Store results in runtime for later use
        for result in results.results:
            runtime.context.torrents[result.fileName] = result
        summary = "\n".join(result.fileName for result in results.results)
    return f"Search results:\n{summary}"


@tool
async def add_torrent(name: str, runtime: ToolRuntime[KnownTorrents]) -> str:
    """Add a torrent to qBittorrent using a name retrieved from a previous search."""
    if name not in runtime.context.torrents:
        raise FileNotFoundError(
            f"Torrent with name '{name}' not found in known torrents."
        )
    url = runtime.context.torrents[name].fileUrl

    async with QBittorrentClient() as qclient:
        await qclient.add_torrent(url)
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
