from langchain.tools import BaseTool, ToolRuntime, tool
from netcode import QBittorrentClient, SearchResult
from dataclasses import dataclass


@dataclass
class KnownTorrents:
    """A simple class to keep track of known torrents."""

    torrents: dict[str, SearchResult]


@tool
async def search_for_torrents(query: str, runtime: ToolRuntime[KnownTorrents]) -> str:
    """Perform a search query on qBittorrent and return the results."""
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
