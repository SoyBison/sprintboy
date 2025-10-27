import logging
import os
from typing import Callable, TypeVar, List

import aiohttp
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel, RootModel

from yarl import URL

T = TypeVar("T", bound=BaseModel)

PLEX_CONTENT_TYPES: dict[str, int] = {
    "movie": 1,
    "show": 2,
    "season": 3,
    "episode": 4,
    "trailer": 5,
    "person": 7,
    "artist": 8,
    "album": 9,
    "track": 10,
    "clip": 12,
    "photo": 13,
    "photoalbum": 14,
    "playlist": 15,
    "playlistfolder": 16,
}


# Pydantic models for API responses
class SearchStartResponse(BaseModel):
    id: int


class SearchStatusItem(BaseModel):
    status: str


class SearchResult(BaseModel):
    fileName: str
    fileUrl: str
    fileSize: int
    nbSeeders: int
    nbLeechers: int
    siteUrl: str
    descrLink: str


class SearchResultsResponse(BaseModel):
    results: List[SearchResult]


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    expected_model: type[T],
    method: str = "GET",
    data: dict | None = None,
    params: dict | None = None,
    headers: dict | None = None,
) -> T:
    """
    Fetch a URL and parse the JSON response into the expected Pydantic model.

    Args:
        session: The aiohttp client session
        url: The URL to fetch
        expected_model: The Pydantic model class to parse the response into
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        data: Form data for POST/PUT requests
        params: Query parameters for GET requests
        headers: Additional headers to include in the request

    Returns:
        An instance of the expected_model with the parsed response data
    """
    kwargs = {}
    if data is not None:
        kwargs["data"] = data
    if params is not None:
        kwargs["params"] = params
    if headers is not None:
        kwargs["headers"] = headers

    async with session.request(method, url, **kwargs) as response:
        response.raise_for_status()
        json_data = await response.json()
        return expected_model.model_validate(json_data)


def synchronize(func: Callable) -> Callable:
    """Decorator to run async functions in a synchronous context, that won't break existing event loops."""

    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If the loop is already running, create a new one
                new_loop = asyncio.new_event_loop()
                return new_loop.run_until_complete(func(*args, **kwargs))
            else:
                return loop.run_until_complete(func(*args, **kwargs))
        except RuntimeError:
            # No event loop in the current context
            new_loop = asyncio.new_event_loop()
            return new_loop.run_until_complete(func(*args, **kwargs))

    return wrapper


class QBittorrentClient:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("QBITTORRENT_API_URL", "http://localhost:8080/api/v2")
        self.username = os.getenv("QBITTORRENT_USERNAME", "admin")
        self.password = os.getenv("QBITTORRENT_PASSWORD")
        self.session: aiohttp.ClientSession | None = None
        self.torrent_path = os.getenv("QBITTORRENT_DOWNLOAD_PATH", "/data/Music")
        self.dry_run = os.getenv("QBITTORRENT_DRY_RUN", "false").lower() == "true"
        self.cookie: str | None = None

    async def __aenter__(self):
        # Use unsafe cookies because we're in the local network
        jar = aiohttp.CookieJar(unsafe=True)
        self.session = aiohttp.ClientSession(cookie_jar=jar)
        await self.login()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def login(self):
        """Login to qBittorrent and store session cookies."""
        # Check Env
        assert self.session is not None, "Session not initialized"
        login_url = f"{self.base_url}/auth/login"
        data = {"username": self.username, "password": self.password}
        logging.debug(
            f"Logging in to qBittorrent at {login_url} with user {self.username}"
        )
        async with self.session.post(login_url, data=data) as response:
            if response.status == 200:
                logging.debug("Logged in to qBittorrent successfully.")
                logging.debug(f"Response cookies: {response.cookies}")
            else:
                logging.error(f"Failed to log in to qBittorrent: {response.status}")
                raise Exception("Login failed")
        logging.debug(
            f"Session cookies after login: {self.session.cookie_jar.filter_cookies(URL(self.base_url))}"
        )

    async def search(self, query: str) -> SearchResultsResponse:
        """Perform a search query on qBittorrent and return the results."""
        assert self.session is not None, "Session not initialized"
        # We have to start a search,  and poll the status until done, then fetch the results
        logging.debug(f"Starting search for query: {query}")
        logging.debug(
            f"Using session cookies: {self.session.cookie_jar.filter_cookies(URL(self.base_url))}"
        )
        start_search_url = f"{self.base_url}/search/start"
        data = {
            "pattern": query,
            "plugins": "enabled",
            "category": "all",
        }  # TODO: Make category configurable
        search_response = await fetch_url(
            self.session,
            start_search_url,
            SearchStartResponse,
            method="POST",
            data=data,
        )
        search_id = search_response.id
        logging.debug(f"Search started with ID: {search_id}")
        # Polling for search status
        search_status_url = f"{self.base_url}/search/status"
        while True:
            async with self.session.post(
                search_status_url, data={"id": search_id}
            ) as response:
                status_data = await response.json()
                if all(item["status"] == "Stopped" for item in status_data):
                    logging.debug("Search completed.")
                    break
                logging.debug(
                    f"Search still in progress, waiting... Status: {status_data}"
                )
                # This is legal because it's in my local network
                await asyncio.sleep(0.1)

        # Fetching search results
        search_results_url = f"{self.base_url}/search/results"
        results_response = await fetch_url(
            self.session,
            search_results_url,
            SearchResultsResponse,
            method="POST",
            data={"id": search_id},
        )
        logging.debug(
            f"Search results fetched, total results: {len(results_response.results)}"
        )
        return results_response

    async def add_torrent(self, torrent_url: str) -> None:
        """Download a torrent from a given URL."""
        assert self.session is not None, "Session not initialized"
        download_url = f"{self.base_url}/torrents/add"
        logging.debug(f"Downloading torrent from URL: {torrent_url}")
        payload = {
            "urls": torrent_url,
            "savepath": self.torrent_path,
            "category": "Music",
        }
        if self.dry_run:
            logging.info(f"Dry run enabled, not downloading torrent: {torrent_url}")
            return
        async with self.session.post(download_url, data=payload) as response:
            if response.status == 200:
                if (await response.text()) == "Fails.":
                    logging.error(f"Failed to add torrent: {await response.text()}")
                    raise Exception("Download failed")
                logging.debug(
                    f"Torrent download initiated successfully for {torrent_url}"
                )
            else:
                logging.error(f"Failed to initiate torrent download: {response.status}")
                raise Exception("Download failed")


class PlexAPIClient:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("PMS_URL")
        self.token = os.getenv("PLEX_TOKEN")
        self.client_id = os.getenv("PLEX_CLIENT_ID")
        self.client_name = os.getenv("PLEX_CLIENT_NAME")
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self.session is not None
        await self.session.close()

    async def get_library_matches(
        self,
        title: str,
        metadata_type: str,
        year: int | None = None,
        parentTitle: str | None = None,
    ) -> dict:
        assert self.session is not None
        url = f"{self.base_url}/library/matches"
        assert self.token, "Token not found"
        metadata_type_id = PLEX_CONTENT_TYPES[metadata_type]
        headers = {
            "Accept": "application/json",
            "X-Plex-Product": self.client_name,
            "X-Plex-Client-Identifier": self.client_id,
            "X-Plex-Token": self.token,
        }
        # TODO: This return model is a bear we'll do this properly later.
        payload = {
            "title": title,
            "type": metadata_type_id,
            "includeFullMetadata": 1,
        }
        # This crashes if you send Nones
        if year:
            payload["year"] = year
        if parentTitle:
            payload["parentTitle"] = parentTitle
        logging.debug(f"Fetching library matches for: {payload}")
        async with self.session.get(url, headers=headers, params=payload) as response:
            if response.status == 200:
                logging.debug(f"Got library matches: {await response.text()}")
                return await response.json()
            else:
                raise Exception(
                    f"Failed to get library matches: {response.status}, {await response.text()}"
                )

    async def get_all_library_items(self, query_params: dict | None = None) -> dict:
        assert self.session is not None
        url = f"{self.base_url}/library/all"
        assert self.token, "Token not found"
        headers = {
            "Accept": "application/json",
            "X-Plex-Product": self.client_name,
            "X-Plex-Client-Identifier": self.client_id,
            "X-Plex-Token": self.token,
        }
        async with self.session.get(
            url, headers=headers, params=query_params
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(
                    f"Failed to get library items: {response.status}, {await response.text()}"
                )
