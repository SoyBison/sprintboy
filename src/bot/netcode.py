from enum import StrEnum
import logging
import os
from pathlib import Path
from typing import Callable, Sequence, TypeVar, List

import aiohttp
from dotenv import load_dotenv
import asyncio
from pydantic import BaseModel, RootModel
from cyksuid.v2 import KsuidMs

from thefuzz import process

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


class TorrentInfoResponse(BaseModel):
    added_on: int
    amount_left: int
    auto_tmm: bool
    availability: float
    category: str
    completed: int
    completion_on: int
    content_path: str
    dl_limit: int
    dlspeed: int
    downloaded: int
    downloaded_session: int
    eta: int
    f_l_piece_prio: bool
    force_start: bool
    hash: str
    isPrivate: bool | None = None
    last_activity: int
    magnet_uri: str
    max_ratio: float
    max_seeding_time: int
    name: str
    num_complete: int
    num_incomplete: int
    num_leechs: int
    num_seeds: int
    priority: int
    progress: float
    ratio: float
    ratio_limit: float
    save_path: str
    seeding_time: int
    seeding_time_limit: int
    seen_complete: int
    seq_dl: bool
    size: int
    state: str  # TODO: This should be an enum
    super_seeding: bool
    total_size: int
    up_limit: int
    uploaded: int
    uploaded_session: int
    url: str | None = None
    tags: str  # comma separated
    time_active: int
    tracker: str
    upspeed: int


class TorrentInfoResponses(RootModel):
    root: List[TorrentInfoResponse]


BTCategory = StrEnum("BTCategory", ("Music", "TV", "Movies"))


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


class AsyncAPIClient:
    """Shared aiohttp session lifecycle for the API clients in this module.

    Subclasses read their own configuration in __init__ and are only usable
    inside `async with`, which is what `_live_session` enforces.
    """

    def __init__(self):
        load_dotenv()
        self.session: aiohttp.ClientSession | None = None

    def _new_session(self) -> aiohttp.ClientSession:
        """Override to customise the session, e.g. with a cookie jar."""
        return aiohttp.ClientSession()

    async def _on_connect(self) -> None:
        """Override to authenticate or validate config before the first request."""

    async def __aenter__(self):
        self.session = self._new_session()
        try:
            await self._on_connect()
        except BaseException:
            # Don't leak the session when logging in or validating fails.
            await self.session.close()
            self.session = None
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None

    @property
    def _live_session(self) -> aiohttp.ClientSession:
        assert self.session is not None, "Session not initialized"
        return self.session


class QBittorrentClient(AsyncAPIClient):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("QBITTORRENT_API_URL", "http://localhost:8080/api/v2")
        self.username = os.getenv("QBITTORRENT_USERNAME", "admin")
        self.password = os.getenv("QBITTORRENT_PASSWORD")
        self.torrent_path = os.getenv("QBITTORRENT_DOWNLOAD_PATH", "/data/")
        self.dry_run = os.getenv("QBITTORRENT_DRY_RUN", "false").lower() == "true"
        self.cookie: str | None = None

    def _new_session(self) -> aiohttp.ClientSession:
        # Use unsafe cookies because we're in the local network
        return aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))

    async def _on_connect(self) -> None:
        await self.login()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def login(self):
        """Login to qBittorrent and store session cookies."""
        session = self._live_session
        login_url = self._url("/auth/login")
        data = {"username": self.username, "password": self.password}
        logging.debug(
            f"Logging in to qBittorrent at {login_url} with user {self.username}"
        )
        async with session.post(login_url, data=data) as response:
            if response.status == 200:
                logging.debug("Logged in to qBittorrent successfully.")
                logging.debug(f"Response cookies: {response.cookies}")
            else:
                logging.error(f"Failed to log in to qBittorrent: {response.status}")
                raise Exception("Login failed")
        logging.debug(
            f"Session cookies after login: {session.cookie_jar.filter_cookies(URL(self.base_url))}"
        )

    async def search(
        self, query: str, category: BTCategory | None = None
    ) -> SearchResultsResponse:
        """Perform a search query on qBittorrent and return the results."""
        session = self._live_session

        # We have to start a search,  and poll the status until done, then fetch the results
        logging.debug(f"Starting search for query: {query}")
        logging.debug(
            f"Using session cookies: {session.cookie_jar.filter_cookies(URL(self.base_url))}"
        )
        start_search_url = self._url("/search/start")
        data = {
            "pattern": query,
            "plugins": "enabled",
            "category": category if category else "all",
        }
        search_response = await fetch_url(
            session,
            start_search_url,
            SearchStartResponse,
            method="POST",
            data=data,
        )
        search_id = search_response.id
        logging.debug(f"Search started with ID: {search_id}")
        # Polling for search status
        search_status_url = self._url("/search/status")
        while True:
            async with session.post(
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
        search_results_url = self._url("/search/results")
        results_response = await fetch_url(
            session,
            search_results_url,
            SearchResultsResponse,
            method="POST",
            data={"id": search_id},
        )
        logging.debug(
            f"Search results fetched, total results: {len(results_response.results)}"
        )
        return results_response

    def _add_payload(
        self, category: BTCategory, save_path: str | None = None
    ) -> tuple[str, dict[str, str]]:
        """Build the shared torrents/add fields and the memory code tagging them.

        The tag is how we find the torrent again later, since qBittorrent does
        not tell us the hash of a torrent it has not fetched yet.
        """
        memory_code = str(KsuidMs())  # Ksuids should be urlsafe
        return memory_code, {
            "savepath": save_path or f"{self.torrent_path}/{category.capitalize()}",
            "category": category.capitalize(),
            "tags": f"sprintboy_{memory_code}",
        }

    @staticmethod
    async def _raise_if_add_failed(response: aiohttp.ClientResponse, target: object) -> None:
        """qBittorrent rejects an add with the body "Fails." under HTTP 200."""
        body = await response.text()
        if response.status != 200 or body == "Fails.":
            logging.error(f"Failed to add {target}: {response.status} {body}")
            raise Exception(f"Failed to add {target}")

    async def add_torrent(self, torrent_url: str, category: BTCategory) -> str | None:
        """Submit a torrent URL to qBittorrent for download."""
        session = self._live_session
        logging.debug(f"Downloading torrent from URL: {torrent_url}")
        memory_code, payload = self._add_payload(category)
        payload["urls"] = torrent_url
        if self.dry_run:
            logging.info(f"Dry run enabled, not downloading torrent: {torrent_url}")
            return
        async with session.post(self._url("/torrents/add"), data=payload) as response:
            await self._raise_if_add_failed(response, torrent_url)
            logging.debug(f"Torrent download initiated successfully for {torrent_url}")
        return memory_code

    async def add_torrent_file(
        self, torrent_path: Path, category: BTCategory, save_path: str | None = None
    ) -> str | None:
        """Submit a local .torrent file to qBittorrent for seeding."""
        session = self._live_session
        memory_code, payload = self._add_payload(category, save_path)
        if self.dry_run:
            logging.info(f"Dry run enabled, not adding torrent file: {torrent_path}")
            return None
        with open(torrent_path, "rb") as f:
            form = aiohttp.FormData()
            for k, v in payload.items():
                form.add_field(k, v)
            form.add_field(
                "torrents",
                f,
                filename=torrent_path.name,
                content_type="application/x-bittorrent",
            )
            async with session.post(self._url("/torrents/add"), data=form) as response:
                await self._raise_if_add_failed(response, torrent_path)
        logging.debug(f"Torrent file submitted successfully: {torrent_path}")
        return memory_code

    async def list_torrents(self, category: BTCategory | None = None) -> list[TorrentInfoResponse]:
        """Return all torrents, optionally filtered to a category, that are fully downloaded."""
        params: dict = {"filter": "completed"}
        if category:
            params["category"] = category.capitalize()
        result = await fetch_url(
            self._live_session, self._url("/torrents/info"), TorrentInfoResponses, params=params
        )
        return result.root

    async def get_torrent_info(self, memory_code: str, timeout: float = 30.0) -> TorrentInfoResponse:
        session = self._live_session
        info_url = self._url("/torrents/info")
        params = {"tag": f"sprintboy_{memory_code}"}
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            result = await fetch_url(session, info_url, TorrentInfoResponses, params=params)
            if result.root:
                return result.root[0]
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Torrent sprintboy_{memory_code} did not appear in qBittorrent "
                    f"within {timeout}s. The torrent URL may be unreachable by qBittorrent "
                    f"(check Jackett indexer authentication)."
                )
            await asyncio.sleep(0.5)


PLEX_LIBRARY_CATEGORIES: dict[str, BTCategory] = {
    "Music": BTCategory.Music,
    "Movies": BTCategory.Movies,
    "TV Shows": BTCategory.TV,
}

# Playlists the tests create start with this, so they can be found again and
# deleted without having to recognise them by content.
TEST_PLAYLIST_PREFIX = "sprintboy-test"
# test_make_playlist used to name its playlist after the track it was built
# from, leaving empty duplicates that look like anything else on the server.
# Those are only ever swept when they hold no items, so a real playlist that
# happens to share the name survives.
LEGACY_TEST_PLAYLIST_TITLES = ("Rapp Snitch Knishes",)


class PlaylistSummary(BaseModel):
    """Just enough of a Plex playlist to list it and delete it."""

    id: str
    title: str
    items: int = 0


def _playlist_summary(raw: dict) -> PlaylistSummary | None:
    """Read a playlist out of a Plex response, skipping anything unusable.

    Plex gives the id twice: as `ratingKey` and inside `key`
    ("/playlists/39244/items"), which is the form `get_playlist` returns.
    """
    playlist_id = str(raw.get("ratingKey") or "").strip()
    if not playlist_id:
        parts = str(raw.get("key", "")).split("/")
        playlist_id = parts[2] if len(parts) > 2 else ""
    title = str(raw.get("title", "")).strip()
    if not playlist_id or not title:
        return None
    try:
        items = int(raw.get("leafCount") or 0)
    except (TypeError, ValueError):
        items = 0
    return PlaylistSummary(id=playlist_id, title=title, items=items)


class PlexAPIClient(AsyncAPIClient):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("PMS_URL")
        self.token = os.getenv("PLEX_TOKEN")
        self.client_id = os.getenv("PLEX_CLIENT_ID")
        self.client_name = os.getenv("PLEX_CLIENT_NAME")
        self.plex_machine_id = os.getenv("PLEX_MACHINE_ID")
        self.music_library_id = os.getenv("PMS_MUSIC_LIBRARY_ID")
        self.movies_library_id = os.getenv("PMS_MOVIES_LIBRARY_ID")
        self.tv_library_id = os.getenv("PMS_TV_LIBRARY_ID")

    @property
    def _headers(self) -> dict[str, str]:
        """Plex wants its token and the caller's identity on every request."""
        assert self.token, "PLEX_TOKEN is not set"
        assert self.client_name, "PLEX_CLIENT_NAME is not set"
        assert self.client_id, "PLEX_CLIENT_ID is not set"
        return {
            "Accept": "application/json",
            "X-Plex-Product": self.client_name,
            "X-Plex-Client-Identifier": self.client_id,
            "X-Plex-Token": self.token,
        }

    async def _request(
        self,
        method: str,
        path: str,
        description: str,
        params: dict | None = None,
        expected_status: int = 200,
    ) -> dict:
        """Call Plex and return the decoded body, or raise a described error.

        `description` is spliced into the failure message, so phrase it as the
        action being attempted: "get libraries", "delete playlist".
        Endpoints that answer with no content return an empty dict.
        """
        session = self._live_session
        url = f"{self.base_url}{path}"
        logging.debug(f"Plex {method} {url} params={params}")
        async with session.request(
            method, url, headers=self._headers, params=params
        ) as response:
            if response.status != expected_status:
                raise Exception(
                    f"Failed to {description}: {response.status}, {await response.text()}"
                )
            try:
                return await response.json()
            except (aiohttp.ContentTypeError, ValueError):
                # Deletes and scan triggers answer with an empty body.
                return {}

    async def get_library_codes(self) -> dict[BTCategory, str]:
        return {
            PLEX_LIBRARY_CATEGORIES[item["title"]]: item["key"]
            for item in await self.get_libraries()
            if item["title"] in PLEX_LIBRARY_CATEGORIES
        }

    async def scan_media(self, content_path: str, category: BTCategory):
        library_code = (await self.get_library_codes())[category]
        await self._request(
            "POST",
            f"/library/sections/{library_code}/refresh",
            "initiate media scan",
            params={"folder": content_path},
        )
        logging.debug(f"Media scan initiated successfully for {content_path}")

    async def get_library_matches(
        self,
        title: str,
        metadata_type: str,
        year: int | None = None,
        parentTitle: str | None = None,
    ) -> dict:
        # TODO: This return model is a bear we'll do this properly later.
        payload = {
            "title": title,
            "type": PLEX_CONTENT_TYPES[metadata_type],
            "includeFullMetadata": 1,
        }
        # This crashes if you send Nones
        if year:
            payload["year"] = year
        if parentTitle:
            payload["parentTitle"] = parentTitle
        result = await self._request(
            "GET", "/library/matches", "get library matches", params=payload
        )
        logging.debug(f"Got library matches: {result}")
        return result

    async def get_all_library_items(self, query_params: dict | None = None) -> dict:
        return await self._request(
            "GET", "/library/all", "get library items", params=query_params
        )

    async def create_playlist(self, title: str, first_item_id: str) -> dict:
        # Yes, this is how it works
        params = {
            "uri": f"server://{self.plex_machine_id}/com.plexapp.plugins.library{first_item_id}",
            "title": title,
            "type": "audio",
            "smart": 0,
        }
        logging.debug(f"Creating playlist: {params}")
        result = await self._request(
            "POST", "/playlists", "create playlist", params=params
        )
        return result["MediaContainer"]["Metadata"][0]

    async def get_playlists(self) -> list[dict]:
        result = await self._request("GET", "/playlists", "get playlists")
        return result["MediaContainer"]["Metadata"]

    async def get_playlist(self, title: str) -> str:
        logging.debug(f"Getting playlist id for {title}")
        playlists = await self.get_playlists()
        playlist_names = [playlist["title"] for playlist in playlists]
        extraction = process.extractOne(title, playlist_names)

        if not extraction:
            raise Exception("Playlist not found")
        closest_playlist_name = extraction[0]
        closest_playlist_score = extraction[1]

        if closest_playlist_score < 90:
            raise Exception("Playlist not found")
        closest_playlist_idx = playlist_names.index(closest_playlist_name)
        return playlists[closest_playlist_idx]["key"].split("/")[2]

    async def find_test_playlists(
        self, prefix: str = TEST_PLAYLIST_PREFIX, include_legacy: bool = True
    ) -> list[PlaylistSummary]:
        """The playlists on the server that the test suite created."""
        found: list[PlaylistSummary] = []
        for raw in await self.get_playlists():
            summary = _playlist_summary(raw)
            if summary is None:
                continue
            if summary.title.casefold().startswith(prefix.casefold()):
                found.append(summary)
            elif (
                include_legacy
                and summary.items == 0
                and summary.title in LEGACY_TEST_PLAYLIST_TITLES
            ):
                found.append(summary)
        return found

    async def delete_playlists(self, playlists: Sequence[PlaylistSummary]) -> int:
        """Delete the given playlists, carrying on past ones already gone."""
        deleted = 0
        for playlist in playlists:
            try:
                await self.delete_playlist(playlist.id)
            except Exception as e:
                logging.warning(f"Could not delete playlist {playlist.title}: {e}")
                continue
            logging.info(f"Deleted playlist {playlist.title} ({playlist.id})")
            deleted += 1
        return deleted

    async def delete_playlist(self, playlist_id: int | str) -> None:
        await self._request(
            "DELETE",
            f"/playlists/{playlist_id}",
            "delete playlist",
            expected_status=204,
        )

    async def get_libraries(self) -> list[dict]:
        result = await self._request("GET", "/library/sections", "get libraries")
        return result["MediaContainer"]["Directory"]


LASTFM_API_ROOT = "https://ws.audioscrobbler.com/2.0/"


class LastFMError(Exception):
    """Raised when Last.fm reports an API-level error, or when no key is configured."""


class LastFMArtist(BaseModel):
    name: str
    mbid: str | None = None
    url: str | None = None
    # Similarity score 0-1, only present on artist.getSimilar results.
    match: float | None = None
    listeners: int | None = None


class LastFMAlbum(BaseModel):
    name: str
    artist: str | None = None
    mbid: str | None = None
    url: str | None = None
    playcount: int | None = None


class LastFMArtistInfo(BaseModel):
    name: str
    mbid: str | None = None
    url: str | None = None
    listeners: int | None = None
    tags: List[str] = []
    similar: List[str] = []
    summary: str | None = None


class LastFMAlbumInfo(BaseModel):
    name: str
    artist: str | None = None
    mbid: str | None = None
    url: str | None = None
    listeners: int | None = None
    tags: List[str] = []
    tracks: List[str] = []


class CanonicalRelease(BaseModel):
    """Last.fm's canonical spelling for a release, plus the spellings worth querying Plex with.

    Last.fm has no alias endpoint, so `artist_candidates` / `album_candidates` are
    the distinct spellings we actually know about (what the user typed, what
    artist.getCorrection returned, and what album.getInfo echoed back) rather than
    a full alias list. Querying Plex with all of them is what stops "Guns and
    Roses" from looking absent when the library holds "Guns N' Roses".
    """

    artist: str
    artist_candidates: List[str]
    artist_mbid: str | None = None
    album: str | None = None
    album_candidates: List[str] = []
    album_mbid: str | None = None
    corrected: bool = False


def _lastfm_list(value) -> list[dict]:
    """Normalise a Last.fm collection field into a list of dicts.

    Last.fm collapses a single-element collection into a bare object and an empty
    one into "" (sometimes a lone newline), so every list-shaped field has three
    possible JSON shapes.
    """
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _lastfm_dict(value) -> dict:
    """Normalise a Last.fm sub-object field, which is "" when absent."""
    return value if isinstance(value, dict) else {}


def _lastfm_str(value) -> str | None:
    """Last.fm sends "" rather than null for absent mbids, urls and bios."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _lastfm_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _lastfm_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _lastfm_names(value) -> list[str]:
    """Pull the `name` field out of a collection field (tags, similar artists)."""
    names = []
    for item in _lastfm_list(value):
        name = _lastfm_str(item.get("name"))
        if name:
            names.append(name)
    return names


def _parse_lastfm_artist(raw: dict) -> LastFMArtist | None:
    name = _lastfm_str(raw.get("name"))
    if not name:
        return None
    return LastFMArtist(
        name=name,
        mbid=_lastfm_str(raw.get("mbid")),
        url=_lastfm_str(raw.get("url")),
        match=_lastfm_float(raw.get("match")),
        listeners=_lastfm_int(raw.get("listeners")),
    )


def _parse_lastfm_artists(container: dict) -> list[LastFMArtist]:
    """Parse the `artist` collection out of a Last.fm response container."""
    parsed = (_parse_lastfm_artist(raw) for raw in _lastfm_list(container.get("artist")))
    return [artist for artist in parsed if artist]


def _parse_lastfm_albums(container: dict) -> list[LastFMAlbum]:
    """Parse the `album` collection out of a Last.fm response container."""
    parsed = (_parse_lastfm_album(raw) for raw in _lastfm_list(container.get("album")))
    return [album for album in parsed if album]


def _parse_lastfm_album(raw: dict) -> LastFMAlbum | None:
    name = _lastfm_str(raw.get("name"))
    if not name:
        return None
    # On artist.getTopAlbums the artist is a nested object; on album.getInfo and
    # album.search it is a plain string.
    artist = raw.get("artist")
    artist_name = (
        _lastfm_str(artist.get("name")) if isinstance(artist, dict) else _lastfm_str(artist)
    )
    return LastFMAlbum(
        name=name,
        artist=artist_name,
        mbid=_lastfm_str(raw.get("mbid")),
        url=_lastfm_str(raw.get("url")),
        playcount=_lastfm_int(raw.get("playcount")),
    )


def dedupe_casefold(values: Sequence[str | None]) -> list[str]:
    """Drop blanks and case-insensitive duplicates, keeping the first spelling seen."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _lastfm_str(value)
        if cleaned is None or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        result.append(cleaned)
    return result


class LastFMClient(AsyncAPIClient):
    """Read-only Last.fm client for artist/album metadata and recommendations.

    Last.fm answers most failures with HTTP 200 and an {"error": N} body, so
    requests go through `_get` rather than `fetch_url`, which only checks status.
    """

    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("LASTFM_API_URL", LASTFM_API_ROOT)
        self.api_key = os.getenv("LASTFM_API_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _on_connect(self) -> None:
        if not self.api_key:
            raise LastFMError(
                "LASTFM_API_KEY is not set. Create one at "
                "https://www.last.fm/api/account/create and add it to .env"
            )

    async def _get(self, method: str, **params) -> dict:
        session = self._live_session
        query = {"method": method, "api_key": self.api_key, "format": "json"}
        query.update({k: str(v) for k, v in params.items() if v is not None})
        logging.debug(f"Last.fm request {method} {params}")
        # Error bodies come back as text/xml even with format=json, so don't let
        # aiohttp enforce a content type.
        async with session.get(self.base_url, params=query) as response:
            body = await response.json(content_type=None)
            if isinstance(body, dict) and "error" in body:
                raise LastFMError(
                    f"Last.fm {method} failed ({body['error']}): "
                    f"{body.get('message', 'no message')}"
                )
            response.raise_for_status()
            if not isinstance(body, dict):
                raise LastFMError(f"Last.fm {method} returned an unexpected payload")
            return body

    async def correct_artist(self, artist: str) -> LastFMArtist | None:
        """Return Last.fm's canonical spelling of an artist, or None if it has no correction."""
        body = await self._get("artist.getCorrection", artist=artist)
        corrections = body.get("corrections")
        # "No correction available" is an empty string, not null or an empty object.
        if not isinstance(corrections, dict):
            return None
        for correction in _lastfm_list(corrections.get("correction")):
            raw = correction.get("artist")
            if isinstance(raw, dict):
                parsed = _parse_lastfm_artist(raw)
                if parsed:
                    return parsed
        return None

    async def get_artist_info(self, artist: str) -> LastFMArtistInfo:
        body = await self._get("artist.getInfo", artist=artist, autocorrect=1)
        raw = body.get("artist")
        if not isinstance(raw, dict):
            raise LastFMError(f"Last.fm returned no artist for {artist!r}")
        stats = _lastfm_dict(raw.get("stats"))
        bio = _lastfm_dict(raw.get("bio"))
        similar = _lastfm_dict(raw.get("similar"))
        tags = _lastfm_dict(raw.get("tags"))
        return LastFMArtistInfo(
            name=_lastfm_str(raw.get("name")) or artist,
            mbid=_lastfm_str(raw.get("mbid")),
            url=_lastfm_str(raw.get("url")),
            listeners=_lastfm_int(stats.get("listeners")),
            tags=_lastfm_names(tags.get("tag")),
            similar=_lastfm_names(similar.get("artist")),
            summary=_lastfm_str(bio.get("summary")),
        )

    async def get_similar_artists(
        self, artist: str, limit: int = 20
    ) -> list[LastFMArtist]:
        body = await self._get(
            "artist.getSimilar", artist=artist, limit=limit, autocorrect=1
        )
        return _parse_lastfm_artists(_lastfm_dict(body.get("similarartists")))

    async def get_artist_albums(
        self, artist: str, limit: int = 20, page: int = 1
    ) -> list[LastFMAlbum]:
        """Albums by an artist, most listened first.

        This is artist.getTopAlbums, which is the closest thing Last.fm has to a
        discography: raise `limit` to approximate one, at the cost of pulling in
        singles and compilations further down the list.
        """
        body = await self._get(
            "artist.getTopAlbums",
            artist=artist,
            limit=limit,
            page=page,
            autocorrect=1,
        )
        albums = _parse_lastfm_albums(_lastfm_dict(body.get("topalbums")))
        # Last.fm uses the literal string "(null)" for unnamed releases.
        return [
            album
            for album in albums
            if album.name.casefold() not in {"(null)", "null", "undefined"}
        ]

    async def get_album_info(self, artist: str, album: str) -> LastFMAlbumInfo:
        body = await self._get(
            "album.getInfo", artist=artist, album=album, autocorrect=1
        )
        raw = body.get("album")
        if not isinstance(raw, dict):
            raise LastFMError(f"Last.fm returned no album for {artist!r} - {album!r}")
        tags = _lastfm_dict(raw.get("tags"))
        tracks = _lastfm_dict(raw.get("tracks"))
        return LastFMAlbumInfo(
            name=_lastfm_str(raw.get("name")) or album,
            artist=_lastfm_str(raw.get("artist")),
            mbid=_lastfm_str(raw.get("mbid")),
            url=_lastfm_str(raw.get("url")),
            listeners=_lastfm_int(raw.get("listeners")),
            tags=_lastfm_names(tags.get("tag")),
            tracks=_lastfm_names(tracks.get("track")),
        )

    async def get_tag_top_artists(self, tag: str, limit: int = 20) -> list[LastFMArtist]:
        body = await self._get("tag.getTopArtists", tag=tag, limit=limit)
        return _parse_lastfm_artists(_lastfm_dict(body.get("topartists")))

    async def get_tag_top_albums(self, tag: str, limit: int = 20) -> list[LastFMAlbum]:
        body = await self._get("tag.getTopAlbums", tag=tag, limit=limit)
        # The XML schema calls this `topalbums` but the JSON feed calls it
        # `albums`; accept either so a Last.fm change doesn't silently empty this.
        return _parse_lastfm_albums(
            _lastfm_dict(body.get("albums") or body.get("topalbums"))
        )

    async def get_top_tags(self, limit: int = 500) -> list[tuple[str, int]]:
        """Last.fm's most used tags as (name, reach), most reached first."""
        body = await self._get("chart.getTopTags", limit=limit)
        tags = []
        for raw in _lastfm_list(_lastfm_dict(body.get("tags")).get("tag")):
            name = _lastfm_str(raw.get("name"))
            if not name:
                continue
            reach = _lastfm_int(raw.get("reach"))
            if reach is None:
                reach = _lastfm_int(raw.get("taggings")) or 0
            tags.append((name, reach))
        return sorted(tags, key=lambda t: -t[1])

    async def search_album(self, album: str) -> LastFMAlbum | None:
        """The best Last.fm match for an album name, or None."""
        body = await self._get("album.search", album=album, limit=1)
        results = _lastfm_dict(body.get("results"))
        found = _parse_lastfm_albums(_lastfm_dict(results.get("albummatches")))
        return found[0] if found else None


async def resolve_release(artist: str, album: str | None = None) -> CanonicalRelease:
    """Resolve a user-supplied artist/album to Last.fm's canonical spelling.

    Raises LastFMError if Last.fm is unconfigured or unreachable; callers that
    only want canonicalisation as a bonus should catch it and fall back to the
    names they were given.
    """
    async with LastFMClient() as lastfm:
        correction = await lastfm.correct_artist(artist)
        canonical_artist = correction.name if correction else artist
        artist_mbid = correction.mbid if correction else None
        artist_candidates = [canonical_artist, artist]

        canonical_album = album
        album_mbid = None
        album_candidates: list[str] = []

        if album:
            try:
                info = await lastfm.get_album_info(canonical_artist, album)
            except LastFMError:
                # An unknown album is not a failure to resolve the artist.
                logging.debug(f"Last.fm has no album {album!r} by {canonical_artist!r}")
                info = None
            if info:
                canonical_album = info.name
                album_mbid = info.mbid
                if info.artist:
                    # album.getInfo echoes the release's own artist credit, which
                    # is the spelling most likely to match a tagged library.
                    canonical_artist = info.artist
                    artist_candidates.insert(0, info.artist)
            album_candidates = dedupe_casefold([canonical_album, album])

    return CanonicalRelease(
        artist=canonical_artist,
        artist_candidates=dedupe_casefold(artist_candidates),
        artist_mbid=artist_mbid,
        album=canonical_album,
        album_candidates=album_candidates,
        album_mbid=album_mbid,
        corrected=bool(correction) and correction.name.casefold() != artist.casefold(),
    )
