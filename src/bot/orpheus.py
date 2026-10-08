"""Orpheus (OPS) API client and Album of the Month parsing."""

import asyncio
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiohttp

from bot.config import Config

# OPS allows 5 requests per 10 s; stay comfortably under it.
MIN_REQUEST_GAP = 2.1
_rate_lock = asyncio.Lock()
_last_request = 0.0


class OrpheusError(Exception):
    """Orpheus said no, or could not be reached."""


class OrpheusClient:
    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"token {Config.ORPHEUS_API_KEY}"}
        )
        return self

    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
            self.session = None

    async def _get(self, params: dict) -> tuple[aiohttp.ClientResponse, bytes]:
        global _last_request
        if self.session is None:
            raise RuntimeError("OrpheusClient must be used with `async with`")
        url = f"{Config.ORPHEUS_URL.rstrip('/')}/ajax.php"
        async with _rate_lock:
            wait = _last_request + MIN_REQUEST_GAP - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await self.session.get(url, params=params)
                body = await resp.read()
            except aiohttp.ClientError as e:
                raise OrpheusError(f"Couldn't reach Orpheus: {e}") from e
            finally:
                _last_request = time.monotonic()
        return resp, body

    @staticmethod
    def _unwrap(data) -> object:
        if not isinstance(data, dict) or data.get("status") != "success":
            error = data.get("error") if isinstance(data, dict) else data
            raise OrpheusError(f"Orpheus error: {error}")
        return data["response"]

    async def _json(self, params: dict):
        resp, body = await self._get(params)
        try:
            data = json.loads(body)
        except ValueError as e:
            raise OrpheusError(f"Orpheus returned non-JSON (HTTP {resp.status})") from e
        return self._unwrap(data)

    async def announcements(self) -> dict:
        """The raw announcements response, `{"status": ..., "response": ...}`."""
        resp = await self._json({"action": "announcements"})
        return {"status": "success", "response": resp}

    async def browse(self, searchstr: str) -> list[dict]:
        response = await self._json({"action": "browse", "searchstr": searchstr})
        return response["results"]

    async def download_torrent(self, torrent_id: int) -> bytes:
        resp, body = await self._get({"action": "download", "id": torrent_id})
        if "json" in resp.headers.get("Content-Type", "").lower():
            try:
                data = json.loads(body)
            except ValueError as e:
                raise OrpheusError("Orpheus returned unreadable JSON") from e
            self._unwrap(data)
            raise OrpheusError("Orpheus returned JSON instead of a torrent file")
        if resp.status != 200:
            raise OrpheusError(f"Download failed with HTTP {resp.status}")
        return body


@dataclass(frozen=True)
class AotmPick:
    round: str  # e.g. "September Round 2"
    artist: str
    album: str
    announced: datetime  # UTC
    news_id: int

    @property
    def key(self) -> str:
        return f"{self.news_id}"

    @property
    def freeleech_until(self) -> datetime:
        return self.announced + timedelta(days=Config.AOTM_FREELEECH_DAYS)


_WINNER = re.compile(r"AoTM\s+(?P<round>.+?)\s+Winner\s+-\s+(?P<rest>.+)")


def _plain(body: str) -> list[str]:
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", body))
    return [line.strip() for line in text.splitlines() if line.strip()]


def _announced(item: dict) -> datetime:
    news_time = item["newsTime"]
    stamp = datetime.fromisoformat(news_time["date"])
    if stamp.tzinfo is None:
        offset = news_time.get("timezone", "+00:00")
        stamp = stamp.replace(tzinfo=datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").tzinfo)
    return stamp.astimezone(timezone.utc)


def parse_aotm(announcements: dict) -> list[AotmPick]:
    """Album of the Month winners in the announcements, newest first."""
    picks = []
    for item in announcements.get("response", {}).get("announcements", []):
        try:
            title = item["title"]
            if "AoTM" not in title or "Winner" not in title:
                continue
            lines = _plain(item["body"])
            start = next(i for i, ln in enumerate(lines) if "has closed" in ln)
            m = _WINNER.search(lines[start + 1])
            artist, album = m["rest"].split(" - ", 1)
            picks.append(
                AotmPick(
                    round=m["round"].strip(),
                    artist=artist.strip(),
                    album=album.strip(),
                    announced=_announced(item),
                    news_id=int(item["newsId"]),
                )
            )
        except (KeyError, ValueError, StopIteration, TypeError, IndexError, AttributeError):
            continue
    picks.sort(key=lambda p: p.announced, reverse=True)
    return picks


@dataclass(frozen=True)
class AotmTorrent:
    torrent_id: int
    group_id: int
    artist: str
    group_name: str
    year: int
    release_type: str
    encoding: str
    media: str
    seeders: int
    size: int
    freeleech: bool

    @property
    def label(self) -> str:
        return (
            f"{self.artist} - {self.group_name} [{self.year}] "
            f"FLAC {self.encoding} {self.media}"
        )


def _rank(torrent: dict) -> tuple:
    media = torrent.get("media", "")
    if media == "SACD":
        source = 4
    elif media == "CD" and torrent.get("logScore") == 100 and torrent.get("hasCue"):
        source = 3
    elif media == "WEB":
        source = 2
    elif media == "Vinyl":
        source = 0
    else:
        source = 1
    deep = 1 if "24bit" in torrent.get("encoding", "") else 0
    return (source, deep, torrent.get("seeders", 0))


def pick_freeleech_torrent(results: list[dict]) -> AotmTorrent | None:
    best = None
    for group in results:
        for t in group.get("torrents", []):
            if t.get("format") != "FLAC" or not t.get("isFreeleech"):
                continue
            if best is None or _rank(t) > _rank(best[1]):
                best = (group, t)
    if best is None:
        return None
    g, t = best
    return AotmTorrent(
        torrent_id=int(t["torrentId"]),
        group_id=int(g["groupId"]),
        artist=html.unescape(g.get("artist", "")),
        group_name=html.unescape(g.get("groupName", "")),
        year=int(g.get("groupYear") or 0),
        release_type=g.get("releaseType", ""),
        encoding=t.get("encoding", ""),
        media=t.get("media", ""),
        seeders=int(t.get("seeders", 0)),
        size=int(t.get("size", 0)),
        freeleech=True,
    )


async def current_pick(
    client: OrpheusClient, now: datetime | None = None
) -> tuple[AotmPick, AotmTorrent | None] | None:
    """The newest winner that is still freeleech, and the torrent to offer."""
    now = now or datetime.now(timezone.utc)
    picks = [p for p in parse_aotm(await client.announcements()) if p.freeleech_until > now]
    if not picks:
        return None
    pick = picks[0]
    torrent = pick_freeleech_torrent(await client.browse(f"{pick.artist} {pick.album}"))
    if torrent is None:
        torrent = pick_freeleech_torrent(await client.browse(pick.artist))
    return pick, torrent
