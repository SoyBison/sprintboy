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

    async def _web(
        self, method: str, path: str, data: dict | None = None
    ) -> tuple[int, dict, str]:
        """A rate-limited request to the website (not ajax.php), logged in by cookie.

        Redirects are not followed. Returns (status, headers, body text).
        """
        global _last_request
        if self.session is None:
            raise RuntimeError("OrpheusClient must be used with `async with`")
        url = f"{Config.ORPHEUS_URL.rstrip('/')}/{path}"
        headers = {"Cookie": f"session={Config.ORPHEUS_SESSION_COOKIE}"}
        async with _rate_lock:
            wait = _last_request + MIN_REQUEST_GAP - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await self.session.request(
                    method, url, data=data, headers=headers, allow_redirects=False
                )
                body = (await resp.read()).decode("utf-8", errors="replace")
            except aiohttp.ClientError as e:
                raise OrpheusError(f"Couldn't reach Orpheus: {e}") from e
            finally:
                _last_request = time.monotonic()
        return resp.status, dict(resp.headers), body

    async def account(self) -> tuple["AccountStats", str]:
        """The user's stats, and their authkey (needed to buy; keep it private)."""
        response = await self._json({"action": "index"})
        try:
            stats = response["userstats"]
            return (
                AccountStats(
                    username=str(response["username"]),
                    uploaded=int(stats["uploaded"]),
                    downloaded=int(stats["downloaded"]),
                    ratio=float(stats["ratio"]),
                    required_ratio=float(stats["requiredratio"]),
                    bonus_points=int(stats["bonusPoints"]),
                    bonus_per_hour=float(stats["bonusPointsPerHour"]),
                    tokens=int(stats["tokens"]),
                    user_class=str(stats.get("class", "")),
                ),
                str(response["authkey"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise OrpheusError("Orpheus returned an unexpected account response") from e

    async def token_shop(self) -> list["ShopItem"]:
        """The freeleech token packs on sale in the bonus shop."""
        status, headers, body = await self._web("GET", "bonus.php")
        if 300 <= status < 400 and "login" in headers.get("Location", ""):
            raise OrpheusError(_SESSION_ERROR)
        if status in (401, 403):
            raise OrpheusError(_SESSION_ERROR)
        if 'name="username"' in body and 'name="password"' in body:
            raise OrpheusError(_SESSION_ERROR)
        if status != 200:
            raise OrpheusError(f"Orpheus bonus shop returned HTTP {status}")
        return parse_token_shop(body)

    async def buy(self, label: str, authkey: str) -> None:
        """Buy one shop item. Raises OrpheusError unless Orpheus confirms it."""
        status, headers, body = await self._web(
            "POST", "bonus.php", {"auth": authkey, "action": "purchase", "label": label}
        )
        if 300 <= status < 400 and "complete=" in headers.get("Location", ""):
            return
        if 300 <= status < 400 and "login" in headers.get("Location", ""):
            raise OrpheusError(_SESSION_ERROR)
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split())
        raise OrpheusError(text[:200] or f"HTTP {status}")

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


_SESSION_ERROR = "Orpheus session cookie is missing or expired"


@dataclass
class AccountStats:
    username: str
    uploaded: int
    downloaded: int
    ratio: float
    required_ratio: float
    bonus_points: int
    bonus_per_hour: float
    tokens: int
    user_class: str

    @property
    def exact_ratio(self) -> float:
        """Uploaded / downloaded. The API's `ratio` is rounded to two places,
        which hides the difference between 0.6049 and 0.5951."""
        return self.uploaded / self.downloaded if self.downloaded else float("inf")

    @property
    def headroom(self) -> int:
        """Bytes that can still be downloaded (not freeleech) before the ratio
        drops below the required one; negative when already below it."""
        if not self.required_ratio:
            return 1 << 62
        return int(self.uploaded / self.required_ratio) - self.downloaded


@dataclass(frozen=True)
class ShopItem:
    label: str  # "token-1"
    title: str  # "1 Freeleech Token"
    tokens: int
    price: int


_ROW = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_LABEL = re.compile(r"""name=["'](token-[\w-]+)["']|item=(token-[\w-]+)""", re.I)
_PRICE = re.compile(
    r"""<td[^>]*text-align:\s*right[^>]*>\s*([\d,]+)\s*</td>""", re.S | re.I
)


def parse_token_shop(page: str) -> list[ShopItem]:
    """Freeleech token rows from the bonus shop page (affordable or not)."""
    items = []
    for row in _ROW.findall(page):
        label = _LABEL.search(row)
        price = _PRICE.search(row)
        if not label or not price:
            continue
        cells = [
            " ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
            for c in re.findall(r"<td\b.*?</td>", row, re.S | re.I)
        ]
        title = next((c for c in cells if re.match(r"\d+\s", c)), "")
        count = re.match(r"(\d+)", title)
        if not count:
            continue
        items.append(
            ShopItem(
                label=(label.group(1) or label.group(2)).lower(),
                title=title,
                tokens=int(count.group(1)),
                price=int(price.group(1).replace(",", "")),
            )
        )
    return items


MAX_PLAN_TOKENS = 200


def plan_purchase(items: list[ShopItem], count: int | None, budget: int) -> list[ShopItem]:
    """Cheapest items reaching exactly `count` tokens within budget.

    `count=None` maximises tokens within budget (ties: cheaper). [] if impossible.
    """
    usable = [i for i in items if 0 < i.tokens <= MAX_PLAN_TOKENS and i.price >= 0]
    if not usable or budget < 0:
        return []
    top = MAX_PLAN_TOKENS if count is None else count
    if count is not None and (count < 1 or count > MAX_PLAN_TOKENS):
        return []
    # best[n] = (cost, items) for exactly n tokens
    best: list[tuple[int, list[ShopItem]] | None] = [None] * (top + 1)
    best[0] = (0, [])
    for n in range(1, top + 1):
        for item in usable:
            prev = n - item.tokens
            if prev < 0 or best[prev] is None:
                continue
            cost = best[prev][0] + item.price
            if best[n] is None or cost < best[n][0]:
                best[n] = (cost, [*best[prev][1], item])
    if count is not None:
        found = best[count]
        return found[1] if found and found[0] <= budget else []
    for n in range(top, 0, -1):
        if best[n] is not None and best[n][0] <= budget:
            return best[n][1]
    return []


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
    cover: str = ""

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
        cover=str(g.get("cover") or ""),
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
