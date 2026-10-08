import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bot.orpheus import current_pick, parse_aotm, pick_freeleech_torrent

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_parse_aotm_newest_first():
    picks = parse_aotm(load("ops_announcements.json"))
    assert len(picks) == 5
    top = picks[0]
    assert top.round == "September Round 2"
    assert top.artist == "Ichiko Aoba"
    assert top.album == "Windswept Adan"
    assert top.announced.date() == datetime(2026, 10, 1).date()
    assert top.announced.tzinfo is not None
    assert top.key == "219"
    assert top.freeleech_until.day == 15


def test_pick_freeleech_torrent():
    results = load("ops_browse_aotm.json")["response"]["results"]
    t = pick_freeleech_torrent(results)
    assert t.group_id == 764376
    assert t.torrent_id == 1663183
    assert t.encoding == "24bit Lossless"
    assert t.media == "WEB"
    assert t.freeleech
    assert t.label.startswith('Ichiko Aoba - "Windswept Adan')
    assert t.label.endswith("[2021] FLAC 24bit Lossless WEB")


def test_no_freeleech_returns_none():
    results = copy.deepcopy(load("ops_browse_aotm.json")["response"]["results"])
    for g in results:
        for t in g["torrents"]:
            t["isFreeleech"] = False
    assert pick_freeleech_torrent(results) is None


@pytest.mark.asyncio
async def test_current_pick_expires():
    client = AsyncMock()
    client.announcements.return_value = load("ops_announcements.json")
    client.browse.return_value = load("ops_browse_aotm.json")["response"]["results"]

    soon = datetime(2026, 10, 5, tzinfo=timezone.utc)
    pick, torrent = await current_pick(client, now=soon)
    assert pick.artist == "Ichiko Aoba"
    assert torrent.torrent_id == 1663183

    late = datetime(2026, 10, 20, tzinfo=timezone.utc)
    assert await current_pick(client, now=late) is None


def test_freeleech_torrent_carries_the_cover():
    import json
    from pathlib import Path

    from bot.orpheus import pick_freeleech_torrent

    results = json.loads(
        (Path(__file__).parent / "fixtures" / "ops_browse_aotm.json").read_text()
    )["response"]["results"]
    assert pick_freeleech_torrent(results).cover.startswith("https://")
