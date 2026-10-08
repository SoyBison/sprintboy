from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot import aotm
from bot.config import Config
from bot.orpheus import AotmPick, AotmTorrent

PICK = AotmPick("September Round 2", "Ichiko Aoba", "Windswept Adan",
                datetime(2026, 10, 1, tzinfo=timezone.utc), 219)
TORRENT = AotmTorrent(1663183, 764376, "Ichiko Aoba", '"Windswept Adan" Live', 2021,
                      "Live album", "24bit Lossless", "WEB", 462, 759747616, True)


@pytest.mark.asyncio
async def test_check_and_ask_once(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "AOTM_STATE_PATH", str(tmp_path / "sub" / "aotm.json"))
    monkeypatch.setattr(Config, "ORPHEUS_API_KEY", "k")
    monkeypatch.setattr(aotm, "current_pick", AsyncMock(return_value=(PICK, TORRENT)))
    monkeypatch.setattr(aotm, "OrpheusClient", MagicMock())
    monkeypatch.setattr(
        "bot.tools.check_for_album",
        MagicMock(ainvoke=AsyncMock(return_value="The user has no albums by Ichiko Aoba.")),
    )
    owner = MagicMock()
    owner.send = AsyncMock(return_value=MagicMock(id=42))

    assert await aotm.check_and_ask(MagicMock(), owner) == "asked"
    owner.send.assert_awaited_once()
    assert "You don't seem to have anything by Ichiko Aoba" in owner.send.call_args.args[0]
    state = aotm.load_state()
    assert state["219"]["status"] == "asked"
    assert state["219"]["message_id"] == 42

    assert await aotm.check_and_ask(MagicMock(), owner) == "already asked"
    owner.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_key(monkeypatch):
    monkeypatch.setattr(Config, "ORPHEUS_API_KEY", "")
    assert await aotm.check_and_ask(MagicMock(), MagicMock()) == "no key"


@pytest.mark.asyncio
async def test_custom_id_round_trip():
    button = aotm.AotmButton("grab", 1663183, "219")
    cid = button.item.custom_id
    match = aotm.AotmButton.__discord_ui_compiled_template__.fullmatch(cid)
    assert match
    rebuilt = await aotm.AotmButton.from_custom_id(None, button.item, match)
    assert (rebuilt.action, rebuilt.torrent_id, rebuilt.key) == ("grab", 1663183, "219")
    assert rebuilt.item.custom_id == cid
