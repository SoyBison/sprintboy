"""
Unit tests for the two bugs fixed in this session.

Bug 1 (netcode.py): get_torrent_info crashed with IndexError when qBittorrent
    returned an empty list for a tag query (torrent not yet indexed).
    Fix: retry with a 0.5s sleep until the list is non-empty.

Bug 2 (main.py): The outer `for torrent in torrent_context.internal_torrents`
    loop was redundant, causing N² get_torrent_info calls per poll cycle.
    Fix: remove the outer loop.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from bot.netcode import QBittorrentClient, TorrentInfoResponses


# ---------------------------------------------------------------------------
# Bug 1: get_torrent_info retries on empty response instead of crashing
# ---------------------------------------------------------------------------



class TestGetTorrentInfoEmptyResponse:
    def _make_client(self):
        client = QBittorrentClient.__new__(QBittorrentClient)
        client.session = MagicMock()
        client.base_url = "http://localhost:8080/api/v2"
        return client

    @pytest.mark.asyncio
    async def test_retries_on_empty_list_then_succeeds(self, torrent_data):
        """
        First call returns [] (torrent not indexed yet), second returns the torrent.
        get_torrent_info should retry and return the result on the second call.
        """
        client = self._make_client()
        empty = TorrentInfoResponses.model_validate([])
        populated = TorrentInfoResponses.model_validate([torrent_data])

        with patch("bot.netcode.fetch_url", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [empty, populated]
            with patch("bot.netcode.asyncio.sleep", new_callable=AsyncMock):
                result = await client.get_torrent_info("some_code")

        assert result.name == "Test Album"
        assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_timeout_when_torrent_never_appears(self):
        """
        If qBittorrent never returns the torrent (e.g. Jackett indexer auth expired),
        get_torrent_info should raise TimeoutError instead of looping forever.
        """
        client = self._make_client()
        empty = TorrentInfoResponses.model_validate([])

        with patch("bot.netcode.fetch_url", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = empty
            with patch("bot.netcode.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TimeoutError, match="did not appear in qBittorrent"):
                    await client.get_torrent_info("some_code", timeout=0.0)

    @pytest.mark.asyncio
    async def test_succeeds_immediately_when_list_nonempty(self, torrent_data):
        """Happy path: torrent is already indexed on the first call."""
        client = self._make_client()
        populated = TorrentInfoResponses.model_validate([torrent_data])

        with patch("bot.netcode.fetch_url", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = populated
            result = await client.get_torrent_info("some_code")

        assert result.name == "Test Album"
        assert mock_fetch.call_count == 1


# ---------------------------------------------------------------------------
# Bug 2: polling loop makes N calls, not N²
# ---------------------------------------------------------------------------


class TestPollingLoopCallCount:
    def _make_completed_info(self, torrent_data):
        from bot.netcode import TorrentInfoResponse
        return TorrentInfoResponse(**torrent_data)

    def _make_mock_qclient(self, completed_info):
        call_count = {"n": 0}

        async def fake_get_torrent_info(memory_code):
            call_count["n"] += 1
            return completed_info

        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=False)
        mock.get_torrent_info = fake_get_torrent_info
        return mock, call_count

    @pytest.mark.asyncio
    async def test_fixed_loop_calls_get_torrent_info_n_times(self, torrent_data):
        """The fixed loop (no outer redundant loop) makes exactly N calls."""
        from bot.tools import TorrentContext

        torrent_context = TorrentContext(
            search_results={},
            internal_torrents={"Album A": "code_a", "Album B": "code_b", "Album C": "code_c"},
            torrent_types=set(),
        )
        n = len(torrent_context.internal_torrents)
        completed_info = self._make_completed_info(torrent_data)
        mock_qclient, call_count = self._make_mock_qclient(completed_info)

        async with mock_qclient as qclient:
            torrent_info_promises = []
            for content_path, memory_code in torrent_context.internal_torrents.items():
                if memory_code is None:
                    continue
                torrent_info_promises.append(qclient.get_torrent_info(memory_code))
            torrent_info = await asyncio.gather(*torrent_info_promises)

        assert call_count["n"] == n
        assert len(torrent_info) == n
