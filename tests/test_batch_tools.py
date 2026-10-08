"""Tests for the batch tools and the library-ownership annotations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.netcode import BTCategory, LastFMAlbum, TorrentInfoResponse
from bot.tools import (
    add_torrent,
    check_albums,
    lastfm_artist_albums,
    search_for_torrent,
)
from test_silent_success import _context, _mock_qclient


def _runtime(ctx):
    return SimpleNamespace(context=ctx)


def _search_result(name):
    return _context(name).search_results[name]


class TestAddTorrentBatch:
    @pytest.mark.asyncio
    async def test_two_names_are_both_added(self, torrent_data):
        ctx = _context("Alpha One [FLAC]", "Beta Two [FLAC]")
        client = _mock_qclient(TorrentInfoResponse(**torrent_data))
        with patch("bot.tools.QBittorrentClient", return_value=client):
            result = await add_torrent.coroutine(
                names=["Alpha One [FLAC]", "Beta Two [FLAC]"],
                category=BTCategory.Music,
                runtime=_runtime(ctx),
            )
        assert result.count("Added") == 2
        assert set(ctx.internal_torrents) == {"Alpha One [FLAC]", "Beta Two [FLAC]"}

    @pytest.mark.asyncio
    async def test_bad_name_is_not_added_but_good_one_is(self, torrent_data):
        ctx = _context("Alpha One [FLAC]")
        client = _mock_qclient(TorrentInfoResponse(**torrent_data))
        with patch("bot.tools.QBittorrentClient", return_value=client):
            result = await add_torrent.coroutine(
                names=["Alpha One [FLAC]", "zzzzzzzzzz qqqqqqq"],
                category=BTCategory.Music,
                runtime=_runtime(ctx),
            )
        lines = result.split("\n")
        assert lines[0].startswith("Added")
        assert lines[1].startswith("NOT ADDED")
        assert list(ctx.internal_torrents) == ["Alpha One [FLAC]"]

    @pytest.mark.asyncio
    async def test_duplicate_resolution_adds_once(self, torrent_data):
        ctx = _context("Alpha One [FLAC]")
        client = _mock_qclient(TorrentInfoResponse(**torrent_data))
        with patch("bot.tools.QBittorrentClient", return_value=client):
            await add_torrent.coroutine(
                names=["Alpha One [FLAC]", "Alpha One FLAC"],
                category=BTCategory.Music,
                runtime=_runtime(ctx),
            )
        assert client.add_torrent.await_count == 1

    def test_bare_string_becomes_list(self):
        args = add_torrent.args_schema(names="X", category=BTCategory.Music)
        assert args.names == ["X"]


def _search_client(by_query):
    def make():
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        async def search(query, category):
            outcome = by_query[query]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(results=outcome)

        client.search = search
        return client

    return make


class TestSearchBatch:
    @pytest.mark.asyncio
    async def test_two_queries_store_and_report_both(self):
        ctx = _context()
        by_query = {
            "A One": [_search_result("A One [FLAC]")],
            "B Two": [_search_result("B Two [FLAC]")],
        }
        with patch("bot.tools.QBittorrentClient", side_effect=_search_client(by_query)):
            out = await search_for_torrent.coroutine(
                queries=["A One", "B Two"],
                category=BTCategory.Music,
                runtime=_runtime(ctx),
            )
        assert "Results for 'A One':" in out
        assert "Results for 'B Two':" in out
        assert set(ctx.search_results) == {"A One [FLAC]", "B Two [FLAC]"}

    @pytest.mark.asyncio
    async def test_one_failing_query_does_not_fail_the_other(self):
        ctx = _context()
        by_query = {
            "A One": [_search_result("A One [FLAC]")],
            "B Two": RuntimeError("boom"),
        }
        with patch("bot.tools.QBittorrentClient", side_effect=_search_client(by_query)):
            out = await search_for_torrent.coroutine(
                queries=["A One", "B Two"],
                category=BTCategory.Music,
                runtime=_runtime(ctx),
            )
        assert "Results for 'A One':" in out
        assert "Search for 'B Two' failed: boom" in out
        assert "A One [FLAC]" in ctx.search_results

    def test_bare_string_becomes_list(self):
        args = search_for_torrent.args_schema(queries="X", category=BTCategory.Music)
        assert args.queries == ["X"]


class TestLastFMOwnership:
    @pytest.mark.asyncio
    async def test_owned_album_is_marked(self):
        lastfm = AsyncMock()
        lastfm.__aenter__ = AsyncMock(return_value=lastfm)
        lastfm.__aexit__ = AsyncMock(return_value=False)
        lastfm.get_artist_albums = AsyncMock(
            return_value=[
                LastFMAlbum(name="Fly or Die", artist="Jaimie Branch"),
                LastFMAlbum(name="Iron Lung", artist="Jaimie Branch"),
            ]
        )
        plex = AsyncMock()
        plex.__aenter__ = AsyncMock(return_value=plex)
        plex.__aexit__ = AsyncMock(return_value=False)
        plex.get_all_library_items = AsyncMock(
            return_value={"MediaContainer": {"Metadata": [{"title": "Fly or Die"}]}}
        )
        with patch("bot.tools.LastFMClient", return_value=lastfm), patch(
            "bot.tools.PlexAPIClient", return_value=plex
        ):
            out = await lastfm_artist_albums.ainvoke({"artists": ["Jaimie Branch"]})
        lines = out.split("\n")
        assert "- Fly or Die [OWNED]" in lines
        assert "- Iron Lung" in lines
        assert "1 of these are not in the library." in out

    @pytest.mark.asyncio
    async def test_plex_failure_still_lists(self):
        lastfm = AsyncMock()
        lastfm.__aenter__ = AsyncMock(return_value=lastfm)
        lastfm.__aexit__ = AsyncMock(return_value=False)
        lastfm.get_artist_albums = AsyncMock(
            return_value=[LastFMAlbum(name="X", artist="Y")]
        )
        with patch("bot.tools.LastFMClient", return_value=lastfm), patch(
            "bot.tools.PlexAPIClient", side_effect=RuntimeError("no plex")
        ):
            out = await lastfm_artist_albums.ainvoke({"artists": "Y"})
        assert "- X [library check failed]" in out


class TestCheckAlbums:
    @pytest.mark.asyncio
    async def test_two_albums_two_sections(self):
        async def fake(artist, title):
            if title == "Bad":
                raise RuntimeError("nope")
            return f"ok {title}"

        with patch("bot.tools._check_album", side_effect=fake):
            out = await check_albums.ainvoke(
                {
                    "albums": [
                        {"artist": "A", "title": "Good"},
                        {"artist": "B", "title": "Bad"},
                    ]
                }
            )
        assert "### A - Good\nok Good" in out
        assert "### B - Bad\nCOULD NOT CHECK: nope" in out
