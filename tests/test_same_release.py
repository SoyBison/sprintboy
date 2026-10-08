"""Decision-model judgement of ambiguous 'maybe owned' album cases."""

from unittest.mock import AsyncMock, patch

import pytest

from bot.netcode import CanonicalRelease, LastFMAlbum
from bot.tools import check_for_album, lastfm_artist_albums
from test_lastfm import _mock_plex, _plex_response


def _decision(p):
    d = AsyncMock()
    d.noul = lambda qid: p
    return d


async def _check(p):
    plex, _ = _mock_plex(
        {"Metallica": _plex_response(("Metallica Through the Never", "Metallica"))}
    )
    release = CanonicalRelease(
        artist="Metallica",
        artist_candidates=["Metallica"],
        album="Metallica",
        album_candidates=["Metallica"],
    )
    dec = AsyncMock(return_value=None if p is None else _decision(p))
    with patch("bot.tools.PlexAPIClient", return_value=plex), patch(
        "bot.tools.resolve_release", new_callable=AsyncMock, return_value=release
    ), patch("bot.tools.decide", dec):
        out = await check_for_album.ainvoke({"artist": "Metallica", "title": "Metallica"})
    return out, dec


class TestCheckAlbum:
    @pytest.mark.asyncio
    async def test_same(self):
        out, _ = await _check(0.9)
        assert "COLLISION" in out
        assert "judged the same release, p=0.90" in out

    @pytest.mark.asyncio
    async def test_different(self):
        out, _ = await _check(0.1)
        assert "does NOT have" in out
        assert "may be the same release" not in out

    @pytest.mark.asyncio
    async def test_in_between(self):
        out, _ = await _check(0.5)
        assert "may be the same release" in out
        assert "same-release probability 0.50" in out

    @pytest.mark.asyncio
    async def test_no_backend_keeps_old_behaviour(self):
        out, _ = await _check(None)
        assert "may be the same release" in out
        assert "probability" not in out
        assert "COLLISION" not in out

    @pytest.mark.asyncio
    async def test_no_possibles_no_decide(self):
        plex, _ = _mock_plex(
            {"Fleet Foxes": _plex_response(("Helplessness Blues", "Fleet Foxes"))}
        )
        release = CanonicalRelease(
            artist="Fleet Foxes",
            artist_candidates=["Fleet Foxes"],
            album="Shore",
            album_candidates=["Shore"],
        )
        dec = AsyncMock()
        with patch("bot.tools.PlexAPIClient", return_value=plex), patch(
            "bot.tools.resolve_release", new_callable=AsyncMock, return_value=release
        ), patch("bot.tools.decide", dec):
            await check_for_album.ainvoke({"artist": "Fleet Foxes", "title": "Shore"})
        dec.assert_not_called()


async def _listing(p):
    lastfm = AsyncMock()
    lastfm.__aenter__ = AsyncMock(return_value=lastfm)
    lastfm.__aexit__ = AsyncMock(return_value=False)
    lastfm.get_artist_albums = AsyncMock(
        return_value=[
            LastFMAlbum(name="Fly or Die", artist="Jaimie Branch"),
            LastFMAlbum(name="Fly or Die (Deluxe Edition)", artist="Jaimie Branch"),
            LastFMAlbum(name="Iron Lung", artist="Jaimie Branch"),
        ]
    )
    plex = AsyncMock()
    plex.__aenter__ = AsyncMock(return_value=plex)
    plex.__aexit__ = AsyncMock(return_value=False)
    plex.get_all_library_items = AsyncMock(
        return_value={"MediaContainer": {"Metadata": [{"title": "Fly or Die"}]}}
    )
    dec = AsyncMock(return_value=_decision(p))
    with patch("bot.tools.LastFMClient", return_value=lastfm), patch(
        "bot.tools.PlexAPIClient", return_value=plex
    ), patch("bot.tools.decide", dec):
        out = await lastfm_artist_albums.ainvoke({"artists": ["Jaimie Branch"]})
    return out, dec


class TestListing:
    @pytest.mark.asyncio
    async def test_maybe_resolved_owned(self):
        out, dec = await _listing(0.9)
        assert "- Fly or Die (Deluxe Edition) [OWNED]" in out.split("\n")
        assert "1 of these are not in the library." in out
        assert dec.await_count == 1  # exact [OWNED] and unmarked not judged

    @pytest.mark.asyncio
    async def test_maybe_resolved_unowned(self):
        out, _ = await _listing(0.1)
        assert "- Fly or Die (Deluxe Edition)" in out.split("\n")
        assert "2 of these are not in the library." in out
