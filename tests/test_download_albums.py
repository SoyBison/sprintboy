from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot.netcode import BTCategory, SearchResult, TorrentInfoResponse
from bot.tools import TorrentContext, download_albums, search_for_torrent

WEB24 = "Khruangbin - Texas Sun [2020] [EP] FLAC / 24bit Lossless / WEB / 2020 [Orpheus]"
ALB24 = "Khruangbin - Con todo el mundo [2018] [Album] FLAC / 24bit Lossless / WEB / 2018 [Orpheus]"
ALB_VINYL = "Khruangbin - Con todo el mundo [2018] [Album] FLAC / 24bit Lossless / Vinyl [Orpheus]"
ONLY_VINYL = "The Murlocs - Young Blindness [2016] [Album] FLAC / 24bit Lossless / Vinyl [Orpheus]"
COLLAB = "Khruangbin and Leon Bridges - Texas Sun [2020] [EP] FLAC / Lossless / WEB [Orpheus]"


def _sr(name):
    return SearchResult(
        fileName=name, fileUrl=f"http://j/{name}", fileSize=1, nbSeeders=1,
        nbLeechers=0, siteUrl="http://t", descrLink="http://t/1",
    )


def _client(names, info=None, code="code"):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.search = AsyncMock(return_value=SimpleNamespace(results=[_sr(n) for n in names]))
    client.add_torrent = AsyncMock(return_value=code)
    client.get_torrent_info = AsyncMock(return_value=info)
    return client


def _ctx():
    return TorrentContext(search_results={}, internal_torrents={}, torrent_types=set())


@pytest.fixture
def info(torrent_data):
    return TorrentInfoResponse(**torrent_data)


async def _run(client, albums, owned="no match", ctx=None, **kw):
    ctx = ctx or _ctx()
    with patch("bot.tools.QBittorrentClient", return_value=client), patch(
        "bot.tools._check_album", AsyncMock(return_value=owned)
    ) as chk:
        out = await download_albums.coroutine(
            albums=albums, runtime=SimpleNamespace(context=ctx), **kw
        )
    return out, ctx, chk


@pytest.mark.asyncio
async def test_prefers_non_vinyl_and_adds(info):
    c = _client([ALB_VINYL, ALB24], info)
    out, ctx, _ = await _run(c, [{"artist": "Khruangbin", "title": "Con todo el mundo"}])
    assert out.startswith("Added")
    assert ctx.internal_torrents == {ALB24: "code"}
    assert BTCategory.Music in ctx.torrent_types
    assert ALB_VINYL in ctx.search_results


@pytest.mark.asyncio
async def test_owned_skips_search(info):
    c = _client([ALB24], info)
    out, _, _ = await _run(
        c, [{"artist": "Khruangbin", "title": "Con todo el mundo"}],
        owned="(Last.fm corrected x) COLLISION: you own it",
    )
    assert out.startswith("OWNED:")
    c.search.assert_not_called()


@pytest.mark.asyncio
async def test_only_vinyl_not_added(info):
    c = _client([ONLY_VINYL], info)
    out, _, _ = await _run(c, [{"artist": "The Murlocs", "title": "Young Blindness"}])
    assert out.startswith("NOT ADDED: only a vinyl rip")
    c.add_torrent.assert_not_called()


@pytest.mark.asyncio
async def test_asking_for_vinyl_adds_vinyl(info):
    c = _client([ONLY_VINYL], info)
    out, _, _ = await _run(
        c, [{"artist": "The Murlocs", "title": "Young Blindness"}], media="Vinyl"
    )
    assert out.startswith("Added")


@pytest.mark.asyncio
async def test_nothing_matching_lists_closest(info):
    c = _client([ALB24], info)
    out, _, _ = await _run(c, [{"artist": "Bonobo", "title": "Migration"}])
    assert out.startswith("NOT FOUND")
    assert "Closest results: " + ALB24 in out
    c.add_torrent.assert_not_called()


@pytest.mark.asyncio
async def test_collaboration_credit_matches(info):
    c = _client([COLLAB], info)
    out, ctx, _ = await _run(c, [{"artist": "Khruangbin", "title": "Texas Sun"}])
    assert out.startswith("Added")
    assert COLLAB in ctx.internal_torrents


@pytest.mark.asyncio
async def test_duplicate_in_one_call(info):
    c = _client([ALB24], info)
    ref = {"artist": "Khruangbin", "title": "Con todo el mundo"}
    out, _, _ = await _run(c, [ref, ref])
    lines = out.split("\n")
    assert sorted(l.split(":")[0].split(" ")[0] for l in lines) == ["Added", "NOT"]
    assert any("duplicate" in l for l in lines)


@pytest.mark.asyncio
async def test_search_for_torrent_music_groups_and_labels():
    c = _client([ALB24, ALB_VINYL, ONLY_VINYL, "weird FLAC thing"])
    ctx = _ctx()
    with patch("bot.tools.QBittorrentClient", return_value=c):
        out = await search_for_torrent.coroutine(
            queries=["q"], category=BTCategory.Music, runtime=SimpleNamespace(context=ctx)
        )
    lines = out.split("\n")
    assert lines.count(ALB24) == 1
    assert ALB_VINYL not in out
    assert ONLY_VINYL + " (vinyl only)" in lines
    assert "weird FLAC thing" in lines
    assert len(ctx.search_results) == 4


def test_title_match_allows_editions_but_not_other_albums():
    from bot.tools import _title_matches

    assert _title_matches("Mordechai", "Mordechai")
    assert _title_matches("MM..FOOD", "MM.. FOOD")
    assert _title_matches("Djesse Vol. 4", "Djesse Vol. 4 (Deluxe Edition)")
    assert _title_matches("OK Computer", "OK Computer (2017 Remaster)")
    assert not _title_matches("Mordechai", "Mordechai Remixes")
    assert not _title_matches("Texas Sun", "Texas Moon")
    assert not _title_matches("The Universe Smiles Upon You", "The Universe Smiles Upon You ii")
