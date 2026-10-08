"""
Unit tests for the MusicBrainz integration.

All mocked: conftest takes MusicBrainzClient._get offline for every test, and
the tests here that exercise _get itself put the real one back over a fake
aiohttp session.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from bot import netcode
from bot.netcode import (
    MBArtist,
    MBReleaseGroup,
    MusicBrainzClient,
    MusicBrainzError,
    _Throttle,
    _mb_year,
    _parse_mb_artist,
    _parse_mb_release_group,
    artist_catalogue,
)
from bot.tools import _musicbrainz_catalogue, musicbrainz_discography

# Captured at import, before the autouse fixture swaps it out.
REAL_GET = MusicBrainzClient._get

KHRUANGBIN = "aea4c9b9-9f8d-49dc-b2ca-57d6f26e8634"


def _raw_group(title, date="2020-06-26", primary="Album", secondary=(), mbid=None):
    return {
        "id": mbid or f"id-{title}",
        "title": title,
        "primary-type": primary,
        "secondary-types": list(secondary),
        "first-release-date": date,
    }


def _client() -> MusicBrainzClient:
    client = MusicBrainzClient.__new__(MusicBrainzClient)
    client.base_url = "https://musicbrainz.test/ws/2/"
    client.user_agent = "sprintboy/test ( test )"
    client.session = None
    return client


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize(
        "date, year", [("2020-06-26", 2020), ("2017-03", 2017), ("2012", 2012), ("", None), (None, None)]
    )
    def test_year_from_partial_dates(self, date, year):
        assert _mb_year(date) == year

    def test_studio_album(self):
        group = _parse_mb_release_group(_raw_group("Mordechai"))
        assert group.studio and group.kind == "Album" and group.year == 2020

    @pytest.mark.parametrize(
        "primary, secondary, kind",
        [
            ("Album", ["Live"], "Live Album"),
            ("Album", ["Compilation", "DJ-mix"], "Compilation DJ-mix Album"),
            ("EP", [], "EP"),
        ],
    )
    def test_not_studio(self, primary, secondary, kind):
        group = _parse_mb_release_group(_raw_group("x", primary=primary, secondary=secondary))
        assert not group.studio and group.kind == kind

    def test_missing_type_is_not_studio(self):
        raw = _raw_group("x")
        raw["primary-type"] = None
        group = _parse_mb_release_group(raw)
        assert not group.studio and group.kind == "Release"

    def test_untitled_group_is_dropped(self):
        assert _parse_mb_release_group({"id": "x", "title": "  "}) is None
        assert _parse_mb_release_group({"title": "no id"}) is None

    def test_artist_with_aliases(self):
        artist = _parse_mb_artist(
            {
                "id": KHRUANGBIN, "name": "Khruangbin", "score": 100, "disambiguation": "",
                "aliases": [{"name": "Kruangbin"}, {"name": "kruangbin"}, "junk"],
            }
        )
        assert artist == MBArtist(mbid=KHRUANGBIN, name="Khruangbin", score=100, aliases=["Kruangbin"])


# ---------------------------------------------------------------------------
# Requests: throttling, retries, errors
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body if body is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self.body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


@pytest.fixture
def real_get(monkeypatch):
    monkeypatch.setattr(MusicBrainzClient, "_get", REAL_GET)
    monkeypatch.setattr(netcode, "_MB_THROTTLE", _Throttle(0))
    monkeypatch.setattr(netcode, "MUSICBRAINZ_INTERVAL", 0)


@pytest.mark.asyncio
class TestRequests:
    async def test_sends_json_format_and_params(self, real_get):
        client = _client()
        client.session = FakeSession(FakeResponse(body={"artists": []}))
        assert await client._get("artist", query="q", limit=None) == {"artists": []}
        url, params = client.session.calls[0]
        assert url == "https://musicbrainz.test/ws/2/artist"
        assert params == {"fmt": "json", "query": "q"}

    async def test_retries_rate_limit_then_succeeds(self, real_get):
        client = _client()
        client.session = FakeSession(FakeResponse(503), FakeResponse(503), FakeResponse(body={"ok": 1}))
        assert await client._get("artist") == {"ok": 1}
        assert len(client.session.calls) == 3

    async def test_gives_up_after_retries(self, real_get):
        client = _client()
        client.session = FakeSession(*[FakeResponse(503) for _ in range(3)])
        with pytest.raises(MusicBrainzError, match="HTTP 503"):
            await client._get("artist")

    @pytest.mark.parametrize("status, message", [(404, "nothing at"), (400, "HTTP 400")])
    async def test_errors(self, real_get, status, message):
        client = _client()
        client.session = FakeSession(FakeResponse(status))
        with pytest.raises(MusicBrainzError, match=message):
            await client._get("artist/x")

    async def test_non_object_body(self, real_get):
        client = _client()
        client.session = FakeSession(FakeResponse(body=["no"]))
        with pytest.raises(MusicBrainzError, match="unexpected payload"):
            await client._get("artist")



def test_user_agent_names_a_contact(monkeypatch):
    monkeypatch.setenv("MUSICBRAINZ_CONTACT", "me@example.com")
    assert MusicBrainzClient().user_agent == "sprintboy/1.0 ( me@example.com )"


@pytest.mark.asyncio
async def test_throttle_spaces_requests():
    throttle = _Throttle(0.05)
    loop = asyncio.get_running_loop()
    times = []

    async def one():
        async with throttle:
            times.append(loop.time())

    await asyncio.gather(one(), one(), one())
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(gap >= 0.045 for gap in gaps)


def test_throttle_survives_a_new_event_loop():
    throttle = _Throttle(0)

    async def one():
        async with throttle:
            pass

    asyncio.run(one())
    asyncio.run(one())


# ---------------------------------------------------------------------------
# Artist matching and catalogues
# ---------------------------------------------------------------------------


def _artist(name, score=100, aliases=(), mbid=KHRUANGBIN):
    return MBArtist(mbid=mbid, name=name, score=score, aliases=list(aliases))


@pytest.mark.asyncio
class TestFindArtist:
    async def test_uses_lastfm_mbid_first(self):
        client = _client()
        client.get_artist = AsyncMock(return_value=_artist("Khruangbin"))
        client.search_artists = AsyncMock()
        assert (await client.find_artist("khruangbin", KHRUANGBIN)).name == "Khruangbin"
        client.search_artists.assert_not_called()

    async def test_falls_back_to_search_when_mbid_unknown(self):
        client = _client()
        client.get_artist = AsyncMock(side_effect=MusicBrainzError("404"))
        client.search_artists = AsyncMock(return_value=[_artist("Khruangbin")])
        assert (await client.find_artist("Khruangbin", "stale")).name == "Khruangbin"

    async def test_matches_an_alias(self):
        client = _client()
        client.search_artists = AsyncMock(return_value=[_artist("Khruangbin", aliases=["Kruangbin"])])
        assert await client.find_artist("Kruangbin") is not None

    async def test_matches_typographic_apostrophe(self):
        client = _client()
        client.search_artists = AsyncMock(return_value=[_artist("Guns N’ Roses")])
        assert await client.find_artist("Guns N' Roses") is not None

    async def test_rejects_low_scores_and_other_names(self):
        client = _client()
        client.search_artists = AsyncMock(
            return_value=[_artist("Khruangbin & Leon Bridges", 99), _artist("Khruangbin", 60)]
        )
        assert await client.find_artist("Khruangbin") is None

    async def test_search_quotes_the_name(self):
        client = _client()
        client._get = AsyncMock(return_value={"artists": []})
        await client.search_artists('Boards "of" Canada')
        query = client._get.await_args.kwargs["query"]
        assert query.count('"') == 4 and "alias:" in query


@pytest.mark.asyncio
async def test_release_groups_page_and_sort():
    client = _client()
    first = [_raw_group(f"T{i}", date=str(2000 + i % 30)) for i in range(100)]
    second = [_raw_group("Undated", date=""), _raw_group("Early", date="1990")]
    client._get = AsyncMock(
        side_effect=[
            {"release-groups": first, "release-group-count": 102},
            {"release-groups": second, "release-group-count": 102},
        ]
    )
    groups = await client.get_release_groups(KHRUANGBIN)
    assert len(groups) == 102
    assert groups[0].title == "Early" and groups[-1].title == "Undated"
    params = client._get.await_args_list[1].kwargs
    assert params["offset"] == 100 and params["type"] == "album|ep"
    assert params["release-group-status"] == "website-default"


@pytest.mark.asyncio
async def test_catalogue_none_when_artist_unknown():
    with patch.object(MusicBrainzClient, "find_artist", AsyncMock(return_value=None)):
        assert await artist_catalogue("nobody") is None


@pytest.mark.asyncio
async def test_best_effort_catalogue_swallows_outages():
    # conftest has MusicBrainz offline.
    assert await _musicbrainz_catalogue("Khruangbin") is None


@pytest.mark.asyncio
async def test_best_effort_catalogue_times_out(monkeypatch):
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr("bot.tools.artist_catalogue", slow)
    monkeypatch.setattr("bot.tools.MUSICBRAINZ_TIMEOUT", 0.01)
    assert await _musicbrainz_catalogue("Khruangbin") is None


# ---------------------------------------------------------------------------
# The agent tool
# ---------------------------------------------------------------------------


GROUPS = [
    MBReleaseGroup(mbid="1", title="Con todo el mundo", primary_type="Album", year=2018),
    MBReleaseGroup(mbid="2", title="Mordechai", primary_type="Album", year=2020),
    MBReleaseGroup(mbid="3", title="Live at Stubb's", primary_type="Album", secondary_types=["Live"], year=2023),
]


def _tool_patches(owned=("Mordechai",)):
    return (
        patch.object(MusicBrainzClient, "find_artist", AsyncMock(return_value=_artist("Khruangbin"))),
        patch.object(MusicBrainzClient, "get_release_groups", AsyncMock(return_value=GROUPS)),
        patch("bot.tools._library_lookup", AsyncMock(return_value={"Khruangbin": list(owned)})),
    )


@pytest.mark.asyncio
async def test_tool_lists_studio_albums_with_ownership():
    a, b, c = _tool_patches()
    with a, b, c:
        out = await musicbrainz_discography.ainvoke({"artists": "Khruangbin"})
    assert "Studio albums by Khruangbin on MusicBrainz, oldest first:" in out
    assert "- Con todo el mundo (2018)\n- Mordechai (2020) [OWNED]" in out
    assert "Stubb" not in out
    assert "1 of these are not in the library." in out


@pytest.mark.asyncio
async def test_tool_include_other_labels_types():
    a, b, c = _tool_patches(owned=())
    with a, b, c:
        out = await musicbrainz_discography.ainvoke({"artists": ["Khruangbin"], "include_other": True})
    assert "- Live at Stubb's (2023) [Live Album]" in out
    assert "- Mordechai (2020) [Album]" in out


@pytest.mark.asyncio
async def test_tool_reports_failures_per_artist():
    out = await musicbrainz_discography.ainvoke({"artists": ["Khruangbin"]})
    assert out.startswith("MusicBrainz lookup for Khruangbin failed:")
