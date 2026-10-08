"""
Unit tests for the shared client plumbing in netcode.

test_netcode.py exercises these clients against real qBittorrent and Plex
instances. These tests instead mock the transport, so the request building,
status handling and session lifecycle stay covered when no services are up.
"""

import json

import aiohttp
import pytest

from bot.netcode import (
    TEST_PLAYLIST_PREFIX,
    AsyncAPIClient,
    BTCategory,
    PlaylistSummary,
    PlexAPIClient,
    QBittorrentClient,
)


class _FakeResponse:
    """Stands in for an aiohttp response used as an async context manager."""

    def __init__(self, status: int, payload=None, text: str = "", json_error=None):
        self.status = status
        self._payload = payload
        self._text = text
        self._json_error = json_error

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.calls: list[dict] = []
        self.closed = False

    # aiohttp returns a context manager, not a coroutine, so this is not async.
    def request(self, method, url, headers=None, params=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "params": params}
        )
        return self.response

    async def close(self):
        self.closed = True


def _plex_client(response: _FakeResponse) -> tuple[PlexAPIClient, _FakeSession]:
    client = PlexAPIClient.__new__(PlexAPIClient)
    client.base_url = "http://plex.local:32400"
    client.token = "tok"
    client.client_id = "cid"
    client.client_name = "sprintboy"
    client.plex_machine_id = "machine"
    session = _FakeSession(response)
    client.session = session  # type: ignore[assignment]
    return client, session


# ---------------------------------------------------------------------------
# AsyncAPIClient: the session lifecycle all three clients share
# ---------------------------------------------------------------------------


class TestAsyncAPIClient:
    @pytest.mark.asyncio
    async def test_session_is_closed_when_connecting_fails(self):
        """A failed login must not leak the session it was going to use."""
        session = _FakeSession(_FakeResponse(200))

        class Failing(AsyncAPIClient):
            def _new_session(self):  # type: ignore[override]
                return session

            async def _on_connect(self):
                raise RuntimeError("login failed")

        client = Failing()
        with pytest.raises(RuntimeError, match="login failed"):
            await client.__aenter__()

        assert session.closed is True
        assert client.session is None

    @pytest.mark.asyncio
    async def test_exit_closes_and_clears_the_session(self):
        session = _FakeSession(_FakeResponse(200))

        class Plain(AsyncAPIClient):
            def _new_session(self):  # type: ignore[override]
                return session

        client = Plain()
        async with client:
            assert client.session is session
        assert session.closed is True
        assert client.session is None

    def test_live_session_rejects_use_outside_the_context_manager(self):
        client = AsyncAPIClient()
        with pytest.raises(AssertionError, match="Session not initialized"):
            client._live_session


# ---------------------------------------------------------------------------
# PlexAPIClient request plumbing
# ---------------------------------------------------------------------------


class TestPlexHeaders:
    def test_every_request_carries_token_and_identity(self):
        client, _ = _plex_client(_FakeResponse(200))
        assert client._headers == {
            "Accept": "application/json",
            "X-Plex-Product": "sprintboy",
            "X-Plex-Client-Identifier": "cid",
            "X-Plex-Token": "tok",
        }

    def test_missing_token_is_named_in_the_assertion(self):
        client, _ = _plex_client(_FakeResponse(200))
        client.token = None
        with pytest.raises(AssertionError, match="PLEX_TOKEN is not set"):
            client._headers

    def test_missing_client_name_is_named_in_the_assertion(self):
        client, _ = _plex_client(_FakeResponse(200))
        client.client_name = None
        with pytest.raises(AssertionError, match="PLEX_CLIENT_NAME is not set"):
            client._headers


class TestPlexRequest:
    @pytest.mark.asyncio
    async def test_returns_decoded_body_and_sends_headers(self):
        client, session = _plex_client(
            _FakeResponse(200, payload={"MediaContainer": {"size": 0}})
        )
        result = await client._request(
            "GET", "/library/all", "get library items", params={"type": 9}
        )

        assert result == {"MediaContainer": {"size": 0}}
        call = session.calls[0]
        assert call["method"] == "GET"
        assert call["url"] == "http://plex.local:32400/library/all"
        assert call["params"] == {"type": 9}
        assert call["headers"]["X-Plex-Token"] == "tok"

    @pytest.mark.asyncio
    async def test_unexpected_status_raises_with_the_description_and_body(self):
        client, _ = _plex_client(_FakeResponse(500, text="boom"))
        with pytest.raises(Exception, match=r"Failed to get libraries: 500, boom"):
            await client._request("GET", "/library/sections", "get libraries")

    @pytest.mark.asyncio
    async def test_empty_body_becomes_an_empty_dict(self):
        """Scan triggers and deletes answer with no JSON at all."""
        client, _ = _plex_client(
            _FakeResponse(200, json_error=json.JSONDecodeError("nope", "", 0))
        )
        assert await client._request("POST", "/refresh", "initiate media scan") == {}

    @pytest.mark.asyncio
    async def test_non_json_content_type_becomes_an_empty_dict(self):
        client, _ = _plex_client(
            _FakeResponse(200, json_error=aiohttp.ContentTypeError(None, ()))  # type: ignore[arg-type]
        )
        assert await client._request("POST", "/refresh", "initiate media scan") == {}

    @pytest.mark.asyncio
    async def test_delete_playlist_expects_204_not_200(self):
        client, session = _plex_client(_FakeResponse(204))
        await client.delete_playlist(7)
        assert session.calls[0]["method"] == "DELETE"
        assert session.calls[0]["url"].endswith("/playlists/7")

        client, _ = _plex_client(_FakeResponse(200, text="unexpected"))
        with pytest.raises(Exception, match="Failed to delete playlist: 200"):
            await client.delete_playlist(7)


def _playlists_response(*playlists: dict) -> _FakeResponse:
    return _FakeResponse(200, payload={"MediaContainer": {"Metadata": list(playlists)}})


def _raw_playlist(title: str, playlist_id: str, items: int = 0) -> dict:
    return {
        "ratingKey": playlist_id,
        "key": f"/playlists/{playlist_id}/items",
        "title": title,
        "leafCount": items,
        "playlistType": "audio",
    }


class TestFindTestPlaylists:
    """The sweep has to find every playlist the tests made and nothing else."""

    @pytest.mark.asyncio
    async def test_prefixed_playlists_are_found_whatever_their_size(self):
        client, _ = _plex_client(
            _playlists_response(
                _raw_playlist(f"{TEST_PLAYLIST_PREFIX}-playlist", "1"),
                _raw_playlist(f"{TEST_PLAYLIST_PREFIX}-playlist", "2", items=3),
                _raw_playlist("Summervibes 2026 A", "3", items=10),
            )
        )
        found = await client.find_test_playlists()
        assert [p.id for p in found] == ["1", "2"]

    @pytest.mark.asyncio
    async def test_empty_legacy_playlists_are_swept_but_used_ones_are_not(self):
        """The old test named its playlist after the track, so the name alone is
        not proof: only the empty ones are safe to delete."""
        client, _ = _plex_client(
            _playlists_response(
                _raw_playlist("Rapp Snitch Knishes", "1"),
                _raw_playlist("Rapp Snitch Knishes", "2", items=12),
            )
        )
        found = await client.find_test_playlists()
        assert [p.id for p in found] == ["1"]

        client, _ = _plex_client(_playlists_response(_raw_playlist("Rapp Snitch Knishes", "1")))
        assert await client.find_test_playlists(include_legacy=False) == []

    @pytest.mark.asyncio
    async def test_id_falls_back_to_the_key_and_junk_is_skipped(self):
        client, _ = _plex_client(
            _playlists_response(
                {"key": "/playlists/9/items", "title": f"{TEST_PLAYLIST_PREFIX}-a"},
                {"title": f"{TEST_PLAYLIST_PREFIX}-no-id"},
                {"ratingKey": "10"},
            )
        )
        found = await client.find_test_playlists()
        assert [(p.id, p.title) for p in found] == [("9", f"{TEST_PLAYLIST_PREFIX}-a")]

    @pytest.mark.asyncio
    async def test_delete_playlists_counts_successes_and_survives_failures(self):
        client, session = _plex_client(_FakeResponse(204))
        deleted = await client.delete_playlists(
            [PlaylistSummary(id="1", title="a"), PlaylistSummary(id="2", title="b")]
        )
        assert deleted == 2
        assert [call["url"].rsplit("/", 1)[-1] for call in session.calls] == ["1", "2"]

        # A playlist deleted by hand in between must not abort the sweep.
        client, session = _plex_client(_FakeResponse(404))
        assert await client.delete_playlists([PlaylistSummary(id="1", title="a")]) == 0


class TestPlexUnwrapping:
    @pytest.mark.asyncio
    async def test_get_libraries_unwraps_the_directory_list(self):
        client, _ = _plex_client(
            _FakeResponse(
                200,
                payload={
                    "MediaContainer": {
                        "Directory": [{"title": "Music", "key": "1"}],
                    }
                },
            )
        )
        assert await client.get_libraries() == [{"title": "Music", "key": "1"}]

    @pytest.mark.asyncio
    async def test_library_codes_map_plex_titles_to_categories(self):
        client, _ = _plex_client(
            _FakeResponse(
                200,
                payload={
                    "MediaContainer": {
                        "Directory": [
                            {"title": "Music", "key": "1"},
                            {"title": "Movies", "key": "2"},
                            {"title": "TV Shows", "key": "3"},
                            {"title": "Home Videos", "key": "4"},
                        ]
                    }
                },
            )
        )
        codes = await client.get_library_codes()

        assert codes == {
            BTCategory.Music: "1",
            BTCategory.Movies: "2",
            BTCategory.TV: "3",
        }

    @pytest.mark.asyncio
    async def test_scan_media_posts_the_folder_to_the_right_section(self):
        client, session = _plex_client(
            _FakeResponse(
                200,
                payload={
                    "MediaContainer": {"Directory": [{"title": "Music", "key": "1"}]}
                },
            )
        )
        await client.scan_media("/data/Music/Album", BTCategory.Music)

        # First call resolves the library codes, second triggers the scan.
        scan = session.calls[-1]
        assert scan["method"] == "POST"
        assert scan["url"].endswith("/library/sections/1/refresh")
        assert scan["params"] == {"folder": "/data/Music/Album"}


# ---------------------------------------------------------------------------
# QBittorrentClient: the add payload shared by the URL and file paths
# ---------------------------------------------------------------------------


class TestAddPayload:
    def _client(self) -> QBittorrentClient:
        client = QBittorrentClient.__new__(QBittorrentClient)
        client.torrent_path = "/data"
        client.dry_run = False
        return client

    def test_payload_tags_the_torrent_with_its_memory_code(self):
        memory_code, payload = self._client()._add_payload(BTCategory.Music)
        assert payload["tags"] == f"sprintboy_{memory_code}"
        assert payload["category"] == "Music"
        assert payload["savepath"] == "/data/Music"

    def test_explicit_save_path_overrides_the_category_default(self):
        _, payload = self._client()._add_payload(
            BTCategory.Music, save_path="/seed/here"
        )
        assert payload["savepath"] == "/seed/here"
        assert payload["category"] == "Music"

    def test_memory_codes_are_unique_per_call(self):
        client = self._client()
        first, _ = client._add_payload(BTCategory.Movies)
        second, _ = client._add_payload(BTCategory.Movies)
        assert first != second

    @pytest.mark.asyncio
    async def test_fails_body_under_http_200_is_still_a_failure(self):
        """qBittorrent reports a rejected add as the body "Fails." with HTTP 200."""
        response = _FakeResponse(200, text="Fails.")
        with pytest.raises(Exception, match="Failed to add magnet:test"):
            await QBittorrentClient._raise_if_add_failed(response, "magnet:test")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_non_200_is_a_failure(self):
        response = _FakeResponse(403, text="Forbidden")
        with pytest.raises(Exception, match="Failed to add magnet:test"):
            await QBittorrentClient._raise_if_add_failed(response, "magnet:test")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_ok_body_passes(self):
        response = _FakeResponse(200, text="Ok.")
        await QBittorrentClient._raise_if_add_failed(response, "magnet:test")  # type: ignore[arg-type]
