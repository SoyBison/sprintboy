"""
Unit tests for the silent success this session found: on 2026-09-18 the bot had
been answering "Added: <album>" in Discord for days while qBittorrent's log
showed one WebAPI login per invocation and no add at all since 2026-09-14.

Reproduced with `agent_cli`: qwen3:30b-a3b searched, never called add_torrent,
and finished with "Added: Jaimie Branch - Fly or Die (2017)". Because nothing
was added, the download wait never ran, so nothing in the chat ever
contradicted the claim. The container's own log shows the same three turns:

    6 tool calls [... 'check_for_album', 'search_for_torrent'] and replied:
      'Added: The Naked and Famous - Passive Me, Aggressive You (2010) ...'
    4 tool calls [... 'lastfm_artist_albums', 'check_for_album'] and replied:
      'Added "Alone in IZ World" by Israel Kamakawiwo\'ole to your library.'

-- the second of which never even searched, which is why the note triggers on
the library checks too and not just on search_for_torrent.

Two layers are covered here:
  1. add_torrent only says "Added" once the torrent is really in qBittorrent,
     which closes the other silent path: qBittorrent answers an add with "Ok."
     before it has fetched the URL, and never reports the fetch failing.
  2. unfulfilled_note states the ground truth when a turn searched and added
     nothing, whatever the model claimed in prose.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot.netcode import BTCategory, CanonicalRelease, SearchResult, TorrentInfoResponse
from bot.output import unfulfilled_note
from bot.tools import TorrentContext, add_torrent, check_for_album


class _AI:
    type = "ai"

    def __init__(self, content, calls=()):
        self.content = content
        self.tool_calls = [{"name": name} for name in calls]


# ---------------------------------------------------------------------------
# add_torrent says "Added" only when it can prove it
# ---------------------------------------------------------------------------


def _context(*names: str) -> TorrentContext:
    return TorrentContext(
        search_results={
            name: SearchResult(
                fileName=name,
                fileUrl=f"http://jackett/dl/{name}",
                fileSize=1,
                nbSeeders=1,
                nbLeechers=0,
                siteUrl="http://tracker",
                descrLink="http://tracker/1",
            )
            for name in names
        },
        internal_torrents={},
        torrent_types=set(),
    )


def _mock_qclient(info, memory_code="code", timeout=False):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.add_torrent = AsyncMock(return_value=memory_code)
    if timeout:
        client.get_torrent_info = AsyncMock(
            side_effect=TimeoutError("did not appear in qBittorrent within 20.0s")
        )
    else:
        client.get_torrent_info = AsyncMock(return_value=info)
    return client


async def _add(context, client, name="Test Album [FLAC]"):
    with patch("bot.tools.QBittorrentClient", return_value=client):
        return await add_torrent.coroutine(
            names=[name],
            category=BTCategory.Music,
            runtime=SimpleNamespace(context=context),
        )


class TestAddTorrentConfirmation:
    @pytest.mark.asyncio
    async def test_confirmed_add_is_recorded_and_reported(self, torrent_data):
        context = _context("Test Album [FLAC]")
        info = TorrentInfoResponse(**torrent_data)
        result = await _add(context, _mock_qclient(info))

        assert result.startswith("Added")
        assert context.internal_torrents == {"Test Album [FLAC]": "code"}

    @pytest.mark.asyncio
    async def test_accepted_but_absent_torrent_is_not_a_success(self, torrent_data):
        """qBittorrent answers "Ok." before fetching the URL, so an unreachable
        indexer used to look exactly like a completed add."""
        context = _context("Test Album [FLAC]")
        info = TorrentInfoResponse(**torrent_data)
        result = await _add(context, _mock_qclient(info, timeout=True))

        assert result.startswith("NOT ADDED")
        assert "never appeared" in result
        # Not recorded, so the download wait does not sit on a phantom torrent.
        assert context.internal_torrents == {}

    @pytest.mark.asyncio
    async def test_dry_run_says_nothing_was_downloaded(self, torrent_data):
        context = _context("Test Album [FLAC]")
        info = TorrentInfoResponse(**torrent_data)
        result = await _add(context, _mock_qclient(info, memory_code=None))

        assert result.startswith("NOT ADDED")
        assert "DRY_RUN" in result
        assert context.internal_torrents == {}

    @pytest.mark.asyncio
    async def test_adding_without_searching_first_does_not_reach_qbittorrent(self, torrent_data):
        """extractOne returns None on an empty pool, which used to be a
        TypeError the model was free to answer "added" to anyway."""
        context = _context()
        client = _mock_qclient(TorrentInfoResponse(**torrent_data))
        result = await _add(context, client)

        assert result.startswith("NOT ADDED")
        assert "search_for_torrent" in result
        client.add_torrent.assert_not_awaited()


# ---------------------------------------------------------------------------
# The reply posted to Discord contradicts an invented download
# ---------------------------------------------------------------------------


class TestUnfulfilledNote:
    def test_searching_and_adding_nothing_is_contradicted(self):
        messages = [
            _AI("", ["search_for_torrent"]),
            _AI("Added: Jaimie Branch - Fly or Die (2017)"),
        ]
        note = unfulfilled_note(messages, [])
        assert "Nothing was added to qBittorrent" in note

    def test_claiming_an_add_after_only_a_library_check_is_contradicted(self):
        """The deployed turn that got no further than check_for_album and still
        replied 'Added "Alone in IZ World" ... to your library'."""
        messages = [
            _AI("", ["lastfm_artist_albums"]),
            _AI("", ["check_for_album"]),
            _AI('Added "Alone in IZ World" by Israel Kamakawiwo\'ole to your library.'),
        ]
        assert "Nothing was added to qBittorrent" in unfulfilled_note(messages, [])

    def test_a_failed_add_is_named_as_such(self):
        messages = [
            _AI("", ["search_for_torrent"]),
            _AI("", ["add_torrent"]),
            _AI("Added Fly or Die."),
        ]
        assert "every add failed" in unfulfilled_note(messages, [])

    def test_a_failed_download_albums_is_named_as_such(self):
        messages = [_AI("", ["download_albums"]), _AI("Added it.")]
        assert "every add failed" in unfulfilled_note(messages, [])

    def test_a_real_add_is_left_alone(self):
        messages = [_AI("", ["search_for_torrent", "add_torrent"]), _AI("Added it.")]
        assert unfulfilled_note(messages, ["Fly or Die [FLAC]"]) == ""

    def test_a_conversation_is_left_alone(self):
        """Asking what an artist sounds like adds nothing, and that is fine."""
        messages = [_AI("", ["lastfm_artist_info"]), _AI("She plays free jazz.")]
        assert unfulfilled_note(messages, []) == ""


# ---------------------------------------------------------------------------
# A Plex outage is not an empty library
# ---------------------------------------------------------------------------


class TestPlexLookupFailure:
    @pytest.mark.asyncio
    async def test_failed_lookups_are_not_reported_as_not_owning_it(self):
        """Plex answered 401 during the repro and the tool said the user had no
        albums by the artist, which is how the same album gets downloaded twice."""
        plex = AsyncMock()
        plex.__aenter__ = AsyncMock(return_value=plex)
        plex.__aexit__ = AsyncMock(return_value=False)
        plex.get_all_library_items = AsyncMock(
            side_effect=Exception("Failed to get library items: 401, Unauthorized")
        )
        release = CanonicalRelease(
            artist="Jaimie Branch",
            artist_candidates=["Jaimie Branch"],
            album="Fly or Die",
            album_candidates=["Fly or Die"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "Jaimie Branch", "title": "Fly or Die"}
                )

        assert "COULD NOT CHECK" in result
        assert "no albums" not in result


class TestAddNudge:
    def test_searched_but_never_added_is_nudged(self):
        from bot.output import needs_add_nudge

        assert needs_add_nudge([_AI("", ["search_for_torrent"]), _AI("Added X")], [])

    def test_a_failed_add_is_not_nudged(self):
        """Dry run or a dead indexer: retrying the add would just fail again."""
        from bot.output import needs_add_nudge

        messages = [_AI("", ["search_for_torrent"]), _AI("", ["add_torrent"])]
        assert not needs_add_nudge(messages, [])

    def test_download_albums_counts_as_an_add(self):
        from bot.output import needs_add_nudge

        messages = [_AI("", ["search_for_torrent", "download_albums"])]
        assert not needs_add_nudge(messages, [])

    def test_a_real_add_or_a_chat_is_not_nudged(self):
        from bot.output import needs_add_nudge

        assert not needs_add_nudge([_AI("", ["search_for_torrent"])], ["X [FLAC]"])
        assert not needs_add_nudge([_AI("", ["lastfm_artist_info"])], [])


class TestRouteAwareNudge:
    def _route(self, domain, kind, p=1.0):
        from bot.routing import Route

        return Route(domain, p, kind, p, None)

    def test_music_download_that_never_tried_to_add_is_nudged(self):
        from bot.output import MUSIC_NUDGE, add_nudge

        messages = [_AI("", ["lastfm_artist_albums"]), _AI("Added Texas Sun.")]
        assert add_nudge(messages, [], self._route("music", "discography")) == MUSIC_NUDGE

    def test_movie_request_points_at_add_torrent(self):
        from bot.output import VIDEO_NUDGE, add_nudge

        messages = [_AI("", ["check_for_movie"]), _AI("Added Dune.")]
        assert add_nudge(messages, [], self._route("movie", "specific")) == VIDEO_NUDGE

    def test_questions_and_attempted_adds_are_left_alone(self):
        from bot.output import add_nudge

        assert add_nudge([_AI("", ["check_for_album"])], [], self._route("music", "question")) is None
        assert add_nudge([_AI("", ["download_albums"])], [], self._route("music", "open_ended")) is None
        assert add_nudge([_AI("", [])], ["X [FLAC]"], self._route("music", "open_ended")) is None

    def test_untrusted_route_falls_back_to_the_search_rule(self):
        from bot.output import ADD_NUDGE, add_nudge

        shaky = self._route("music", "open_ended", p=0.4)
        assert "download_albums" in add_nudge([_AI("", ["lastfm_browse_tag"])], [], shaky)
        assert add_nudge([_AI("", ["search_for_torrent"])], [], shaky) == ADD_NUDGE


class _Tool:
    type = "tool"
    tool_calls = []

    def __init__(self, content):
        self.content = content


class TestNudgeEdgeCases:
    def test_specific_request_they_already_own_is_not_nudged(self):
        from bot.output import add_nudge
        from bot.routing import Route

        messages = [
            _AI("", ["check_for_album"]),
            _Tool("COLLISION: the user already owns 'Kind of Blue' by Miles Davis."),
            _AI("You already own it."),
        ]
        assert add_nudge(messages, [], Route("music", 1.0, "specific", 1.0, None)) is None

    def test_no_route_recommendation_turn_is_nudged(self):
        from bot.output import add_nudge

        messages = [_AI("", ["lastfm_browse_tag"]), _AI("Added: Distressor by Whirr")]
        assert "download_albums" in add_nudge(messages, [], None)

    def test_no_route_artist_question_is_not_nudged(self):
        from bot.output import add_nudge

        assert add_nudge([_AI("", ["lastfm_artist_info"]), _AI("Desert funk.")], [], None) is None
