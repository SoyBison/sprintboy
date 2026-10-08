"""Fixtures shared across the test modules.

`tests` is not a package, so test modules cannot import each other: under
`uv run pytest` only the tests directory lands on sys.path, and
`from tests.test_bugs import ...` fails to collect. Shared data belongs here.
"""

import asyncio

import pytest

from bot.netcode import TEST_PLAYLIST_PREFIX, PlexAPIClient

# One complete row from qBittorrent's /torrents/info, which every model change
# in TorrentInfoResponse has to keep parsing.
TORRENT_DATA = {
    "added_on": 0, "amount_left": 0, "auto_tmm": False, "availability": 1.0,
    "category": "Music", "completed": 100, "completion_on": 0,
    "content_path": "/data/Music/Test", "dl_limit": 0, "dlspeed": 0,
    "downloaded": 500, "downloaded_session": 500, "eta": 0,
    "f_l_piece_prio": False, "force_start": False, "hash": "abc",
    "isPrivate": None, "last_activity": 0, "magnet_uri": "",
    "max_ratio": -1.0, "max_seeding_time": -1, "name": "Test Album",
    "num_complete": 1, "num_incomplete": 0, "num_leechs": 0, "num_seeds": 1,
    "priority": 0, "progress": 1.0, "ratio": 1.0, "ratio_limit": -1.0,
    "save_path": "/data/Music/", "seeding_time": 0, "seeding_time_limit": -1,
    "seen_complete": 0, "seq_dl": False, "size": 500, "state": "uploading",
    "super_seeding": False, "total_size": 500, "up_limit": 0,
    "uploaded": 500, "uploaded_session": 500, "url": None,
    "tags": "sprintboy_x", "time_active": 0, "tracker": "", "upspeed": 0,
}


@pytest.fixture
def torrent_data() -> dict:
    """A fresh copy of TORRENT_DATA, safe for a test to mutate."""
    return dict(TORRENT_DATA)


@pytest.fixture
def plex_test_playlist():
    """A playlist title for a test to use, deleted from Plex afterwards.

    The Plex tests run against the real server, so before this every `just test`
    added another playlist to the library: six had piled up unnoticed. The
    teardown sweeps by prefix rather than by the one title, so a run that
    crashed before its own cleanup is tidied up by the next one.
    """
    yield f"{TEST_PLAYLIST_PREFIX}-playlist"

    async def sweep():
        async with PlexAPIClient() as plex:
            await plex.delete_playlists(await plex.find_test_playlists())

    asyncio.run(sweep())
