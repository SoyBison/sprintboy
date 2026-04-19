import pytest
import json
import asyncio

import logging
from bot.netcode import (
    QBittorrentClient,
    PlexAPIClient,
    SearchResultsResponse,
    TorrentInfoResponses,
    fetch_url,
    BTCategory,
)

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)


class TestQBittorrentClient:
    """Test QBittorrent client against actual service."""

    @pytest.mark.asyncio
    async def test_client_login_and_context_manager(self):
        """Test that client can login and context manager works."""
        async with QBittorrentClient() as client:
            # If we get here without exception, login worked
            assert client.session is not None
            assert client.cookie is None or isinstance(client.cookie, str)
            print(f"✓ Successfully logged in to qBittorrent at {client.base_url}")

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search functionality with a common query."""
        async with QBittorrentClient() as client:
            # Search for something likely to have results
            results = await client.search("ubuntu")

            assert isinstance(results, SearchResultsResponse)
            assert hasattr(results, "results")
            assert isinstance(results.results, list)

            print(f"✓ Search returned {len(results.results)} results")

            if len(results.results) > 0:
                first_result = results.results[0]
                assert hasattr(first_result, "fileName")
                assert hasattr(first_result, "fileSize")
                assert hasattr(first_result, "nbSeeders")
                assert hasattr(first_result, "nbLeechers")
                print(f"  First result: {first_result.fileName}")
                print(f"  Size: {first_result.fileSize} bytes")
                print(
                    f"  Seeders: {first_result.nbSeeders}, Leechers: {first_result.nbLeechers}"
                )

    @pytest.mark.asyncio
    async def test_search_music(self):
        """Test search with music-related query."""
        async with QBittorrentClient() as client:
            # Search for free/legal music content
            results = await client.search("MF DOOM")

            assert isinstance(results, SearchResultsResponse)
            print(f"✓ Music search returned {len(results.results)} results")

            # Just verify the structure is correct
            for result in results.results[:3]:  # Check first 3 results
                assert result.fileName
                assert result.fileUrl
                assert result.fileSize >= 0
                assert result.nbSeeders >= 0
                assert result.nbLeechers >= 0

    @pytest.mark.asyncio
    async def test_add_torrent_appears_in_qbittorrent(self):
        """
        Integration test: adds a real torrent, then verifies it appears in
        qBittorrent when queried by its sprintboy tag.

        This test catches two failure modes:
          1. add_torrent silently fails (torrent never added)
          2. get_torrent_info tag query returns empty (lookup broken)

        The test uses a magnet URI with a known-invalid hash so nothing is
        actually downloaded, but the torrent entry IS created in qBittorrent.
        """
        test_magnet = (
            "magnet:?xt=urn:btih:0000000000000000000000000000000000000001"
            "&dn=sprintboy-integration-test"
        )
        async with QBittorrentClient() as client:
            # Step 1: add the torrent and get back the memory_code tag
            memory_code = await client.add_torrent(test_magnet, BTCategory.Music)
            assert memory_code is not None, "add_torrent returned None (dry_run mode?)"

            # Step 2: give qBittorrent a moment to register the entry
            await asyncio.sleep(1)

            # Step 3: query directly (bypassing the retry loop) to see raw result
            assert client.session is not None
            info_url = f"{client.base_url}/torrents/info"
            params = {"tag": f"sprintboy_{memory_code}"}
            result = await fetch_url(client.session, info_url, TorrentInfoResponses, params=params)

            assert len(result.root) > 0, (
                f"Torrent with tag sprintboy_{memory_code} not found in qBittorrent. "
                "Either add_torrent failed silently or tag-based filtering is broken."
            )
            assert f"sprintboy_{memory_code}" in result.root[0].tags

            # Cleanup: delete the test torrent
            delete_url = f"{client.base_url}/torrents/delete"
            torrent_hash = result.root[0].hash
            async with client.session.post(
                delete_url, data={"hashes": torrent_hash, "deleteFiles": "false"}
            ) as resp:
                assert resp.status == 200, f"Cleanup failed: {resp.status}"

    @pytest.mark.asyncio
    async def test_add_torrent_via_jackett_url_appears_in_qbittorrent(self):
        """
        Full pipeline test matching the real bot flow:
          search → pick a result URL → add_torrent → verify tag lookup.

        Unlike test_add_torrent_appears_in_qbittorrent which uses a magnet link,
        this uses a real Jackett HTTP URL, which requires qBittorrent to fetch the
        .torrent file from Jackett. If qBittorrent can't reach Jackett, the torrent
        entry will not appear and this test will fail.

        Also prints the torrent state so we can see if it's downloading/stalled/error.
        """
        async with QBittorrentClient() as client:
            # Search for something that definitely has results
            results = await client.search("Herbie Hancock Head Hunters")
            assert results.results, "Search returned no results — is qBittorrent search working?"

            # Pick the first FLAC result with a fileUrl and use its real Jackett URL
            flac_results = [r for r in results.results if "FLAC" in r.fileName]
            assert flac_results, "No FLAC results found"
            first = flac_results[0]
            print(f"\n  Torrent name: {first.fileName}")
            print(f"  URL (first 100 chars): {first.fileUrl[:100]}")

            assert client.session is not None
            memory_code = await client.add_torrent(first.fileUrl, BTCategory.Music)
            assert memory_code is not None

            # Poll for up to 10 seconds — qBittorrent needs time to fetch from Jackett
            info_url = f"{client.base_url}/torrents/info"
            params = {"tag": f"sprintboy_{memory_code}"}
            result = None
            for _ in range(10):
                await asyncio.sleep(1)
                result = await fetch_url(client.session, info_url, TorrentInfoResponses, params=params)
                if result.root:
                    break

            assert result is not None and len(result.root) > 0, (
                "Torrent added via Jackett URL did not appear in qBittorrent after 10s.\n"
                "qBittorrent cannot reach the Jackett URL — the bot is silently failing to add torrents."
            )
            info = result.root[0]
            print(f"  Torrent state: {info.state!r}, progress: {info.progress:.0%}")

            # Cleanup
            delete_url = f"{client.base_url}/torrents/delete"
            async with client.session.post(
                delete_url, data={"hashes": info.hash, "deleteFiles": "false"}
            ) as resp:
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_add_torrent_dry_run(self):
        """Test add torrent in dry run mode (if configured)."""
        async with QBittorrentClient() as client:
            if client.dry_run:
                # This should not actually download anything
                test_url = "magnet:?xt=urn:btih:test"
                await client.add_torrent(test_url, BTCategory.Music)
                print("✓ Dry run add_torrent completed without error")
            else:
                print("⚠ Skipping add_torrent test (dry_run not enabled)")


class TestPlexAPIClient:
    """Test Plex API client against actual service."""

    @pytest.mark.asyncio
    async def test_client_initialization_and_context_manager(self):
        """Test that Plex client initializes correctly."""
        async with PlexAPIClient() as client:
            assert client.base_url is not None
            assert client.token is not None
            assert client.client_id is not None
            assert client.client_name is not None
            assert client.session is not None
            print("✓ Plex client initialized successfully")
            print(f"  Server: {client.base_url}")
            print(f"  Client: {client.client_name}")

    @pytest.mark.asyncio
    async def test_library_search(self):
        """Test library search with a generic query."""
        async with PlexAPIClient() as client:
            # Search for something generic
            results = await client.get_library_matches(
                title="Tenet", metadata_type="movie"
            )

            assert isinstance(results, dict)
            print("✓ Library search returned valid response")

            # Check if MediaContainer exists (standard Plex response structure)
            assert "MediaContainer" in results
            logging.debug(results)
            assert len(results["MediaContainer"]["Metadata"]) > 0
            if "MediaContainer" in results:
                container = results["MediaContainer"]
                if "size" in container:
                    print(f"  Found {container['size']} matches")
                if "Metadata" in container:
                    print(f"  Metadata entries: {len(container['Metadata'])}")
                    # Show first few results if any
                    for item in container["Metadata"][:3]:
                        if "title" in item:
                            print(f"    - {item['title']}")

    @pytest.mark.asyncio
    async def test_library_dump(self):
        async with PlexAPIClient() as client:
            # Dump all library items
            items = await client.get_all_library_items({"type": 1})

            assert isinstance(items, dict)
            print("✓ Library dump returned valid response")

            logging.debug(items)

            # Check if MediaContainer exists (standard Plex response structure)
            if "MediaContainer" in items:
                container = items["MediaContainer"]
                if "size" in container:
                    print(f"  Found {container['size']} items")
                if "Metadata" in container:
                    print(f"  Metadata entries: {len(container['Metadata'])}")
                    # Show first few results if any
                    for item in container["Metadata"][:3]:
                        if "title" in item:
                            print(f"    - {item['title']}")

    @pytest.mark.asyncio
    async def test_get_shore_fleet_foxes_by_get_all_library_items(self):
        async with PlexAPIClient() as client:
            results = await client.get_all_library_items(
                {
                    "type": 9,
                    "artist.title": "Fleet Foxes",
                    "title": "Shore",
                    "year": 2020,
                }
            )

            assert isinstance(results, dict)
            song_id = results["MediaContainer"]["Metadata"][0]["key"]
            assert song_id
            print("✓ Library search returned valid response, shore id: ", song_id)

            # Check if MediaContainer exists (standard Plex response structure)
            assert "MediaContainer" in results
            logging.debug(results)
            assert len(results["MediaContainer"]["Metadata"]) > 0

    @pytest.mark.asyncio
    async def test_get_specific_song(self):
        async with PlexAPIClient() as client:
            results = await client.get_all_library_items(
                {
                    "type": 10,
                    "artist.title": "MF DOOM",
                    "album.title": "MM..FOOD",
                    "title": "Rapp Snitch Knishes",
                }
            )

            assert isinstance(results, dict)
            print(results)
            song_id = results["MediaContainer"]["Metadata"][0]["key"]
            assert song_id
            print(
                "✓ Library search returned valid response, rapp snitch knishes id: ",
                song_id,
            )

            # Check if MediaContainer exists (standard Plex response structure)
            assert "MediaContainer" in results
            logging.debug(results)
            assert len(results["MediaContainer"]["Metadata"]) > 0

    @pytest.mark.asyncio
    async def test_make_playlist(self):
        async with PlexAPIClient() as client:
            # first get rapp snitch knishes by mf doom
            results = await client.get_all_library_items(
                {
                    "type": 10,
                    "parentTitle": "MF Doom",
                    "title": "Rapp Snitch Knishes",
                }
            )
            logging.debug(json.dumps(results, indent=2))
            song_id = results["MediaContainer"]["Metadata"][0]["key"]
            logging.debug(song_id)
            await client.create_playlist("Rapp Snitch Knishes", song_id)
            print("✓ Playlist created successfully")
