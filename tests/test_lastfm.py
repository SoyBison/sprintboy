"""
Unit tests for the Last.fm integration.

These are all mocked, so they run without a LASTFM_API_KEY and without touching
the network. They cover two things:

  1. Last.fm's JSON shape quirks. Collections collapse to a bare object when
     they hold one item and to "" when empty, numbers arrive as strings, absent
     ids arrive as "", and API errors arrive with HTTP 200.
  2. The collision logic in check_for_album, which is the whole point of
     canonicalising names in the first place.
"""

from unittest.mock import AsyncMock, patch

import pytest

from bot.netcode import (
    CanonicalRelease,
    LastFMClient,
    LastFMError,
    _lastfm_list,
    dedupe_casefold,
    resolve_release,
)
from bot.tools import _album_scores, check_for_album


def _make_client() -> LastFMClient:
    """A client with an api_key but no session, for use with _get patched out."""
    client = LastFMClient.__new__(LastFMClient)
    client.base_url = "https://ws.audioscrobbler.com/2.0/"
    client.api_key = "test-key"
    client.session = None
    return client


# ---------------------------------------------------------------------------
# Response shape normalisation
# ---------------------------------------------------------------------------


class TestLastFMListShapes:
    def test_single_item_collapses_to_dict(self):
        """One-element collections come back as a bare object, not a list."""
        assert _lastfm_list({"name": "Boards of Canada"}) == [
            {"name": "Boards of Canada"}
        ]

    def test_empty_collection_is_a_string(self):
        """Last.fm sends "" or a lone newline for an empty collection."""
        assert _lastfm_list("") == []
        assert _lastfm_list("\n") == []
        assert _lastfm_list(None) == []

    def test_list_is_passed_through_without_non_dicts(self):
        assert _lastfm_list([{"name": "a"}, "junk", {"name": "b"}]) == [
            {"name": "a"},
            {"name": "b"},
        ]


class TestDedupeCasefold:
    def test_drops_case_duplicates_and_blanks(self):
        assert dedupe_casefold(["MF DOOM", "mf doom", "", None, "Madvillain"]) == [
            "MF DOOM",
            "Madvillain",
        ]

    def test_keeps_first_spelling_seen(self):
        assert dedupe_casefold(["Guns N' Roses", "guns n' roses"]) == ["Guns N' Roses"]


# ---------------------------------------------------------------------------
# Error handling: Last.fm reports failures with HTTP 200
# ---------------------------------------------------------------------------


class TestLastFMErrors:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_on_enter(self):
        client = LastFMClient.__new__(LastFMClient)
        client.base_url = "https://ws.audioscrobbler.com/2.0/"
        client.api_key = None
        client.session = None
        with pytest.raises(LastFMError, match="LASTFM_API_KEY is not set"):
            await client.__aenter__()

    @pytest.mark.asyncio
    async def test_artist_info_raises_when_artist_absent(self):
        client = _make_client()
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {}
            with pytest.raises(LastFMError, match="no artist"):
                await client.get_artist_info("Nonexistent Band")


# ---------------------------------------------------------------------------
# artist.getCorrection: the endpoint goal C depends on
# ---------------------------------------------------------------------------


class TestCorrectArtist:
    @pytest.mark.asyncio
    async def test_returns_canonical_spelling(self):
        client = _make_client()
        body = {
            "corrections": {
                "correction": {
                    "artist": {
                        "name": "Guns N' Roses",
                        "mbid": "eeb1195b-f213-4ce1-b28c-8565211f8e43",
                        "url": "https://www.last.fm/music/Guns+N%27+Roses",
                    },
                    "@attr": {"index": "0"},
                }
            }
        }
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            result = await client.correct_artist("guns and roses")

        assert result is not None
        assert result.name == "Guns N' Roses"
        assert result.mbid == "eeb1195b-f213-4ce1-b28c-8565211f8e43"

    @pytest.mark.asyncio
    async def test_no_correction_available_is_an_empty_string_not_null(self):
        """
        When Last.fm has no correction it sends {"corrections": "\\n"}, not null
        and not an empty object. Treating that as a dict raises AttributeError.
        """
        client = _make_client()
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"corrections": "\n"}
            assert await client.correct_artist("MF DOOM") is None

    @pytest.mark.asyncio
    async def test_correction_without_a_name_is_ignored(self):
        client = _make_client()
        body = {"corrections": {"correction": {"artist": {"name": "", "mbid": ""}}}}
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            assert await client.correct_artist("whoever") is None


# ---------------------------------------------------------------------------
# Collection endpoints
# ---------------------------------------------------------------------------


class TestArtistAlbums:
    @pytest.mark.asyncio
    async def test_parses_nested_artist_and_string_playcount(self):
        client = _make_client()
        body = {
            "topalbums": {
                "album": [
                    {
                        "name": "Music Has the Right to Children",
                        "playcount": "1234567",
                        "mbid": "abc",
                        "artist": {"name": "Boards of Canada", "mbid": "def"},
                    }
                ]
            }
        }
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            albums = await client.get_artist_albums("boards of canada")

        assert len(albums) == 1
        assert albums[0].name == "Music Has the Right to Children"
        assert albums[0].artist == "Boards of Canada"
        assert albums[0].playcount == 1234567

    @pytest.mark.asyncio
    async def test_drops_the_null_placeholder_release(self):
        """Last.fm lists unnamed releases as the literal string "(null)"."""
        client = _make_client()
        body = {
            "topalbums": {
                "album": [
                    {"name": "(null)", "artist": {"name": "Some Artist"}},
                    {"name": "A Real Album", "artist": {"name": "Some Artist"}},
                ]
            }
        }
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            albums = await client.get_artist_albums("some artist")

        assert [album.name for album in albums] == ["A Real Album"]

    @pytest.mark.asyncio
    async def test_empty_topalbums_string_yields_no_albums(self):
        client = _make_client()
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"topalbums": ""}
            assert await client.get_artist_albums("who") == []


class TestTagTopAlbums:
    @pytest.mark.asyncio
    async def test_accepts_the_json_albums_key(self):
        client = _make_client()
        body = {"albums": {"album": [{"name": "Shore", "artist": {"name": "Fleet Foxes"}}]}}
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            albums = await client.get_tag_top_albums("folk")

        assert [(a.name, a.artist) for a in albums] == [("Shore", "Fleet Foxes")]

    @pytest.mark.asyncio
    async def test_accepts_the_documented_topalbums_key(self):
        """The XML schema calls this topalbums; accept it so either shape works."""
        client = _make_client()
        body = {"topalbums": {"album": [{"name": "Shore", "artist": {"name": "Fleet Foxes"}}]}}
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            albums = await client.get_tag_top_albums("folk")

        assert [a.name for a in albums] == ["Shore"]


class TestSimilarArtists:
    @pytest.mark.asyncio
    async def test_parses_string_match_score(self):
        client = _make_client()
        body = {
            "similarartists": {
                "artist": [{"name": "Autechre", "match": "0.912", "mbid": ""}]
            }
        }
        with patch.object(LastFMClient, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = body
            similar = await client.get_similar_artists("Aphex Twin")

        assert similar[0].name == "Autechre"
        assert similar[0].match == pytest.approx(0.912)
        # An absent mbid arrives as "", which must normalise to None.
        assert similar[0].mbid is None


# ---------------------------------------------------------------------------
# resolve_release: turning a user's wording into spellings to query Plex with
# ---------------------------------------------------------------------------


class TestResolveRelease:
    @pytest.mark.asyncio
    async def test_keeps_both_spellings_as_plex_candidates(self):
        """
        The corrected spelling is what Plex probably has, but the user's spelling
        might be too, so both get queried.
        """
        correction = AsyncMock(return_value=type("A", (), {"name": "Guns N' Roses", "mbid": "gnr"})())
        with patch.object(LastFMClient, "__aenter__", new_callable=AsyncMock) as enter:
            client = _make_client()
            client.correct_artist = correction  # type: ignore[method-assign]
            enter.return_value = client
            with patch.object(LastFMClient, "__aexit__", new_callable=AsyncMock):
                release = await resolve_release("guns and roses")

        assert release.artist == "Guns N' Roses"
        assert release.artist_candidates == ["Guns N' Roses", "guns and roses"]
        assert release.artist_mbid == "gnr"
        assert release.corrected is True

    @pytest.mark.asyncio
    async def test_not_corrected_when_only_the_casing_differs(self):
        correction = AsyncMock(return_value=type("A", (), {"name": "MF DOOM", "mbid": None})())
        with patch.object(LastFMClient, "__aenter__", new_callable=AsyncMock) as enter:
            client = _make_client()
            client.correct_artist = correction  # type: ignore[method-assign]
            enter.return_value = client
            with patch.object(LastFMClient, "__aexit__", new_callable=AsyncMock):
                release = await resolve_release("mf doom")

        assert release.corrected is False
        assert release.artist_candidates == ["MF DOOM"]

    @pytest.mark.asyncio
    async def test_album_mbid_is_returned(self):
        """The album MusicBrainz id is what makes a resolved release unambiguous."""
        client = _make_client()
        client.correct_artist = AsyncMock(  # type: ignore[method-assign]
            return_value=type("A", (), {"name": "Fleet Foxes", "mbid": "ff-mbid"})()
        )
        client.get_album_info = AsyncMock(  # type: ignore[method-assign]
            return_value=type(
                "B", (), {"name": "Shore", "mbid": "shore-mbid", "artist": "Fleet Foxes"}
            )()
        )
        with patch.object(LastFMClient, "__aenter__", new_callable=AsyncMock) as enter:
            enter.return_value = client
            with patch.object(LastFMClient, "__aexit__", new_callable=AsyncMock):
                release = await resolve_release("fleet foxes", "shore")

        assert release.album == "Shore"
        assert release.album_mbid == "shore-mbid"
        assert release.artist_mbid == "ff-mbid"
        # "shore" only differs in case, so it collapses into one Plex query.
        assert release.album_candidates == ["Shore"]

    @pytest.mark.asyncio
    async def test_unknown_album_still_resolves_the_artist(self):
        """album.getInfo failing is not a reason to give up on the artist name."""
        client = _make_client()
        client.correct_artist = AsyncMock(  # type: ignore[method-assign]
            return_value=type("A", (), {"name": "Fleet Foxes", "mbid": "ff"})()
        )
        client.get_album_info = AsyncMock(side_effect=LastFMError("not found"))  # type: ignore[method-assign]

        with patch.object(LastFMClient, "__aenter__", new_callable=AsyncMock) as enter:
            enter.return_value = client
            with patch.object(LastFMClient, "__aexit__", new_callable=AsyncMock):
                release = await resolve_release("fleet foxes", "Not A Real Album")

        assert release.artist == "Fleet Foxes"
        assert release.album == "Not A Real Album"
        assert release.album_candidates == ["Not A Real Album"]


# ---------------------------------------------------------------------------
# Album title scoring: the tier boundary between a collision and a maybe
# ---------------------------------------------------------------------------


class TestAlbumScores:
    def test_punctuation_difference_is_a_whole_title_match(self):
        whole, _ = _album_scores(["MM..FOOD"], "MM.. FOOD")
        assert whole >= 90

    def test_word_order_difference_is_a_whole_title_match(self):
        whole, _ = _album_scores(["Rated R"], "R Rated")
        assert whole >= 90

    def test_deluxe_edition_is_contained_but_not_a_whole_match(self):
        whole, contained = _album_scores(["Shore"], "Shore (Deluxe Edition)")
        assert contained >= 90
        assert whole < 90

    def test_shared_prefix_is_not_a_whole_match(self):
        """
        "Metallica" is a substring of "Metallica Through the Never", so
        containment alone must never be reported as a collision.
        """
        whole, contained = _album_scores(["Metallica"], "Metallica Through the Never")
        assert whole < 90
        assert contained >= 90

    def test_unrelated_albums_score_low_on_both(self):
        whole, contained = _album_scores(["Kid A"], "The Dark Side of the Moon")
        assert whole < 90
        assert contained < 90


# ---------------------------------------------------------------------------
# check_for_album: does the canonicalisation actually prevent a re-download?
# ---------------------------------------------------------------------------


def _plex_response(*albums: tuple[str, str]) -> dict:
    return {
        "MediaContainer": {
            "size": len(albums),
            "Metadata": [
                {"ratingKey": str(i), "title": title, "parentTitle": artist}
                for i, (title, artist) in enumerate(albums)
            ],
        }
    }


def _mock_plex(responses: dict[str, dict]):
    """Patch PlexAPIClient so library lookups return canned data per artist name."""
    calls: list[str] = []

    async def get_all_library_items(query_params):
        artist = query_params["artist.title"]
        calls.append(artist)
        return responses.get(artist, {"MediaContainer": {"size": 0}})

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get_all_library_items = get_all_library_items
    return client, calls


class TestCheckForAlbum:
    @pytest.mark.asyncio
    async def test_alias_in_request_still_finds_the_owned_album(self):
        """
        The collision this integration exists to prevent: the user asks for
        "guns and roses", the library holds "Guns N' Roses", and the old
        exact-match check reported it as absent.
        """
        plex, calls = _mock_plex(
            {"Guns N' Roses": _plex_response(("Appetite for Destruction", "Guns N' Roses"))}
        )
        release = CanonicalRelease(
            artist="Guns N' Roses",
            artist_candidates=["Guns N' Roses", "guns and roses"],
            album="Appetite for Destruction",
            album_candidates=["Appetite for Destruction"],
            corrected=True,
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "guns and roses", "title": "Appetite for Destruction"}
                )

        assert "COLLISION" in result
        assert "Appetite for Destruction" in result
        # Both spellings are queried, not just the canonical one.
        assert set(calls) == {"Guns N' Roses", "guns and roses"}

    @pytest.mark.asyncio
    async def test_punctuation_difference_is_reported_as_a_collision(self):
        plex, _ = _mock_plex({"MF DOOM": _plex_response(("MM.. FOOD", "MF DOOM"))})
        release = CanonicalRelease(
            artist="MF DOOM",
            artist_candidates=["MF DOOM"],
            album="MM..FOOD",
            album_candidates=["MM..FOOD"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "MF DOOM", "title": "MM..FOOD"}
                )

        assert "COLLISION" in result

    @pytest.mark.asyncio
    async def test_shared_prefix_is_not_reported_as_a_collision(self):
        """A different album that starts with the same word must stay downloadable."""
        plex, _ = _mock_plex(
            {"Metallica": _plex_response(("Metallica Through the Never", "Metallica"))}
        )
        release = CanonicalRelease(
            artist="Metallica",
            artist_candidates=["Metallica"],
            album="Metallica",
            album_candidates=["Metallica"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "Metallica", "title": "Metallica"}
                )

        assert "COLLISION" not in result
        assert "may be the same release" in result

    @pytest.mark.asyncio
    async def test_absent_album_is_reported_as_safe_to_download(self):
        plex, _ = _mock_plex(
            {"Fleet Foxes": _plex_response(("Helplessness Blues", "Fleet Foxes"))}
        )
        release = CanonicalRelease(
            artist="Fleet Foxes",
            artist_candidates=["Fleet Foxes"],
            album="Shore",
            album_candidates=["Shore"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "Fleet Foxes", "title": "Shore"}
                )

        assert "does NOT have" in result
        assert "Helplessness Blues" in result

    @pytest.mark.asyncio
    async def test_no_albums_by_artist_at_all(self):
        plex, _ = _mock_plex({})
        release = CanonicalRelease(
            artist="Nonexistent",
            artist_candidates=["Nonexistent"],
            album="Whatever",
            album_candidates=["Whatever"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "Nonexistent", "title": "Whatever"}
                )

        assert "no albums by Nonexistent" in result

    @pytest.mark.asyncio
    async def test_omitting_the_title_lists_the_whole_artist(self):
        plex, _ = _mock_plex(
            {
                "Fleet Foxes": _plex_response(
                    ("Shore", "Fleet Foxes"), ("Helplessness Blues", "Fleet Foxes")
                )
            }
        )
        release = CanonicalRelease(
            artist="Fleet Foxes", artist_candidates=["Fleet Foxes"]
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "Fleet Foxes", "title": None}
                )

        assert "already has these albums" in result
        assert "Shore" in result
        assert "Helplessness Blues" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_plain_names_when_lastfm_is_unavailable(self):
        """A missing API key must degrade the check, not break it."""
        plex, calls = _mock_plex({"MF DOOM": _plex_response(("MM..FOOD", "MF DOOM"))})
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.side_effect = LastFMError("LASTFM_API_KEY is not set")
                result = await check_for_album.ainvoke(
                    {"artist": "MF DOOM", "title": "MM..FOOD"}
                )

        assert calls == ["MF DOOM"]
        assert "COLLISION" in result

    @pytest.mark.asyncio
    async def test_one_failing_artist_lookup_does_not_lose_the_other(self):
        plex, _ = _mock_plex({"MF DOOM": _plex_response(("MM..FOOD", "MF DOOM"))})
        original = plex.get_all_library_items

        async def flaky(query_params):
            if query_params["artist.title"] == "Metal Fingers":
                raise RuntimeError("Plex timed out")
            return await original(query_params)

        plex.get_all_library_items = flaky
        release = CanonicalRelease(
            artist="MF DOOM",
            artist_candidates=["Metal Fingers", "MF DOOM"],
            album="MM..FOOD",
            album_candidates=["MM..FOOD"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "MF DOOM", "title": "MM..FOOD"}
                )

        assert "COLLISION" in result

    @pytest.mark.asyncio
    async def test_same_album_under_two_artist_spellings_reported_once(self):
        response = _plex_response(("MM..FOOD", "MF DOOM"))
        plex, _ = _mock_plex({"MF DOOM": response, "MF Doom": response})
        release = CanonicalRelease(
            artist="MF DOOM",
            artist_candidates=["MF DOOM", "MF Doom"],
            album="MM..FOOD",
            album_candidates=["MM..FOOD"],
        )
        with patch("bot.tools.PlexAPIClient", return_value=plex):
            with patch("bot.tools.resolve_release", new_callable=AsyncMock) as resolve:
                resolve.return_value = release
                result = await check_for_album.ainvoke(
                    {"artist": "MF DOOM", "title": "MM..FOOD"}
                )

        # The title also appears in the prose, so count only the listed albums.
        listed = [line for line in result.splitlines() if line.startswith("- ")]
        assert listed == ["- MM..FOOD by MF DOOM"]
