from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.decide import Decision
from bot.netcode import LastFMAlbum, LastFMAlbumInfo, LastFMError
from bot.questions import SPECIFIC_EXTRACT_NAME, SPECIFIC_PICK_NAME
from bot.tools import TorrentContext
from bot.workflows import Choice, Pending, specific


def _ctx():
    return TorrentContext(search_results={}, internal_torrents={}, torrent_types=set())


def extract_decision(state, *, albums=(), artist=None, track=None, p=0.95):
    spans = state["spans"]
    answers = {
        "artist": {
            "choice": artist or "none",
            "probabilities": {artist or "none": p},
        },
        "track": {"choice": track or "none", "probabilities": {track or "none": p}},
    }
    wanted = {a.casefold() for a in albums}
    for key, span in spans.items():
        answers[key] = {"noul": 0.9 if span.casefold() in wanted else 0.05}
    return Decision("jev", "m", answers, 0.1)


def pick_decision(choice="r0", p=0.95, probabilities=None):
    return Decision(
        "jev", "m",
        {"pick": {"choice": choice, "probabilities": probabilities or {choice: p}}},
        0.1,
    )


def fake_lastfm(
    *, albums=None, search=None, tracks=None, track_album=None, artist_albums=None, info=None
):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.correct_artist = AsyncMock(return_value=None)

    async def get_info(artist, album):
        if info is None:
            raise LastFMError("no album")
        return info

    client.get_album_info = AsyncMock(side_effect=get_info)
    client.get_artist_albums = AsyncMock(
        side_effect=lambda artist, limit=20, page=1: list((artist_albums or {}).get(artist, []))
    )
    client.search_albums = AsyncMock(
        side_effect=lambda album, limit=8: list((search or {}).get(album, []))
    )
    client.search_tracks = AsyncMock(return_value=list(tracks or []))
    client.get_track_album = AsyncMock(side_effect=lambda a, t: (track_album or {}).get(t))
    return client


def added(ref):
    return (
        f"Added '{ref.artist} - {ref.title} [2020] [Album] FLAC / Lossless / WEB [Orpheus]', "
        f"confirmed present in qBittorrent."
    )


async def run(
    text,
    *,
    extract,
    pick=None,
    lastfm=None,
    download=None,
    earlier=None,
    tracker=(),
):
    lastfm = lastfm or fake_lastfm()

    async def fake_decide(name, state, questions, run_id=None):
        if name == SPECIFIC_EXTRACT_NAME:
            return extract(state)
        assert name == SPECIFIC_PICK_NAME
        return pick(state) if callable(pick) else pick

    async def fake_download(refs, context, media=None):
        return [added(r) for r in refs]

    qclient = MagicMock()
    qclient.__aenter__ = AsyncMock(return_value=qclient)
    qclient.__aexit__ = AsyncMock(return_value=False)
    qclient.search = AsyncMock(return_value=SimpleNamespace(results=list(tracker)))
    mocks = SimpleNamespace(
        decide=AsyncMock(side_effect=fake_decide),
        download=AsyncMock(side_effect=download or fake_download),
        lastfm=lastfm,
        qclient=qclient,
    )
    with patch("bot.workflows.decide", mocks.decide), patch(
        "bot.workflows.LastFMClient", return_value=lastfm
    ), patch("bot.workflows.QBittorrentClient", return_value=qclient), patch(
        "bot.workflows.download_many", mocks.download
    ):
        result = await specific(text, _ctx(), earlier=earlier, run_id="r1")
    return result, mocks


def downloaded(mocks):
    return [[(r.artist, r.title) for r in call.args[0]] for call in mocks.download.await_args_list]


def picks(mocks):
    return [c for c in mocks.decide.await_args_list if c.args[0] == SPECIFIC_PICK_NAME]


MORDECHAI = LastFMAlbum(name="Mordechai", artist="Khruangbin")


@pytest.mark.asyncio
async def test_album_by_artist():
    lastfm = fake_lastfm(
        info=LastFMAlbumInfo(name="Mordechai", artist="Khruangbin"),
        artist_albums={"Khruangbin": [MORDECHAI, LastFMAlbum(name="Texas Sun", artist="Khruangbin")]},
        search={"Mordechai": [MORDECHAI]},
    )
    result, mocks = await run(
        "get me Mordechai by Khruangbin",
        extract=lambda s: extract_decision(s, albums=["Mordechai"], artist="Khruangbin"),
        pick=pick_decision("r0", 0.95),
        lastfm=lastfm,
    )
    assert downloaded(mocks) == [[("Khruangbin", "Mordechai")]]
    assert result.reply == "Added Khruangbin - Mordechai."
    assert result.pending == []
    assert [s.name for s in result.steps] == [
        "specific/extract", "specific/candidates", "specific/popularity", "specific/pick", "specific/download"
    ]
    state = picks(mocks)[0].args[1]
    assert state["wanted"] == {"title": "Mordechai", "artist": "Khruangbin", "track": None}
    assert state["candidates"] == {"r0": {"artist": "Khruangbin", "title": "Mordechai"}}


@pytest.mark.asyncio
async def test_two_albums_are_downloaded_in_one_call():
    lastfm = fake_lastfm(
        search={
            "Loveless": [LastFMAlbum(name="Loveless", artist="My Bloody Valentine")],
            "Souvlaki": [LastFMAlbum(name="Souvlaki", artist="Slowdive")],
        }
    )
    result, mocks = await run(
        "download Loveless and Souvlaki",
        extract=lambda s: extract_decision(s, albums=["Loveless", "Souvlaki"]),
        pick=pick_decision("r0", 0.95),
        lastfm=lastfm,
    )
    assert len(picks(mocks)) == 2
    assert downloaded(mocks) == [
        [("My Bloody Valentine", "Loveless"), ("Slowdive", "Souvlaki")]
    ]
    assert result.reply == (
        "Added:\n- My Bloody Valentine - Loveless\n- Slowdive - Souvlaki"
    )


@pytest.mark.asyncio
async def test_misspelling_resolves_through_search():
    lastfm = fake_lastfm(
        search={
            "Rainbowz": [
                LastFMAlbum(name="In Rainbows", artist="Radiohead"),
                LastFMAlbum(name="In Rainbows Disk 2", artist="Radiohead"),
            ]
        }
    )
    result, mocks = await run(
        "get me In Rainbowz",
        extract=lambda s: extract_decision(s, albums=["Rainbowz"]),
        pick=pick_decision("r0", 0.9),
        lastfm=lastfm,
    )
    labels = list(picks(mocks)[0].args[1]["candidates"].values())
    titles = [c["title"] for c in labels]
    assert "In Rainbows" in titles and "In Rainbows Disk 2" in titles
    assert downloaded(mocks) == [[("Radiohead", "In Rainbows")]]


@pytest.mark.asyncio
async def test_ambiguous_pick_asks_instead_of_downloading():
    lastfm = fake_lastfm(
        search={
            "Rainbowz": [
                LastFMAlbum(name="In Rainbows", artist="Radiohead"),
                LastFMAlbum(name="In Rainbows Disk 2", artist="Radiohead"),
            ]
        }
    )
    result, mocks = await run(
        "get me Rainbowz",
        extract=lambda s: extract_decision(s, albums=["Rainbowz"]),
        pick=pick_decision("r0", 0.45, {"r0": 0.45, "r1": 0.35, "none": 0.2}),
        lastfm=lastfm,
    )
    assert mocks.download.await_count == 0
    assert result.pending == [
        Pending(
            "Rainbowz",
            [
                Choice("Radiohead - In Rainbows", "Radiohead", "In Rainbows"),
                Choice("Radiohead - In Rainbows Disk 2", "Radiohead", "In Rainbows Disk 2"),
            ],
        )
    ]
    assert 'Did you mean one of these for "Rainbowz"?' in result.reply


@pytest.mark.asyncio
async def test_a_single_plausible_option_is_not_offered_alone():
    lastfm = fake_lastfm(search={"Rainbowz": [LastFMAlbum(name="In Rainbows", artist="Radiohead")]})
    result, mocks = await run(
        "get me Rainbowz",
        extract=lambda s: extract_decision(s, albums=["Rainbowz"]),
        pick=pick_decision("r0", 0.6, {"r0": 0.6, "none": 0.4}),
        lastfm=lastfm,
    )
    assert downloaded(mocks) == [[("Radiohead", "In Rainbows")]]
    assert result.pending == []


@pytest.mark.asyncio
async def test_none_of_them_offers_the_closest_by_spelling():
    lastfm = fake_lastfm(
        search={
            "Djesse Vol 5": [
                LastFMAlbum(name="Djesse Vol. 4", artist="Jacob Collier"),
                LastFMAlbum(name="Djesse Vol. 3", artist="Jacob Collier"),
                LastFMAlbum(name="Zzzz", artist="Someone"),
            ]
        }
    )
    result, mocks = await run(
        "grab Djesse Vol 5",
        extract=lambda s: extract_decision(s, albums=["Djesse Vol 5"]),
        pick=pick_decision("none", 0.9, {"none": 0.9, "r0": 0.05, "r1": 0.05}),
        lastfm=lastfm,
    )
    assert mocks.download.await_count == 0
    assert [o.title for o in result.pending[0].options] == ["Djesse Vol. 4", "Djesse Vol. 3"]


@pytest.mark.asyncio
async def test_track_path_finds_the_album_through_the_song():
    lastfm = fake_lastfm(
        tracks=[("Weather Report", "Fast City")],
        track_album={"Fast City": LastFMAlbum(name="Night Passage", artist="Weather Report")},
    )
    result, mocks = await run(
        "get me the weather report album with fast city",
        extract=lambda s: extract_decision(s, artist="Weather Report", track="fast city"),
        pick=pick_decision("r0", 0.9),
        lastfm=lastfm,
    )
    lastfm.search_tracks.assert_awaited_once_with("fast city", "Weather Report", limit=5)
    state = picks(mocks)[0].args[1]
    assert state["wanted"]["title"] == "the album with fast city"
    assert state["wanted"]["track"] == "fast city"
    assert state["candidates"] == {
        "r0": {"artist": "Weather Report", "title": "Night Passage", "has_track": "fast city"}
    }
    assert downloaded(mocks) == [[("Weather Report", "Night Passage")]]


@pytest.mark.asyncio
async def test_follow_up_offers_spans_from_the_earlier_turns():
    earlier = [
        {"role": "user", "content": "get me the weather report album with fast city"},
        {"role": "assistant", "content": "Added Weather Report - Heavy Weather."},
    ]
    lastfm = fake_lastfm(
        search={"night passage": [LastFMAlbum(name="Night Passage", artist="Weather Report")]}
    )
    result, mocks = await run(
        "It's on night passage but nice try",
        extract=lambda s: extract_decision(s, albums=["night passage"], artist="weather report"),
        pick=pick_decision("r0", 0.9),
        lastfm=lastfm,
        earlier=earlier,
    )
    extract_call = mocks.decide.await_args_list[0]
    offered = [s.casefold() for s in extract_call.args[1]["spans"].values()]
    assert "weather report" in offered
    assert "weather report" in [k.casefold() for k in extract_call.args[2]["artist"]["criteria"]]
    lastfm.correct_artist.assert_awaited_once_with("weather report")
    assert downloaded(mocks) == [[("Weather Report", "Night Passage")]]


@pytest.mark.asyncio
async def test_extract_has_one_noul_per_span_and_a_spans_map():
    _, mocks = await run(
        "get me Mordechai by Khruangbin",
        extract=lambda s: extract_decision(s),
    )
    call = mocks.decide.await_args_list[0]
    state, questions = call.args[1], call.args[2]
    spans = state["spans"]
    assert state["message"] == "get me Mordechai by Khruangbin"
    assert spans and all(k.startswith("album_") for k in spans)
    noul = {k for k, q in questions.items() if q["type"] == "noul"}
    assert noul == set(spans)
    assert "`spans.album_0`" in questions["album_0"]["instructions"]
    assert set(questions["artist"]["criteria"]) == set(spans.values()) | {"none"}


@pytest.mark.asyncio
async def test_a_longer_span_replaces_the_one_inside_it():
    lastfm = fake_lastfm()
    _, mocks = await run(
        "get the deluxe edition of Djesse Vol. 4",
        extract=lambda s: extract_decision(s, albums=["Djesse Vol", "Djesse Vol 4"]),
        pick=pick_decision("none"),
        lastfm=lastfm,
    )
    searched = [c.args[0] for c in lastfm.search_albums.await_args_list]
    assert searched == ["Djesse Vol 4"]


@pytest.mark.asyncio
async def test_owned_and_not_found_wording():
    lastfm = fake_lastfm(
        search={
            "Loveless": [LastFMAlbum(name="Loveless", artist="My Bloody Valentine")],
            "Souvlaki": [LastFMAlbum(name="Souvlaki", artist="Slowdive")],
            "Pygmalion": [LastFMAlbum(name="Pygmalion", artist="Slowdive")],
        }
    )

    async def download(refs, context, media=None):
        return [
            "OWNED: 'My Bloody Valentine - Loveless' is already in the library; skipped.",
            "NOT FOUND: no FLAC torrent for 'Slowdive - Souvlaki'.",
            "NOT ADDED: QBITTORRENT_DRY_RUN is on, so 'x' was never submitted. Tell the user nothing was downloaded.",
        ]

    result, _ = await run(
        "get Loveless Souvlaki Pygmalion",
        extract=lambda s: extract_decision(s, albums=["Loveless", "Souvlaki", "Pygmalion"]),
        pick=pick_decision("r0", 0.95),
        lastfm=lastfm,
        download=download,
    )
    assert result.reply.splitlines() == [
        "You already have My Bloody Valentine - Loveless.",
        "Couldn't find a FLAC of Slowdive - Souvlaki on the tracker.",
        "Couldn't add Slowdive - Pygmalion: dry run is on",
    ]


@pytest.mark.asyncio
async def test_nothing_resolvable_defers_to_the_agent():
    result, mocks = await run(
        "get me Blorp",
        extract=lambda s: extract_decision(s, albums=["Blorp"]),
        pick=pick_decision("r0"),
    )
    assert result is None
    assert mocks.download.await_count == 0


@pytest.mark.asyncio
async def test_nothing_named_defers_to_the_agent():
    result, _ = await run("get me something", extract=lambda s: extract_decision(s))
    assert result is None


@pytest.mark.asyncio
async def test_no_decision_defers_to_the_agent():
    result, _ = await run("get me Mordechai", extract=lambda s: None)
    assert result is None


@pytest.mark.asyncio
async def test_unfound_album_is_reported_next_to_a_found_one():
    lastfm = fake_lastfm(
        search={"Loveless": [LastFMAlbum(name="Loveless", artist="My Bloody Valentine")]}
    )
    result, _ = await run(
        "get Loveless and Blorp",
        extract=lambda s: extract_decision(s, albums=["Loveless", "Blorp"]),
        pick=pick_decision("r0"),
        lastfm=lastfm,
    )
    assert result.reply.splitlines() == [
        "Added My Bloody Valentine - Loveless.",
        'Couldn\'t find anything called "Blorp".',
    ]


@pytest.mark.asyncio
async def test_the_artist_span_is_not_an_album_when_a_song_is_named():
    lastfm = fake_lastfm(
        tracks=[("Weather Report", "Fast City")],
        track_album={"Fast City": LastFMAlbum(name="Night Passage", artist="Weather Report")},
    )
    _, mocks = await run(
        "get me the weather report album with fast city",
        extract=lambda s: extract_decision(
            s, albums=["weather report"], artist="weather report", track="fast city"
        ),
        pick=pick_decision("r0", 0.9),
        lastfm=lastfm,
    )
    assert downloaded(mocks) == [[("Weather Report", "Night Passage")]]
    lastfm.search_albums.assert_not_awaited()


@pytest.mark.asyncio
async def test_words_named_as_both_artist_and_album_are_searched_without_the_artist():
    lastfm = fake_lastfm(search={"rainbowz": [LastFMAlbum(name="In Rainbows", artist="Radiohead")]})
    _, mocks = await run(
        "get me rainbowz",
        extract=lambda s: extract_decision(s, albums=["rainbowz"], artist="rainbowz"),
        pick=pick_decision("r0", 0.9),
        lastfm=lastfm,
    )
    lastfm.correct_artist.assert_not_awaited()
    assert picks(mocks)[0].args[1]["wanted"]["artist"] is None
    assert downloaded(mocks) == [[("Radiohead", "In Rainbows")]]


@pytest.mark.asyncio
async def test_a_leading_in_stays_in_the_offered_spans():
    _, mocks = await run("get me In Rainbows", extract=lambda s: extract_decision(s))
    assert "In Rainbows" in mocks.decide.await_args_list[0].args[1]["spans"].values()
