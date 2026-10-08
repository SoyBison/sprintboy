from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot import turn
from bot.decide import Decision
from bot.llm import system, user
from bot.netcode import LastFMAlbum, LastFMArtist
from bot.questions import RECOMMEND_FIT_NAME, RECOMMEND_SEED_NAME
from bot.routing import Route
from bot.tools import TorrentContext
from bot.workflows import WorkflowResult, recommend

SIMILAR = {
    "Khruangbin": ["Hiatus Kaiyote", "Y Loop", "Superorganism", "Cut Worms"],
}
ALBUMS = {
    "Hiatus Kaiyote": ["Choose Your Weapon", "Tawk Tomahawk", "Best Of Hiatus"],
    "Y Loop": ["Y Loop One", "Y Loop Two"],
    "Superorganism": ["Superorganism"],
    "Cut Worms": ["Hollow Ground", "Nobody Lives Here Anymore"],
    "Khruangbin": ["Mordechai", "Con Todo El Mundo", "Khruangbin Live (Live"],
    "Mordechai Band": ["Heavier Things"],
}
SCORES = {
    "Hiatus Kaiyote - Choose Your Weapon": 2.4,
    "Hiatus Kaiyote - Tawk Tomahawk": 2.9,
    "Y Loop - Y Loop One": 1.2,
    "Y Loop - Y Loop Two": 1.1,
    "Superorganism - Superorganism": 0.2,
    "Cut Worms - Hollow Ground": 2.0,
    "Cut Worms - Nobody Lives Here Anymore": 3.0,
}


def _ctx():
    return TorrentContext(search_results={}, internal_torrents={}, torrent_types=set())


def seed_decision(seed="Khruangbin", seed_type="artist", p=0.95, same=0.05, new=0.1):
    return Decision(
        "jev", "m",
        {
            "seed": {"choice": seed, "probabilities": {seed: p}},
            "seed_type": {"choice": seed_type, "probabilities": {seed_type: 0.9}},
            "same_artist": {"noul": same},
            "new_to_them": {"noul": new},
        },
        0.1,
    )


def fit_decision(state, scores=SCORES):
    answers = {
        key: {"type": "score", "score": scores.get(label, 1.5), "confidence": 0.9}
        for key, label in state["candidates"].items()
    }
    return Decision("jev", "m", answers, 0.1)


def added_line(ref):
    return f"Added '{ref.artist} - {ref.title} [2020] [Album] FLAC / Lossless / WEB [Orpheus]', confirmed present in qBittorrent."


def fake_lastfm(similar=SIMILAR, albums=ALBUMS, tag_albums=(), tag_artists=(), found_album=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.correct_artist = AsyncMock(return_value=None)
    client.get_similar_artists = AsyncMock(
        side_effect=lambda artist, limit=20: [LastFMArtist(name=n) for n in similar.get(artist, [])]
    )
    client.get_artist_albums = AsyncMock(
        side_effect=lambda artist, limit=20, page=1: [
            LastFMAlbum(name=n, artist=artist) for n in albums.get(artist, [])[:limit]
        ]
    )
    client.get_tag_top_albums = AsyncMock(return_value=list(tag_albums))
    client.get_tag_top_artists = AsyncMock(return_value=[LastFMArtist(name=n) for n in tag_artists])
    client.search_album = AsyncMock(return_value=found_album)
    return client


async def run(
    text="get me 2 albums like Khruangbin",
    *,
    count=2,
    seed=None,
    fit=None,
    lastfm=None,
    library=None,
    download=None,
    earlier=None,
    vocab=("shoegaze", "chill"),
):
    seed = seed or seed_decision()
    lastfm = lastfm or fake_lastfm()
    library = library or {}

    async def fake_decide(name, state, questions, run_id=None):
        if name == RECOMMEND_SEED_NAME:
            return seed
        assert name == RECOMMEND_FIT_NAME
        return fit_decision(state) if fit is None else (fit(state) if callable(fit) else fit)

    async def fake_library(artists):
        return {a: library.get(a, []) for a in dict.fromkeys(artists)}

    async def fake_download(refs, context, media=None):
        return [added_line(r) for r in refs]

    mocks = SimpleNamespace(
        decide=AsyncMock(side_effect=fake_decide),
        download=AsyncMock(side_effect=download or fake_download),
        lastfm=lastfm,
    )
    with patch("bot.workflows.decide", mocks.decide), patch(
        "bot.workflows.LastFMClient", return_value=lastfm
    ), patch("bot.workflows.tag_vocabulary", AsyncMock(return_value=list(vocab))), patch(
        "bot.workflows._library_lookup", AsyncMock(side_effect=fake_library)
    ), patch("bot.workflows.download_many", mocks.download):
        result = await recommend(text, _ctx(), count=count, earlier=earlier, run_id="r1")
    return result, mocks


def downloaded(mocks):
    return [[(r.artist, r.title) for r in call.args[0]] for call in mocks.download.await_args_list]


@pytest.mark.asyncio
async def test_artist_seed_ranked_one_per_artist():
    result, mocks = await run()
    # Cut Worms' best (3.0) and Hiatus Kaiyote's best (2.9); one album per artist.
    assert downloaded(mocks) == [
        [("Cut Worms", "Nobody Lives Here Anymore"), ("Hiatus Kaiyote", "Tawk Tomahawk")]
    ]
    assert result.reply == (
        "Added 2 albums like Khruangbin:\n"
        "- Nobody Lives Here Anymore (2020) by Cut Worms\n"
        "- Tawk Tomahawk (2020) by Hiatus Kaiyote"
    )
    assert [s.name for s in result.steps] == [
        "recommend/seed", "recommend/candidates", "recommend/library",
        "recommend/rank", "recommend/download 1",
    ]


@pytest.mark.asyncio
async def test_compilations_and_the_seed_artist_are_not_candidates():
    _, mocks = await run()
    labels = list(mocks.decide.await_args_list[1].args[1]["candidates"].values())
    assert "Hiatus Kaiyote - Best Of Hiatus" not in labels
    assert not any(label.startswith("Khruangbin") for label in labels)
    assert len(labels) == 7


@pytest.mark.asyncio
async def test_owned_albums_are_dropped():
    _, mocks = await run(library={"Cut Worms": ["Nobody Lives Here Anymore"]})
    assert ("Cut Worms", "Nobody Lives Here Anymore") not in downloaded(mocks)[0]
    labels = list(mocks.decide.await_args_list[1].args[1]["candidates"].values())
    assert "Cut Worms - Nobody Lives Here Anymore" not in labels
    assert "Cut Worms - Hollow Ground" in labels


@pytest.mark.asyncio
async def test_new_to_them_drops_artists_in_the_library():
    library = {"Hiatus Kaiyote": ["Choose Your Weapon"], "Cut Worms": ["Hollow Ground"]}
    result, mocks = await run(seed=seed_decision(new=0.9), count=1, library=library)
    labels = list(mocks.decide.await_args_list[1].args[1]["candidates"].values())
    assert not any(l.startswith(("Hiatus Kaiyote", "Cut Worms")) for l in labels)
    assert downloaded(mocks) == [[("Y Loop", "Y Loop One")]]


@pytest.mark.asyncio
async def test_new_in_the_text_counts_as_new_to_them():
    library = {"Hiatus Kaiyote": ["Choose Your Weapon"]}
    _, mocks = await run("2 new albums like Khruangbin", count=1, library=library)
    labels = list(mocks.decide.await_args_list[1].args[1]["candidates"].values())
    assert not any(l.startswith("Hiatus Kaiyote") for l in labels)


@pytest.mark.asyncio
async def test_known_artists_kept_but_ranked_last_when_too_few_new():
    library = {a: ["x"] for a in ("Hiatus Kaiyote", "Y Loop", "Cut Worms")}
    _, mocks = await run(
        seed=seed_decision(new=0.9), count=2, library=library,
        fit=lambda s: fit_decision(s, {**SCORES, "Superorganism - Superorganism": 2.0}),
    )
    # Only Superorganism is new; the best known artist's album follows it.
    assert downloaded(mocks) == [[("Superorganism", "Superorganism"), ("Cut Worms", "Nobody Lives Here Anymore")]]


@pytest.mark.asyncio
async def test_replacement_round_when_the_first_pick_is_not_found():
    calls = []

    async def download(refs, context, media=None):
        calls.append([r.artist for r in refs])
        return [
            f"NOT FOUND: no FLAC torrent for '{r.artist} - {r.title}'."
            if r.artist == "Cut Worms" else added_line(r)
            for r in refs
        ]

    result, mocks = await run(download=download)
    assert calls == [["Cut Worms", "Hiatus Kaiyote"], ["Y Loop"]]
    assert result.reply.startswith("Added 2 albums like Khruangbin:")
    assert "Y Loop One" in result.reply and "Cut Worms" not in result.reply
    assert [s.name for s in result.steps][-2:] == ["recommend/download 1", "recommend/download 2"]


@pytest.mark.asyncio
async def test_gives_up_after_four_rounds_and_reports():
    async def download(refs, context, media=None):
        return [f"NOT FOUND: no FLAC torrent for '{r.artist} - {r.title}'." for r in refs]

    result, mocks = await run(download=download, count=1, fit=lambda s: fit_decision(s, {}))
    assert len(mocks.download.await_args_list) == 4
    assert result.reply.startswith("Nothing added: no FLAC torrent for ")


@pytest.mark.asyncio
async def test_short_of_the_count_is_reported():
    async def download(refs, context, media=None):
        return [
            f"NOT FOUND: no FLAC torrent for '{r.artist} - {r.title}'." if r.artist != "Y Loop"
            else added_line(r)
            for r in refs
        ]

    result, _ = await run(download=download, count=2)
    assert result.reply.startswith("Added 1 album like Khruangbin:\n- Y Loop One (2020) by Y Loop")
    assert "Only found 1 of 2: no FLAC torrent for " in result.reply


@pytest.mark.asyncio
async def test_dry_run_stops_after_one_round():
    async def download(refs, context, media=None):
        return [
            f"NOT ADDED: QBITTORRENT_DRY_RUN is on, so '{r.title}' was never submitted." for r in refs
        ]

    result, mocks = await run(download=download)
    assert len(mocks.download.await_args_list) == 1
    assert result.reply == (
        "Nothing added: dry run is on. Would have added: "
        "Nobody Lives Here Anymore by Cut Worms, Tawk Tomahawk by Hiatus Kaiyote."
    )


@pytest.mark.asyncio
async def test_media_preference_is_passed_on():
    _, mocks = await run("2 albums like Khruangbin on vinyl")
    assert mocks.download.await_args.kwargs["media"] == "Vinyl"


@pytest.mark.asyncio
async def test_tag_seed():
    lastfm = fake_lastfm(
        tag_albums=[
            LastFMAlbum(name="Loveless", artist="My Bloody Valentine"),
            LastFMAlbum(name="Souvlaki", artist="Slowdive"),
            LastFMAlbum(name="Souvlaki (Remix)", artist="Slowdive"),
        ],
        tag_artists=["Ride"],
        albums={"Ride": ["Nowhere", "Going Blank Again"]},
    )
    result, mocks = await run(
        "throw some new shoegaze at me", count=3, seed=seed_decision("shoegaze", "tag"),
        lastfm=lastfm, fit=lambda state: fit_decision(state, {}),
    )
    lastfm.get_tag_top_albums.assert_awaited_once_with("shoegaze", limit=50)
    assert result.reply.startswith("Added 3 albums for shoegaze:")
    labels = list(mocks.decide.await_args_list[1].args[1]["candidates"].values())
    assert labels == [
        "My Bloody Valentine - Loveless", "Ride - Nowhere", "Slowdive - Souvlaki",
        "Ride - Going Blank Again",
    ]
    # Equal scores keep Last.fm order, and each artist is used once.
    assert downloaded(mocks) == [
        [("My Bloody Valentine", "Loveless"), ("Ride", "Nowhere"), ("Slowdive", "Souvlaki")]
    ]


@pytest.mark.asyncio
async def test_album_seed_resolves_artist_and_excludes_the_album():
    lastfm = fake_lastfm(
        similar={"Khruangbin": ["Cut Worms"]},
        albums={"Cut Worms": ["Hollow Ground"], "Khruangbin": ["Mordechai"]},
        found_album=LastFMAlbum(name="Mordechai", artist="Khruangbin"),
    )
    result, mocks = await run(
        "find me a couple albums like Mordechai but heavier", count=2,
        seed=seed_decision("Mordechai", "album"), lastfm=lastfm,
    )
    lastfm.search_album.assert_awaited_once_with("Mordechai")
    lastfm.get_similar_artists.assert_awaited_once_with("Khruangbin", limit=15)
    assert downloaded(mocks) == [[("Cut Worms", "Hollow Ground")]]
    assert result.reply.startswith("Added 1 album like Mordechai:")


@pytest.mark.asyncio
async def test_album_seed_in_same_artist_mode_excludes_the_seed_album():
    lastfm = fake_lastfm(
        found_album=LastFMAlbum(name="Mordechai", artist="Khruangbin"),
    )
    result, mocks = await run(
        "more by the Mordechai band", count=2,
        seed=seed_decision("Mordechai", "album", same=0.9), lastfm=lastfm,
        fit=lambda s: fit_decision(s, {}),
    )
    assert downloaded(mocks) == [[("Khruangbin", "Con Todo El Mundo")]]
    assert "by Khruangbin" in result.reply.splitlines()[0]


@pytest.mark.asyncio
async def test_unknown_album_is_deferred():
    result, _ = await run(seed=seed_decision("Nope", "album"))
    assert result is None


@pytest.mark.asyncio
async def test_same_artist_mode_allows_many_albums_by_one_artist():
    lastfm = fake_lastfm(albums={"Khruangbin": ["Mordechai", "Con Todo El Mundo", "Hasta El Cielo"]})
    result, mocks = await run(
        "2 more albums by Khruangbin", count=3, seed=seed_decision(same=0.9), lastfm=lastfm,
        fit=lambda s: fit_decision(s, {}),
    )
    lastfm.get_similar_artists.assert_not_awaited()
    assert [a for a, _ in downloaded(mocks)[0]] == ["Khruangbin"] * 3
    assert result.reply.startswith("Added 3 albums by Khruangbin:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed",
    [
        seed_decision(p=0.3),
        seed_decision("none"),
        seed_decision("Khruangbin", "none"),
    ],
)
async def test_unsure_seed_defers_to_the_agent(seed):
    result, mocks = await run(seed=seed)
    assert result is None
    mocks.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_decision_defers():
    with patch("bot.workflows.decide", AsyncMock(return_value=None)), patch(
        "bot.workflows.tag_vocabulary", AsyncMock(return_value=[])
    ):
        assert await recommend("5 albums like X", _ctx()) is None


@pytest.mark.asyncio
async def test_no_candidates_defers():
    result, _ = await run(lastfm=fake_lastfm(similar={}))
    assert result is None


@pytest.mark.asyncio
async def test_fit_decision_none_keeps_lastfm_order():
    async def fake_decide(name, state, questions, run_id=None):
        return seed_decision() if name == RECOMMEND_SEED_NAME else None

    download = AsyncMock(side_effect=lambda refs, ctx, media=None: [added_line(r) for r in refs])
    with patch("bot.workflows.decide", AsyncMock(side_effect=fake_decide)), patch(
        "bot.workflows.LastFMClient", return_value=fake_lastfm()
    ), patch("bot.workflows.tag_vocabulary", AsyncMock(return_value=[])), patch(
        "bot.workflows._library_lookup",
        AsyncMock(side_effect=lambda artists: {a: [] for a in artists}),
    ), patch("bot.workflows.download_many", download):
        result = await recommend("2 albums like Khruangbin", _ctx(), count=2)
    assert [(r.artist, r.title) for r in download.await_args.args[0]] == [
        ("Hiatus Kaiyote", "Choose Your Weapon"), ("Y Loop", "Y Loop One"),
    ]
    assert result.reply.startswith("Added 2 albums")
    assert any(s.result == "no ranking; Last.fm order" for s in result.steps)


@pytest.mark.asyncio
async def test_poor_fits_are_dropped_when_enough_fit():
    # Only Superorganism is below 1.0, and it is also the last candidate: never chosen.
    result, mocks = await run(count=4)
    assert ("Superorganism", "Superorganism") not in downloaded(mocks)[0]
    assert len(downloaded(mocks)[0]) == 3  # Superorganism excluded, so only 3 artists fit


@pytest.mark.asyncio
async def test_more_like_that_offers_spans_from_the_conversation():
    earlier = [
        {"role": "user", "content": "get me Mordechai"},
        {"role": "assistant", "content": "Added Mordechai."},
    ]
    _, mocks = await run(
        "more like that", seed=seed_decision("Mordechai", "album"), earlier=earlier,
        lastfm=fake_lastfm(found_album=LastFMAlbum(name="Mordechai", artist="Khruangbin")),
    )
    questions = mocks.decide.await_args_list[0].args[2]
    assert "Mordechai" in questions["seed"]["criteria"]
    assert "none" in questions["seed"]["criteria"]
    assert mocks.decide.await_args_list[0].args[1]["earlier"][-1]["from"] == "bot"
    assert "earlier" in mocks.decide.await_args_list[1].args[1]


@pytest.mark.asyncio
async def test_options_include_vocabulary_tags_and_are_capped():
    _, mocks = await run("something chill to work to", seed=seed_decision("none"), vocab=("chill", "chillout"))
    options = list(mocks.decide.await_args_list[0].args[2]["seed"]["criteria"])
    assert "chill" in options and "chillout" in options
    long_text = " ".join(f"word{i}" for i in range(400))
    _, mocks = await run(long_text, seed=seed_decision("none"))
    assert len(mocks.decide.await_args_list[0].args[2]["seed"]["criteria"]) <= 251


# -- turn.run -----------------------------------------------------------------


class RaisingModel:
    name = "fake"

    async def chat(self, messages, tools):
        raise AssertionError("the model must not be called")


def _turn(kind="open_ended", count=None, p=0.99):
    return turn.Turn(
        tools=[],
        messages=[system("sys"), user("5 albums like Khruangbin")],
        route=Route("music", p, kind, p, count),
        run_id="r1",
        text="5 albums like Khruangbin",
        earlier=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.asyncio
async def test_turn_uses_recommend_for_open_ended(monkeypatch):
    monkeypatch.setattr(turn, "get_model", lambda: RaisingModel())
    events = []
    wf = WorkflowResult("Added 5 albums", [])
    with patch("bot.turn.workflows.recommend", AsyncMock(return_value=wf)) as m:
        result = await turn.run(_turn(count=5), _ctx(), on_event=lambda e, d: events.append((e, d)))
    assert m.await_args.args[0] == "5 albums like Khruangbin"
    assert m.await_args.kwargs["count"] == 5
    assert m.await_args.kwargs["earlier"] == [{"role": "user", "content": "hi"}]
    assert m.await_args.kwargs["run_id"] == "r1"
    assert result.stopped == "workflow"
    assert events == [("reply", {"content": "Added 5 albums"})]


@pytest.mark.asyncio
async def test_turn_recommend_defaults_to_three(monkeypatch):
    monkeypatch.setattr(turn, "get_model", lambda: RaisingModel())
    with patch("bot.turn.workflows.recommend", AsyncMock(return_value=WorkflowResult("x", []))) as m:
        await turn.run(_turn(), _ctx())
    assert m.await_args.kwargs["count"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, RuntimeError("boom")])
async def test_turn_falls_back_to_the_agent_from_recommend(monkeypatch, outcome):
    from bot.llm import Message

    class Model:
        name = "fake"

        async def chat(self, messages, tools):
            return Message("assistant", "agent reply")

    monkeypatch.setattr(turn, "get_model", lambda: Model())
    mock = AsyncMock(side_effect=outcome) if outcome else AsyncMock(return_value=None)
    with patch("bot.turn.workflows.recommend", mock):
        result = await turn.run(_turn(), _ctx())
    assert result.messages[-1].content == "agent reply"


@pytest.mark.asyncio
async def test_turn_does_not_recommend_for_untrusted_routes(monkeypatch):
    from bot.llm import Message

    class Model:
        name = "fake"

        async def chat(self, messages, tools):
            return Message("assistant", "agent reply")

    monkeypatch.setattr(turn, "get_model", lambda: Model())
    with patch("bot.turn.workflows.recommend", AsyncMock()) as m:
        await turn.run(_turn(p=0.3), _ctx())
        await turn.run(_turn(kind="specific"), _ctx())
    m.assert_not_called()


def test_name_spans_drop_phrases_of_the_request():
    from bot.workflows import _name_spans

    spans = _name_spans("find me a couple albums like Mordechai but heavier")
    assert "Mordechai" in spans
    assert not any("but" in s.split() or "like" in s.split() for s in spans)
    assert "Band of Horses" in _name_spans("5 albums like Band of Horses")
