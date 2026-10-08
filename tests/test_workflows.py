from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from bot import turn, workflows
from bot.decide import Decision
from bot.llm import system, user
from bot.netcode import BTCategory, CanonicalRelease, MBReleaseGroup, SearchResult
from bot.routing import Route
from bot.tools import TorrentContext
from bot.workflows import WorkflowResult, candidate_spans, discography

MORDECHAI = "Khruangbin - Mordechai [2020] [Album] FLAC / Lossless / WEB [Orpheus]"
CON_TODO_WEB = "Khruangbin - Con todo el mundo [2018] [Album] FLAC / 24bit Lossless / WEB [Orpheus]"
CON_TODO_VINYL = "Khruangbin - Con todo el mundo [2018] [Album] FLAC / 24bit Lossless / Vinyl [Orpheus]"
TEXAS_MOON = "Khruangbin and Leon Bridges - Texas Moon [2022] [EP] FLAC / Lossless / WEB [Orpheus]"
LIVE_EP = "Khruangbin - Late Night Tales [2021] [EP] FLAC / Lossless / WEB [Orpheus]"
SINGLE = "Khruangbin - August 10 [2020] [Single] FLAC / Lossless / WEB [Orpheus]"
NEWEST = "Khruangbin - A La Sala [2024] [Album] FLAC / 24bit Lossless / WEB [Orpheus]"
MAYBE = "Khruangbin - Hasta el Cielo [2019] [Album] FLAC / Lossless / WEB [Orpheus]"


def _sr(name):
    return SearchResult(
        fileName=name, fileUrl=f"http://j/{name}", fileSize=1, nbSeeders=1,
        nbLeechers=0, siteUrl="http://t", descrLink="http://t/1",
    )


def _decision(artist="Khruangbin", artist_p=0.99, scope="albums", scope_p=0.99):
    return Decision(
        "jev", "m",
        {
            "artist": {"choice": artist, "probabilities": {artist: artist_p}},
            "scope": {"choice": scope, "probabilities": {scope: scope_p}},
        },
        0.1,
    )


def _ctx():
    return TorrentContext(search_results={}, internal_torrents={}, torrent_types=set())


async def _add(name, category, context, corrected_name=None):
    context.internal_torrents[corrected_name] = "code"
    return f"Added '{corrected_name}', confirmed present in qBittorrent."


async def _run(
    names,
    *,
    owned=("Mordechai",),
    decision=None,
    text="get me the rest of the Khruangbin discography",
    same=None,
    add=None,
    ctx=None,
    groups=None,
    **kwargs,
):
    ctx = ctx or _ctx()
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.search = AsyncMock(return_value=SimpleNamespace(results=[_sr(n) for n in names]))
    canon = CanonicalRelease(
        artist="Khruangbin", artist_candidates=["Khruangbin"], album=None, album_candidates=[]
    )
    lookup = {"Khruangbin": None if owned is None else list(owned)}
    mocks = SimpleNamespace(
        add=add or AsyncMock(side_effect=_add),
        same=same or AsyncMock(return_value=None),
        client=client,
        catalogue=AsyncMock(return_value=groups),
    )
    with patch("bot.workflows.decide", AsyncMock(return_value=decision or _decision())), patch(
        "bot.workflows._canonicalise", AsyncMock(return_value=canon)
    ), patch("bot.workflows.QBittorrentClient", return_value=client), patch(
        "bot.workflows._library_lookup", AsyncMock(return_value=lookup)
    ), patch("bot.workflows._same_release", mocks.same), patch(
        "bot.workflows._add_one", mocks.add
    ), patch("bot.workflows._musicbrainz_catalogue", mocks.catalogue):
        result = await discography(text, ctx, **kwargs)
    return result, ctx, mocks


def _added_names(mocks):
    return sorted(call.args[0] for call in mocks.add.await_args_list)


def test_candidate_spans():
    spans = candidate_spans("get me the rest of the Khruangbin discography")
    assert "Khruangbin" in spans
    assert not any(s.casefold() in ("the rest", "rest", "the rest of") for s in spans)
    assert all(s.split()[0].casefold() != "the" for s in spans)
    spans = candidate_spans("Kendrick's GNX")
    assert "Kendrick" in spans and "GNX" in spans
    assert "Boards of Canada" in candidate_spans("grab everything by Boards of Canada")
    assert candidate_spans("get me the") == []
    assert len(candidate_spans(" ".join(f"w{i}" for i in range(400)))) <= 250
    dup = candidate_spans("Khruangbin khruangbin")
    assert len([s for s in dup if s.casefold() == "khruangbin"]) == 1


@pytest.mark.asyncio
async def test_albums_scope_adds_only_the_missing_album():
    names = [MORDECHAI, CON_TODO_VINYL, CON_TODO_WEB, TEXAS_MOON, SINGLE, "junk FLAC thing"]
    result, ctx, mocks = await _run(names)
    assert _added_names(mocks) == [CON_TODO_WEB]
    assert BTCategory.Music in ctx.torrent_types
    assert CON_TODO_VINYL in ctx.search_results
    assert "Added 1 album by Khruangbin:" in result.reply
    assert "- Con todo el mundo (2018, WEB, 24bit)" in result.reply
    assert "You already have: Mordechai" in result.reply
    assert "Texas Moon" not in result.reply and "August" not in result.reply
    mocks.client.search.assert_awaited_once_with("Khruangbin", BTCategory.Music)
    assert [s.name for s in result.steps][0] == "discography/decide"
    assert {s.kind for s in result.steps} == {"tool"}


@pytest.mark.asyncio
@pytest.mark.parametrize("p, added", [(0.9, False), (0.5, False), (0.1, True)])
async def test_maybe_owned_is_added_only_when_clearly_different(p, added):
    result, _, mocks = await _run(
        [MAYBE], owned=("Hasta el Cielo (Deluxe Edition)",), same=AsyncMock(return_value=p)
    )
    assert (_added_names(mocks) == [MAYBE]) is added
    if not added:
        assert "Skipped because you may already have them: Hasta el Cielo (as 'Hasta el Cielo (Deluxe Edition)')" in result.reply
    mocks.same.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_owned_with_no_judgement_is_skipped():
    _, _, mocks = await _run([MAYBE], owned=("Hasta el Cielo (Deluxe Edition)",))
    mocks.add.assert_not_called()


@pytest.mark.asyncio
async def test_everything_scope_includes_ep_and_collaboration():
    names = [MORDECHAI, TEXAS_MOON, LIVE_EP, SINGLE, CON_TODO_WEB]
    result, _, mocks = await _run(names, decision=_decision(scope="everything"))
    assert _added_names(mocks) == sorted([TEXAS_MOON, LIVE_EP, SINGLE, CON_TODO_WEB])
    assert "Added 4 releases by Khruangbin:" in result.reply


@pytest.mark.asyncio
async def test_newest_scope_adds_only_the_latest():
    names = [MORDECHAI, CON_TODO_WEB, NEWEST, TEXAS_MOON]
    result, _, mocks = await _run(names, owned=(), decision=_decision(scope="newest"))
    assert _added_names(mocks) == [NEWEST]
    assert "Added 1 newest release by Khruangbin:" in result.reply


@pytest.mark.asyncio
async def test_low_scope_confidence_defaults_to_albums():
    _, _, mocks = await _run(
        [CON_TODO_WEB, TEXAS_MOON], decision=_decision(scope="everything", scope_p=0.3)
    )
    assert _added_names(mocks) == [CON_TODO_WEB]


@pytest.mark.asyncio
async def test_defers_when_unsure():
    result, _, mocks = await _run([CON_TODO_WEB], decision=_decision(artist_p=0.4))
    assert result is None
    result, _, _ = await _run([CON_TODO_WEB], decision=_decision(artist="none"))
    assert result is None
    with patch("bot.workflows.decide", AsyncMock(return_value=None)):
        assert await discography("get me the Khruangbin discography", _ctx()) is None
    result, _, mocks = await _run([])
    assert result is None
    mocks.add.assert_not_called()


@pytest.mark.asyncio
async def test_library_failure_adds_nothing():
    result, _, mocks = await _run([CON_TODO_WEB], owned=None)
    mocks.add.assert_not_called()
    assert "couldn't check your library" in result.reply
    assert "didn't add anything" in result.reply


@pytest.mark.asyncio
async def test_everything_owned():
    result, _, mocks = await _run([MORDECHAI])
    mocks.add.assert_not_called()
    assert result.reply.startswith("You already have every album by Khruangbin")
    assert "You already have: Mordechai" in result.reply


@pytest.mark.asyncio
async def test_max_add_truncation():
    names = [
        f"Khruangbin - Album {n} [{2000 + i}] [Album] FLAC / Lossless / WEB [Orpheus]"
        for i, n in enumerate(["Alpha", "Bravo", "Charlie", "Delta", "Echo"])
    ]
    result, _, mocks = await _run(names, owned=(), max_add=3)
    assert len(mocks.add.await_args_list) == 3
    assert "Album Alpha" in result.reply and "Album Delta" not in result.reply
    assert "2 more not added yet; ask again to get them." in result.reply


@pytest.mark.asyncio
async def test_failed_adds_are_listed():
    async def add(name, category, context, corrected_name=None):
        return "NOT ADDED: QBITTORRENT_DRY_RUN is on, so 'x' was never submitted."

    result, _, _ = await _run([CON_TODO_WEB], add=AsyncMock(side_effect=add))
    assert "Couldn't add:\n- Con todo el mundo - dry run is on" in result.reply
    assert "Added" not in result.reply.replace("Couldn't add", "")


@pytest.mark.asyncio
async def test_duplicate_versions_of_one_album_add_one():
    other = "Khruangbin - Con todo el mundo [2019] [EP] FLAC / Lossless / CD [Orpheus]"
    _, _, mocks = await _run([CON_TODO_WEB, other], decision=_decision(scope="everything"))
    assert _added_names(mocks) == [CON_TODO_WEB]


def _group(title, year, primary="Album", secondary=()):
    return MBReleaseGroup(
        mbid=f"mb-{title}", title=title, primary_type=primary,
        secondary_types=list(secondary), year=year,
    )


KHRUANGBIN_MB = [
    _group("The Universe Smiles Upon You", 2015),
    _group("Con todo el mundo", 2018),
    _group("Hasta El Cielo (Con Todo El Mundo in Dub)", 2019, secondary=["Remix"]),
    _group("Mordechai", 2020),
    _group("Live at Stubb's", 2023, secondary=["Live"]),
    _group("A LA SALA", 2024),
    _group("Texas Moon", 2022, primary="EP"),
]
LIVE_AS_ALBUM = "Khruangbin - Live at Stubb's [2023] [Album] FLAC / Lossless / WEB [Orpheus]"


@pytest.mark.asyncio
async def test_musicbrainz_drops_albums_that_are_not_studio_albums():
    names = [MORDECHAI, CON_TODO_WEB, MAYBE, LIVE_AS_ALBUM, NEWEST]
    result, _, mocks = await _run(names, groups=KHRUANGBIN_MB)
    assert _added_names(mocks) == sorted([CON_TODO_WEB, NEWEST])
    assert (
        "Left out because they aren't studio albums: Hasta el Cielo (remix album), "
        "Live at Stubb's (live album)" in result.reply
    )
    mocks.catalogue.assert_awaited_once_with("Khruangbin", None)
    step = next(s for s in result.steps if s.name == "discography/musicbrainz")
    assert step.result == "4 studio albums of 7 groups"


@pytest.mark.asyncio
async def test_musicbrainz_lists_studio_albums_with_no_torrent():
    result, _, _ = await _run([MORDECHAI, CON_TODO_WEB], groups=KHRUANGBIN_MB)
    # Mordechai is owned, Texas Moon is an EP, and the rest were on the tracker.
    assert "No torrent found for: The Universe Smiles Upon You (2015), A LA SALA (2024)" in result.reply


@pytest.mark.asyncio
async def test_musicbrainz_keeps_titles_it_does_not_know():
    unknown = "Khruangbin - Something New [2026] [Album] FLAC / Lossless / WEB [Orpheus]"
    result, _, mocks = await _run([unknown, LIVE_AS_ALBUM], groups=KHRUANGBIN_MB)
    assert _added_names(mocks) == [unknown]


@pytest.mark.asyncio
async def test_studio_album_sharing_a_title_with_a_live_one_is_kept():
    groups = [_group("Mordechai", 2020), _group("Mordechai", 2021, secondary=["Live"])]
    _, _, mocks = await _run([MORDECHAI], owned=(), groups=groups)
    assert _added_names(mocks) == [MORDECHAI]


@pytest.mark.asyncio
async def test_without_musicbrainz_the_workflow_is_unchanged():
    result, _, mocks = await _run([MORDECHAI, CON_TODO_WEB, MAYBE], groups=None)
    assert _added_names(mocks) == sorted([CON_TODO_WEB, MAYBE])
    assert "studio albums" not in result.reply and "No torrent found" not in result.reply
    step = next(s for s in result.steps if s.name == "discography/musicbrainz")
    assert step.result == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["everything", "newest"])
async def test_musicbrainz_is_only_asked_for_the_albums_scope(scope):
    _, _, mocks = await _run([NEWEST], owned=(), decision=_decision(scope=scope), groups=KHRUANGBIN_MB)
    mocks.catalogue.assert_not_called()


def test_to_agent_result():
    msgs = [system("s"), user("u")]
    result = WorkflowResult("hi", []).to_agent_result(msgs)
    assert result.stopped == "workflow"
    assert [m.content for m in result.messages] == ["s", "u", "hi"]
    assert len(msgs) == 2


class RaisingModel:
    name = "fake"

    async def chat(self, messages, tools):
        raise AssertionError("the model must not be called")


def _turn(kind="discography", p=0.99, domain="music"):
    return turn.Turn(
        tools=[],
        messages=[system("sys"), user("get me the rest of the Khruangbin discography")],
        route=Route(domain, p, kind, p, None),
        run_id="r1",
        text="get me the rest of the Khruangbin discography",
    )


@pytest.mark.asyncio
async def test_turn_uses_workflow(monkeypatch):
    monkeypatch.setattr(turn, "get_model", lambda: RaisingModel())
    wf = WorkflowResult("Added 1 album", [])
    events = []
    with patch("bot.turn.workflows.discography", AsyncMock(return_value=wf)) as m:
        result = await turn.run(_turn(), _ctx(), on_event=lambda e, d: events.append((e, d)))
    assert m.await_args.args[0] == "get me the rest of the Khruangbin discography"
    assert m.await_args.kwargs["run_id"] == "r1"
    assert result.stopped == "workflow"
    assert result.messages[-1].content == "Added 1 album"
    assert events == [("reply", {"content": "Added 1 album"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, RuntimeError("boom")])
async def test_turn_falls_back_to_the_agent(monkeypatch, outcome):
    from bot.llm import Message

    class Model:
        name = "fake"
        calls = 0

        async def chat(self, messages, tools):
            Model.calls += 1
            return Message("assistant", "agent reply")

    monkeypatch.setattr(turn, "get_model", lambda: Model())
    mock = AsyncMock(side_effect=outcome) if outcome else AsyncMock(return_value=None)
    with patch("bot.turn.workflows.discography", mock):
        result = await turn.run(_turn(), _ctx())
    assert Model.calls >= 1
    assert result.messages[-1].content == "agent reply"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"kind": "specific"}, {"p": 0.3}, {"domain": "movie"}]
)
async def test_turn_skips_workflow_for_other_routes(monkeypatch, kwargs):
    from bot.llm import Message

    class Model:
        name = "fake"

        async def chat(self, messages, tools):
            return Message("assistant", "agent reply")

    monkeypatch.setattr(turn, "get_model", lambda: Model())
    with patch("bot.turn.workflows.discography", AsyncMock()) as m:
        await turn.run(_turn(**kwargs), _ctx())
    m.assert_not_called()
