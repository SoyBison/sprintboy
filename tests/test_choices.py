from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot import turn
from bot.choices import NONE_LABEL, DidYouMeanView
from bot.tools import AlbumRef
from bot.workflows import Choice, Pending, describe_lines

NIGHT = Choice("Weather Report - Night Passage", "Weather Report", "Night Passage")
HEAVY = Choice("Weather Report - Heavy Weather", "Weather Report", "Heavy Weather")


def interaction(user_id):
    i = MagicMock()
    i.user.id = user_id
    i.response.edit_message = AsyncMock()
    i.response.send_message = AsyncMock()
    return i


def make_view(on_pick=None):
    on_pick = on_pick or AsyncMock()
    return DidYouMeanView(Pending("night passage", [NIGHT, HEAVY]), 42, on_pick), on_pick


async def press(view, index, user_id=42):
    i = interaction(user_id)
    await view.children[index].callback(i)
    return i


@pytest.mark.asyncio
async def test_one_button_per_option_plus_none():
    view, _ = make_view()
    assert len(view.children) == 3
    assert [b.label for b in view.children] == [NIGHT.label, HEAVY.label, NONE_LABEL]


@pytest.mark.asyncio
async def test_long_labels_are_truncated():
    long = Choice("x" * 200, "a", "b")
    view = DidYouMeanView(Pending("w", [long, NIGHT]), 1, AsyncMock())
    assert len(view.children[0].label) == 80


@pytest.mark.asyncio
async def test_other_users_are_refused():
    view, on_pick = make_view()
    i = await press(view, 0, user_id=7)
    i.response.send_message.assert_awaited_once_with("Not your request.", ephemeral=True)
    i.response.edit_message.assert_not_awaited()
    on_pick.assert_not_awaited()
    assert not any(b.disabled for b in view.children)


@pytest.mark.asyncio
async def test_author_pick_calls_back_and_disables_the_buttons():
    view, on_pick = make_view()
    i = await press(view, 1)
    on_pick.assert_awaited_once_with(i, HEAVY)
    i.response.edit_message.assert_awaited_once_with(view=view)
    assert all(b.disabled for b in view.children)


@pytest.mark.asyncio
async def test_none_of_these_passes_none():
    view, on_pick = make_view()
    i = await press(view, 2)
    on_pick.assert_awaited_once_with(i, None)


@pytest.mark.asyncio
async def test_timeout_removes_the_buttons():
    view, _ = make_view()
    view.message = MagicMock()
    view.message.edit = AsyncMock()
    await view.on_timeout()
    view.message.edit.assert_awaited_once_with(view=None)


def test_describe_lines_wording():
    refs = [AlbumRef(artist="A", title="One"), AlbumRef(artist="B", title="Two"),
            AlbumRef(artist="C", title="Three"), AlbumRef(artist="D", title="Four")]
    lines = [
        "Added 'A - One [2020] [Album] FLAC / Lossless / WEB', confirmed present in qBittorrent.",
        "OWNED: 'B - Two' is already in the library; skipped.",
        "NOT FOUND: no FLAC torrent for 'C - Three'. Closest results: x",
        "NOT ADDED: QBITTORRENT_DRY_RUN is on, so 'D - Four [2020]' was never submitted. Tell the user nothing was downloaded.",
    ]
    assert describe_lines(lines, refs).splitlines() == [
        "Added A - One.",
        "You already have B - Two.",
        "Couldn't find a FLAC of C - Three on the tracker.",
        "Couldn't add D - Four: dry run is on",
    ]


def test_describe_lines_without_refs_reads_the_lines():
    lines = [
        "Added 'A - One [2020] [Album] FLAC / Lossless / WEB', confirmed present in qBittorrent.",
        "Added 'B - Two [2021] [Album] FLAC / Lossless / WEB', confirmed present in qBittorrent.",
        "OWNED: 'C - Three' is already in the library; skipped.",
        "NOT ADDED: only a vinyl rip of 'D - Four' exists (x). Ask for vinyl to get it.",
    ]
    assert describe_lines(lines).splitlines() == [
        "Added:",
        "- A - One",
        "- B - Two",
        "You already have C - Three.",
        "Couldn't add D - Four: only a vinyl rip of 'D - Four' exists (x). Ask for vinyl to get it.",
    ]


@pytest.mark.asyncio
async def test_run_can_skip_workflows():
    from bot.routing import Route

    t = turn.Turn(
        tools=[], messages=[], route=Route("music", 0.9, "specific", 0.9, None), run_id="r", text="get X"
    )
    agent_result = MagicMock(messages=[], steps=[], stopped="reply")
    with patch("bot.turn.workflows.specific", AsyncMock()) as wf, patch(
        "bot.turn.get_model", lambda: MagicMock()
    ), patch("bot.turn.run_agent", AsyncMock(return_value=agent_result)) as agent, patch(
        "bot.turn.add_nudge", lambda *a: None
    ):
        context = MagicMock(internal_torrents={})
        await turn.run(t, context, use_workflows=False)
        wf.assert_not_awaited()
        agent.assert_awaited_once()
        wf.return_value = None
        await turn.run(t, context)
        wf.assert_awaited_once()
