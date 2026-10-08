from unittest.mock import AsyncMock, patch

import pytest
from types import SimpleNamespace

from pydantic import BaseModel

from bot import turn
from bot.llm import Message, system, user
from bot.output import MUSIC_NUDGE
from bot.toolkit import tool
from bot.decide import current_run_id
from bot.routing import Route


def r(domain, kind, p=0.99):
    return Route(domain, p, kind, p, 3)


def names(tools):
    return {t.name for t in tools}


def test_none_and_untrusted_get_all_tools():
    assert turn.select_tools(None) is turn.AGENT_TOOLS
    assert turn.select_tools(r("music", "open_ended", p=0.1)) is turn.AGENT_TOOLS


def test_music_open_ended():
    assert turn.select_tools(r("music", "open_ended")) == turn.MUSIC_READ + turn.MUSIC_DOWNLOAD


def test_music_question_has_no_download():
    tools = turn.select_tools(r("music", "question"))
    assert tools == turn.MUSIC_READ
    assert "add_torrent" not in names(tools)


def test_movie_specific():
    assert turn.select_tools(r("movie", "specific")) == turn.MOVIE_READ + turn.DOWNLOAD


def test_tv_question():
    assert turn.select_tools(r("tv", "question")) == turn.MOVIE_READ


def test_chat_question_and_open_ended():
    assert turn.select_tools(r("chat", "question")) == turn.MUSIC_READ + turn.MOVIE_READ
    assert turn.select_tools(r("chat", "open_ended")) is turn.AGENT_TOOLS


HISTORY = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "five albums like Slowdive"},
]


def test_latest_user_text():
    assert turn.latest_user_text(HISTORY) == "five albums like Slowdive"
    assert turn.latest_user_text([]) == ""


@pytest.mark.asyncio
async def test_prepare_appends_note():
    decided = r("music", "open_ended")
    with patch("bot.turn.route", AsyncMock(return_value=decided)) as m:
        t = await turn.prepare(HISTORY)
    assert t.route is decided
    assert m.await_args.args[0] == "five albums like Slowdive"
    assert t.messages[0].role == "system"
    assert t.messages[0].content == turn.SYSTEM_PROMPT + "\n\n" + decided.note()
    assert [(m.role, m.content) for m in t.messages[1:]] == [
        (h["role"], h["content"]) for h in HISTORY
    ]
    assert all(isinstance(m, Message) for m in t.messages)
    assert t.tools == turn.MUSIC_READ + turn.MUSIC_DOWNLOAD
    assert current_run_id.get() == t.run_id


@pytest.mark.asyncio
async def test_prepare_without_route():
    with patch("bot.turn.route", AsyncMock(return_value=None)):
        t = await turn.prepare(HISTORY)
    assert t.route is None
    assert t.messages[0].content == turn.SYSTEM_PROMPT
    assert t.tools is turn.AGENT_TOOLS


class FakeModel:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, tools):
        self.calls.append(list(messages))
        return self.replies.pop(0)


class NoArgs(BaseModel):
    pass


@tool(args_schema=NoArgs)
async def download_albums(runtime) -> str:
    """Stub."""
    runtime.context.internal_torrents["Album"] = "hash"
    return "Added"


def make_turn(tools):
    return turn.Turn(
        tools=tools,
        messages=[system("sys"), user("five albums like Slowdive")],
        route=r("music", "open_ended"),
        run_id="r1",
    )


def context():
    return SimpleNamespace(internal_torrents={})


@pytest.mark.asyncio
async def test_run_nudges_once_when_nothing_was_added(monkeypatch):
    model = FakeModel([Message("assistant", "Added: stuff"), Message("assistant", "Sorry")])
    monkeypatch.setattr(turn, "get_model", lambda: model)
    result = await turn.run(make_turn([download_albums]), context())
    assert result.nudged
    assert len(model.calls) == 2
    nudge = model.calls[1][-1]
    assert (nudge.role, nudge.content) == ("user", MUSIC_NUDGE)
    assert result.messages[-1].content == "Sorry"
    assert [s.kind for s in result.steps] == ["model", "model"]


@pytest.mark.asyncio
async def test_run_does_not_nudge_after_an_add(monkeypatch):
    call = {"id": "c1", "name": "download_albums", "args": {}}
    model = FakeModel(
        [Message("assistant", "", tool_calls=[call]), Message("assistant", "Added Album")]
    )
    monkeypatch.setattr(turn, "get_model", lambda: model)
    ctx = context()
    result = await turn.run(make_turn([download_albums]), ctx)
    assert not result.nudged
    assert len(model.calls) == 2
    assert ctx.internal_torrents == {"Album": "hash"}
    assert result.messages[-1].content == "Added Album"


@pytest.mark.asyncio
async def test_decide_retries_a_jev_503_once(monkeypatch):
    from unittest.mock import AsyncMock

    from bot import decide as d
    from bot.config import Config

    monkeypatch.setattr(Config, "DECISION_PRIMARY", "jev")
    monkeypatch.setattr(Config, "TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(Config, "DECISION_LOG_PATH", "/dev/null")
    ok = d.Decision("jev", "jev-1", {"q": {"type": "noul", "noul": 0.9}}, 0.1)
    ask = AsyncMock(side_effect=[d.DecisionError("jev 503: unavailable"), ok])
    monkeypatch.setattr(d, "_ask", ask)
    monkeypatch.setattr(d.asyncio, "sleep", AsyncMock())
    result = await d.decide("t/v1", {"x": 1}, {"q": {"type": "noul", "instructions": "?"}})
    assert result is ok and ask.await_count == 2
