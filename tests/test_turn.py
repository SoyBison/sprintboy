from unittest.mock import AsyncMock, patch

import pytest

from bot import turn
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
    sentinel = object()
    with patch("bot.turn.route", AsyncMock(return_value=decided)) as m, patch(
        "bot.turn.get_agent", return_value=sentinel
    ):
        agent, messages, got, run_id = await turn.prepare(HISTORY)
    assert agent is sentinel and got is decided
    assert m.await_args.args[0] == "five albums like Slowdive"
    assert messages[0]["content"] == turn.SYSTEM_PROMPT + "\n\n" + decided.note()
    assert messages[1:] == HISTORY
    assert current_run_id.get() == run_id


@pytest.mark.asyncio
async def test_prepare_without_route():
    with patch("bot.turn.route", AsyncMock(return_value=None)), patch(
        "bot.turn.get_agent", return_value=object()
    ):
        _, messages, got, _ = await turn.prepare(HISTORY)
    assert got is None
    assert messages[0]["content"] == turn.SYSTEM_PROMPT


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
