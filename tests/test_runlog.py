import json

from bot import runlog
from bot.agent import AgentResult, Step
from bot.config import Config
from bot.routing import Route


def test_record_run_writes_one_json_line(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "runs.jsonl"
    monkeypatch.setattr(Config, "RUN_LOG_PATH", str(path))
    result = AgentResult(
        messages=[],
        steps=[
            Step("model", "fake", 1.23456, result="", usage={"input_tokens": 5}),
            Step("tool", "check_albums", 0.5, args={"a": 1}, result="x" * 10_000),
        ],
        stopped="reply",
        nudged=True,
    )
    for _ in range(2):
        runlog.record_run(
            run_id="r1",
            message_id=1,
            conversation_id=2,
            author=object(),
            text="hello",
            route=Route("music", 0.9, "open_ended", 0.9, 3),
            result=result,
            new_torrents=["Album"],
            reply="done",
            note="",
            seconds=2.5,
        )
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["run_id"] == "r1" and entry["nudged"] is True
    assert entry["route"]["domain"] == "music"
    assert entry["new_torrents"] == ["Album"]
    assert entry["steps"][0]["seconds"] == 1.235
    tool_result = entry["steps"][1]["result"]
    assert len(tool_result) < 4100 and tool_result.endswith("...")
    assert isinstance(entry["author"], str)


def test_record_run_without_route_and_unwritable_path(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(Config, "RUN_LOG_PATH", str(blocker / "runs.jsonl"))
    runlog.record_run(
        run_id=None, message_id=None, conversation_id=None, author="cli", text="t",
        route=None, result=AgentResult([]), new_torrents=[], reply="", note="", seconds=0,
    )  # logs a warning instead of raising
