"""
One JSON line per agent turn: what was asked, how it was routed, every model
and tool step, and what was replied. Read it back to see why a run went wrong.
"""

import dataclasses
import json
import logging
import time
from pathlib import Path

from bot.agent import AgentResult
from bot.config import Config, git_sha

logger = logging.getLogger(__name__)

RESULT_LIMIT = 4000


def _clip(text: str | None, limit: int = RESULT_LIMIT) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + "..."


def record_run(
    *,
    run_id: str | None,
    message_id,
    conversation_id,
    author,
    text: str,
    route,
    result: AgentResult,
    new_torrents: list[str],
    reply: str,
    note: str,
    seconds: float,
) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_sha(),
        "run_id": run_id,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "author": str(author),
        "text": text,
        "route": dataclasses.asdict(route) if route is not None else None,
        "nudged": result.nudged,
        "stopped": result.stopped,
        "steps": [
            {
                "kind": step.kind,
                "name": step.name,
                "seconds": round(step.seconds, 3),
                "args": step.args,
                "result": _clip(step.result),
                "error": step.error,
                "usage": step.usage,
            }
            for step in result.steps
        ],
        "new_torrents": list(new_torrents),
        "reply": reply,
        "note": note,
        "seconds": round(seconds, 3),
    }
    try:
        path = Path(Config.RUN_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning(f"Could not write the run log {Config.RUN_LOG_PATH}: {e}")
