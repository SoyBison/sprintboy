"""
Typed decisions from System One models: hosted Jev, or Laya via ollaya.

Both speak TypeSafe's /v1/systemone wire format, so a backend is just a URL, a
key and a model name. `decide()` asks the primary backend and returns its
answers; the shadow backend is asked the same thing in the background and only
logged. Every call appends one JSON line to Config.DECISION_LOG_PATH holding
the state, the questions and each backend's answers, which is the dataset for
comparing the two and, later, for distilling Jev into a local model.

A decision is never load-bearing: callers get None when the primary is off or
fails, and must fall back to what the bot did before decision models existed.
"""

import asyncio
import contextvars
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from bot.config import Config, git_sha

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    api_key: str
    model: str
    # Laya on the CPU takes ~4s to load after it has been idle; Jev is ~0.2s.
    timeout: float


def _backend(name: str) -> Backend | None:
    if name == "jev":
        if not Config.TYPESAFE_API_KEY:
            return None
        return Backend("jev", Config.TYPESAFE_URL, Config.TYPESAFE_API_KEY, Config.TYPESAFE_MODEL, 10.0)
    if name == "laya":
        if not Config.OLLAYA_URL:
            return None
        return Backend("laya", Config.OLLAYA_URL, Config.OLLAYA_API_KEY, Config.OLLAYA_DECISION_MODEL, 20.0)
    return None


class DecisionError(Exception):
    pass


@dataclass
class Decision:
    """Answers keyed by question id, from the backend that produced them."""

    backend: str
    model: str
    answers: dict[str, dict]
    latency: float

    def choice(self, qid: str) -> tuple[str, float]:
        """The chosen option and its probability."""
        answer = self.answers[qid]
        choice = answer["choice"]
        return choice, float(answer["probabilities"].get(choice, 0.0))

    def noul(self, qid: str) -> float:
        """Probability the answer to a yes/no question is yes."""
        return float(self.answers[qid]["noul"])


async def _ask(backend: Backend, state: Any, questions: dict) -> Decision:
    body = {"model": backend.model, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else {}
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{backend.url.rstrip('/')}/v1/systemone",
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=backend.timeout),
        ) as response:
            try:
                payload = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError):
                payload = {"error": await response.text()}
            # TypeSafe answers some auth failures with an error body, so the
            # status alone does not say whether there are answers.
            if response.status != 200 or "answers" not in payload:
                raise DecisionError(f"{backend.name} {response.status}: {payload}")
    return Decision(
        backend=backend.name,
        model=str(payload.get("model", backend.model)),
        answers=payload["answers"],
        latency=time.perf_counter() - start,
    )


def _record(entry: dict) -> None:
    path = Path(Config.DECISION_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as log:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"Could not write decision log {path}: {e}")


def _summary(result: Decision | BaseException | None) -> dict | None:
    if result is None:
        return None
    if isinstance(result, BaseException):
        return {"error": f"{type(result).__name__}: {result}"}
    return {"model": result.model, "answers": result.answers, "latency": round(result.latency, 3)}


# The agent turn a decision belongs to. Set once per message so tools deep in
# the call stack log under the right run without threading an id through.
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)

# Shadow calls still running, so they are not garbage collected mid-flight.
_background: set[asyncio.Task] = set()


async def decide(
    name: str, state: Any, questions: dict, run_id: str | None = None
) -> Decision | None:
    """Ask the primary backend, shadow-ask the other, log both.

    `name` identifies the question set in the log (e.g. "route/v1"); bump its
    version whenever the questions change so old and new answers are not mixed.
    """
    primary = _backend(Config.DECISION_PRIMARY)
    shadow = _backend(Config.DECISION_SHADOW)
    if shadow and primary and shadow.name == primary.name:
        shadow = None
    if primary is None and shadow is None:
        return None
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "ts": time.time(),
        "run_id": run_id or current_run_id.get(),
        "name": name,
        "commit": git_sha(),
        "state": state,
        "questions": questions,
    }

    shadow_task = (
        asyncio.create_task(_ask(shadow, state, questions)) if shadow else None
    )

    result: Decision | BaseException | None = None
    if primary:
        try:
            try:
                result = await _ask(primary, state, questions)
            except DecisionError as e:
                # Jev answers the odd 503 "model unavailable"; one quick retry
                # costs ~0.2s, losing the route costs the whole turn.
                if " 5" not in str(e)[: len(primary.name) + 3]:
                    raise
                logger.info(f"Decision {name} from {primary.name} got {e}; retrying once")
                await asyncio.sleep(0.5)
                result = await _ask(primary, state, questions)
        except (aiohttp.ClientError, asyncio.TimeoutError, DecisionError) as e:
            logger.warning(f"Decision {name} from {primary.name} failed: {e}")
            result = e
    entry["primary"] = primary.name if primary else None
    entry[primary.name if primary else "primary"] = _summary(result)

    if shadow_task is None:
        _record(entry)
    else:
        # The shadow never delays the reply: log once it lands, in the background.
        async def finish(task: asyncio.Task = shadow_task, backend: Backend = shadow):
            try:
                shadow_result: Decision | BaseException = await task
            except BaseException as e:  # noqa: BLE001 - logged, never raised
                shadow_result = e
            entry[backend.name] = _summary(shadow_result)
            _record(entry)

        finisher = asyncio.create_task(finish())
        _background.add(finisher)
        finisher.add_done_callback(_background.discard)

    return result if isinstance(result, Decision) else None


async def drain_shadow(timeout: float = 30.0) -> None:
    """Wait for outstanding shadow calls, so a CLI run logs them before exiting."""
    if _background:
        await asyncio.wait(list(_background), timeout=timeout)
