"""
The agent loop: ask the model, run whatever tools it called, repeat until it
answers in plain text.
"""

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import ValidationError

from bot.llm import ChatModel, Message
from bot.toolkit import Runtime, Tool

logger = logging.getLogger("bot.agent")


OUT_OF_STEPS = (
    "You are out of steps and cannot call any more tools. Reply now: say what was "
    "added (only items a tool answered \"Added\" for), and what you tried that did "
    "not work, in a sentence or two."
)


@dataclass
class Step:
    kind: Literal["model", "tool"]
    name: str
    seconds: float
    args: dict | None = None
    result: str | None = None
    error: str | None = None
    usage: dict | None = None


@dataclass
class AgentResult:
    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    stopped: str = "reply"
    nudged: bool = False
    # Unresolved "did you mean" questions (workflows.Pending) for the chat to ask.
    pending: list = field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _validation_summary(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in e.errors()
    )


async def run_agent(
    model: ChatModel,
    messages: list[Message],
    tools: list[Tool],
    context: Any,
    *,
    run_id: str | None = None,
    max_steps: int = 12,
    on_event: Callable[[str, dict], Awaitable[None] | None] | None = None,
) -> AgentResult:
    transcript = list(messages)
    steps: list[Step] = []
    by_name = {t.name: t for t in tools}
    runtime = Runtime(context=context, run_id=run_id)

    async def emit(event: str, data: dict) -> None:
        if on_event is None:
            return
        result = on_event(event, data)
        if inspect.isawaitable(result):
            await result

    async def execute(call: dict) -> str:
        name, args = call["name"], call["args"]
        started = time.perf_counter()
        error = None
        tool = by_name.get(name)
        if tool is None:
            result = f"Error: there is no tool called {name}. Available: {', '.join(by_name)}"
            error = result
        else:
            try:
                result = await tool.ainvoke(args, runtime)
            except ValidationError as e:
                result = f"Error: invalid arguments for {name}: {_validation_summary(e)}"
                error = result
            except Exception as e:
                logger.exception("Run %s tool %s failed", run_id, name)
                result = f"Error: {name} failed: {type(e).__name__}: {e}"
                error = result
        seconds = time.perf_counter() - started
        steps.append(
            Step("tool", name, seconds, args=args, result=result, error=error)
        )
        logger.info(
            "Run %s tool %s(%s) %.2fs -> %s",
            run_id,
            name,
            _clip(json.dumps(args, default=str), 300),
            seconds,
            _clip(result, 200),
        )
        await emit("tool_result", {"name": name, "result": result})
        return result

    for _ in range(max_steps):
        started = time.perf_counter()
        reply = await model.chat(transcript, tools)
        seconds = time.perf_counter() - started
        transcript.append(reply)
        steps.append(
            Step(
                "model",
                model.name,
                seconds,
                result=reply.content,
                usage=reply.usage or None,
            )
        )
        logger.info(
            "Run %s model %s %.1fs -> %s, tokens %s/%s",
            run_id,
            model.name,
            seconds,
            (
                f"tools {[c['name'] for c in reply.tool_calls]}"
                if reply.tool_calls
                else f"reply {len(reply.content)} chars"
            ),
            reply.usage.get("input_tokens"),
            reply.usage.get("output_tokens"),
        )
        if not reply.tool_calls:
            await emit("reply", {"content": reply.content})
            return AgentResult(transcript, steps, "reply")

        for call in reply.tool_calls:
            await emit("tool_call", {"name": call["name"], "args": call["args"]})
        results = await asyncio.gather(*(execute(call) for call in reply.tool_calls))
        for call, result in zip(reply.tool_calls, results):
            transcript.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    # Out of steps: one last call with no tools, so the person gets an account
    # of what happened instead of silence. Without this a model that keeps
    # finding nothing (obscure artists, no torrents) ends the turn mute.
    transcript.append(Message(role="user", content=OUT_OF_STEPS))
    started = time.perf_counter()
    try:
        reply = await model.chat(transcript, [])
    except Exception:
        logger.exception("Run %s: final no-tools reply failed", run_id)
        return AgentResult(transcript, steps, "max_steps")
    seconds = time.perf_counter() - started
    transcript.append(reply)
    steps.append(Step("model", model.name, seconds, result=reply.content, usage=reply.usage or None))
    logger.info("Run %s hit %d steps; forced a final reply in %.1fs", run_id, max_steps, seconds)
    await emit("reply", {"content": reply.content})
    return AgentResult(transcript, steps, "max_steps")
