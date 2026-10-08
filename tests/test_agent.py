import asyncio

import pytest
from pydantic import BaseModel

from bot.agent import run_agent
from bot.llm import Message, system, user
from bot.toolkit import tool



class Echo(BaseModel):
    text: str


@tool(args_schema=Echo)
async def echo(text: str, runtime) -> str:
    """Echo."""
    runtime.context.seen.append((text, runtime.run_id))
    return f"echo {text}"


@tool(args_schema=Echo)
async def boom(text: str) -> str:
    raise RuntimeError("kaput")


class Ctx:
    def __init__(self):
        self.seen = []


class FakeModel:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, tools):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def call(i, name, **args):
    return {"id": f"c{i}", "name": name, "args": args}


def ai(content="", *calls):
    return Message(role="assistant", content=content, tool_calls=list(calls))


START = [system("sys"), user("hi")]


@pytest.mark.asyncio
async def test_tool_then_reply():
    ctx = Ctx()
    model = FakeModel([ai("", call(1, "echo", text="a")), ai("done")])
    result = await run_agent(model, START, [echo], ctx, run_id="r1")
    assert result.stopped == "reply"
    assert [m.type for m in result.messages] == ["system", "human", "ai", "tool", "ai"]
    assert ctx.seen == [("a", "r1")]
    tool_msg = result.messages[3]
    assert (tool_msg.content, tool_msg.tool_call_id, tool_msg.name) == ("echo a", "c1", "echo")
    assert [s.kind for s in result.steps] == ["model", "tool", "model"]
    assert result.steps[1].args == {"text": "a"}
    assert len(START) == 2  # input not mutated


@pytest.mark.asyncio
async def test_unknown_tool_and_invalid_args_continue():
    model = FakeModel(
        [
            ai("", call(1, "nope", x=1), call(2, "echo", wrong=1)),
            ai("sorry"),
        ]
    )
    result = await run_agent(model, START, [echo, boom], Ctx())
    assert result.stopped == "reply"
    unknown, invalid = result.messages[3], result.messages[4]
    assert unknown.content == "Error: there is no tool called nope. Available: echo, boom"
    assert invalid.content.startswith("Error: invalid arguments for echo: ")
    assert "text" in invalid.content
    assert [s.error is not None for s in result.steps if s.kind == "tool"] == [True, True]


@pytest.mark.asyncio
async def test_tool_exception_reported():
    model = FakeModel([ai("", call(1, "boom", text="x")), ai("ok")])
    result = await run_agent(model, START, [boom], Ctx())
    assert result.messages[3].content == "Error: boom failed: RuntimeError: kaput"


@pytest.mark.asyncio
async def test_tool_calls_run_concurrently():
    both_started = asyncio.Event()
    started = []

    class Wait(BaseModel):
        n: int

    @tool(args_schema=Wait)
    async def rendezvous(n: int) -> str:
        started.append(n)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return f"r{n}"

    model = FakeModel([ai("", call(1, "rendezvous", n=1), call(2, "rendezvous", n=2)), ai("x")])
    result = await run_agent(model, START, [rendezvous], Ctx())
    assert [m.content for m in result.messages if m.role == "tool"] == ["r1", "r2"]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_max_steps():
    model = FakeModel(
        [ai("", call(i, "echo", text="a")) for i in range(3)] + [ai("Found nothing.")]
    )
    result = await run_agent(model, START, [echo], Ctx(), max_steps=3)
    assert result.stopped == "max_steps"
    # Three tool rounds, then one forced reply with no tools on offer.
    assert len(model.calls) == 4
    assert result.messages[-2].role == "user" and "out of steps" in result.messages[-2].content
    assert result.messages[-1].content == "Found nothing."


@pytest.mark.asyncio
async def test_events_in_order_sync_and_async():
    for is_async in (False, True):
        events = []
        if is_async:
            async def on_event(kind, data):
                events.append((kind, data))
        else:
            def on_event(kind, data):
                events.append((kind, data))

        model = FakeModel([ai("", call(1, "echo", text="a")), ai("final")])
        await run_agent(model, START, [echo], Ctx(), on_event=on_event)
        assert events == [
            ("tool_call", {"name": "echo", "args": {"text": "a"}}),
            ("tool_result", {"name": "echo", "result": "echo a"}),
            ("reply", {"content": "final"}),
        ]
