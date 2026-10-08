import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from bot.llm import (
    AnthropicChat,
    LLMError,
    Message,
    OllamaChat,
    assistant,
    from_dict,
    system,
    user,
)
from bot.toolkit import tool



class Q(BaseModel):
    artist: str


@tool(args_schema=Q)
async def lookup(artist: str) -> str:
    """Look up."""
    return artist


def transcript():
    return [
        system("sys"),
        user("hi"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=[
                {"id": "c1", "name": "lookup", "args": {"artist": "A"}},
                {"id": "c2", "name": "lookup", "args": {"artist": "B"}},
            ],
        ),
        Message(role="tool", content="ra", tool_call_id="c1", name="lookup"),
        Message(role="tool", content="rb", tool_call_id="c2", name="lookup"),
    ]


def test_message_types_and_helpers():
    assert [m.type for m in (system("a"), user("a"), assistant("a"))] == ["system", "human", "ai"]
    assert Message(role="tool").type == "tool"
    assert from_dict({"role": "user", "content": "x"}) == user("x")
    with pytest.raises(ValueError):
        from_dict({"role": "tool", "content": "x"})


class FakeOllama(OllamaChat):
    def __init__(self, status=200, body=None, **kw):
        super().__init__("http://h:11434/", "m", 4096, **kw)
        self.status, self.reply = status, body
        self.posts = []

    async def _post(self, path, body):
        self.posts.append((path, body))
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return self.status, text


@pytest.mark.asyncio
async def test_ollama_request_body():
    model = FakeOllama(body={"message": {"content": "hi"}})
    await model.chat(transcript(), [lookup])
    path, body = model.posts[0]
    assert path == "/api/chat"
    assert body["model"] == "m" and body["stream"] is False and body["keep_alive"] == "30m"
    assert body["options"] == {"num_ctx": 4096, "temperature": 0.0}
    assert "think" not in body
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up.",
                "parameters": lookup.json_schema(),
            },
        }
    ]
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"artist": "A"}}},
                {"function": {"name": "lookup", "arguments": {"artist": "B"}}},
            ],
        },
        {"role": "tool", "content": "ra", "tool_name": "lookup"},
        {"role": "tool", "content": "rb", "tool_name": "lookup"},
    ]


@pytest.mark.asyncio
async def test_ollama_think_flag():
    model = FakeOllama(body={"message": {}}, think=False)
    await model.chat([user("x")], [])
    assert model.posts[0][1]["think"] is False


@pytest.mark.asyncio
async def test_ollama_response_parsing():
    model = FakeOllama(
        body={
            "message": {
                "content": "",
                "thinking": "hmm",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"artist": "A"}}},
                    {"function": {"name": "lookup", "arguments": '{"artist": "B"}'}},
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
    )
    reply = await model.chat([user("x")], [lookup])
    assert reply.role == "assistant" and reply.thinking == "hmm"
    assert [(c["name"], c["args"]) for c in reply.tool_calls] == [
        ("lookup", {"artist": "A"}),
        ("lookup", {"artist": "B"}),
    ]
    assert len({c["id"] for c in reply.tool_calls}) == 2
    assert reply.usage == {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.asyncio
async def test_ollama_error_keeps_server_text():
    model = FakeOllama(status=500, body="error parsing tool call: raw='{'")
    with pytest.raises(LLMError, match="ollama 500: error parsing tool call"):
        await model.chat([user("x")], [])


@pytest.mark.asyncio
async def test_ollama_validate():
    with pytest.raises(LLMError, match="ollama pull m"):
        await FakeOllama(status=404, body="nope").validate()
    ok = FakeOllama(body={})
    await ok.validate()
    assert ok.posts == [("/api/show", {"model": "m"})]


def test_anthropic_conversion_merges_tool_results():
    converted = AnthropicChat._convert(transcript())
    assert converted == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"artist": "A"}},
                {"type": "tool_use", "id": "c2", "name": "lookup", "input": {"artist": "B"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "ra"},
                {"type": "tool_result", "tool_use_id": "c2", "content": "rb"},
            ],
        },
    ]


@pytest.mark.asyncio
async def test_anthropic_chat_request_and_response():
    model = AnthropicChat("claude-x", "key")
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="pondering"),
                SimpleNamespace(type="text", text="Hello "),
                SimpleNamespace(type="text", text="there"),
                SimpleNamespace(type="tool_use", id="tu1", name="lookup", input={"artist": "A"}),
            ],
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )

    model.client.messages.create = create
    reply = await model.chat([system("s1"), system("s2"), user("hi")], [lookup])
    assert sent["system"] == "s1\n\ns2"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["tools"] == [
        {"name": "lookup", "description": "Look up.", "input_schema": lookup.json_schema()}
    ]
    assert reply.content == "Hello there"
    assert reply.thinking == "pondering"
    assert reply.tool_calls == [{"id": "tu1", "name": "lookup", "args": {"artist": "A"}}]
    assert reply.usage == {"input_tokens": 7, "output_tokens": 3}
