"""
Chat backends: Ollama over HTTP and Anthropic through its SDK, both behind the
same `ChatModel.chat(messages, tools) -> Message` call.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

import aiohttp

from bot.config import Config
from bot.toolkit import Tool

logger = logging.getLogger(__name__)

_TYPE_NAMES = {"system": "system", "user": "human", "assistant": "ai", "tool": "tool"}


class LLMError(Exception):
    pass


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # Each {"id": str, "name": str, "args": dict}
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str | None = None  # for role == "tool"
    name: str | None = None  # tool name, for role == "tool"
    # Reasoning the backend returned separately. Never shown to users.
    thinking: str = ""
    usage: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        """LangChain's names for the roles, which output.py relies on."""
        return _TYPE_NAMES[self.role]


def system(text: str) -> Message:
    return Message(role="system", content=text)


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str) -> Message:
    return Message(role="assistant", content=text)


def from_dict(d: dict) -> Message:
    role = d["role"]
    if role not in ("system", "user", "assistant"):
        raise ValueError(f"unsupported role {role!r}")
    return Message(role=role, content=d.get("content", "") or "")


class ChatModel(Protocol):
    name: str

    async def chat(self, messages: list[Message], tools: list[Tool]) -> Message: ...


class OllamaChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        temperature: float = 0.0,
        think: bool | None = None,
        timeout: float = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self.timeout = timeout

    @staticmethod
    def _convert(message: Message) -> dict:
        if message.role == "tool":
            return {"role": "tool", "content": message.content, "tool_name": message.name}
        out: dict = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": call["name"], "arguments": call["args"]}}
                for call in message.tool_calls
            ]
        return out

    def _request_body(self, messages: list[Message], tools: list[Tool]) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [self._convert(m) for m in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema(),
                    },
                }
                for t in tools
            ],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
            "keep_alive": "30m",
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    async def _post(self, path: str, body: dict) -> tuple[int, str]:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}{path}", json=body) as response:
                return response.status, await response.text()

    async def chat(self, messages: list[Message], tools: list[Tool]) -> Message:
        status, text = await self._post("/api/chat", self._request_body(messages, tools))
        if status != 200:
            raise LLMError(f"ollama {status}: {text[:500]}")
        try:
            body = json.loads(text)
            msg = body["message"]
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError(f"ollama sent an unreadable reply: {text[:500]}") from e
        tool_calls = []
        for i, call in enumerate(msg.get("tool_calls") or []):
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError as e:
                    raise LLMError(
                        f"ollama sent unparseable arguments for {fn['name']}: {args[:500]}"
                    ) from e
            tool_calls.append(
                {"id": f"call_{i}_{uuid.uuid4().hex[:8]}", "name": fn["name"], "args": args}
            )
        return Message(
            role="assistant",
            content=msg.get("content") or "",
            thinking=msg.get("thinking") or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"),
            },
        )

    async def validate(self) -> None:
        """Fail loudly if the model has not been pulled."""
        status, text = await self._post("/api/show", {"model": self.model})
        if status == 404:
            raise LLMError(
                f"ollama model {self.model!r} not found at {self.base_url}: "
                f"run `ollama pull {self.model}`"
            )
        if status != 200:
            raise LLMError(f"ollama {status}: {text[:500]}")


class AnthropicChat:
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        from anthropic import AsyncAnthropic

        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                last = out[-1] if out else None
                if (
                    last
                    and last["role"] == "user"
                    and isinstance(last["content"], list)
                    and last["content"]
                    and all(b.get("type") == "tool_result" for b in last["content"])
                ):
                    last["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif message.role == "assistant" and message.tool_calls:
                blocks: list[dict] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks += [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                    for c in message.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": message.role, "content": message.content})
        return out

    async def chat(self, messages: list[Message], tools: list[Tool]) -> Message:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._convert(messages),
        }
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.json_schema()}
                for t in tools
            ]
        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as e:
            raise LLMError(f"anthropic: {type(e).__name__}: {e}"[:500]) from e
        text, thinking, tool_calls = [], [], []
        for block in response.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "thinking":
                thinking.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": dict(block.input)})
        usage = getattr(response, "usage", None)
        return Message(
            role="assistant",
            content="".join(text),
            thinking="\n".join(thinking),
            tool_calls=tool_calls,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            },
        )


def build_chat_model() -> ChatModel:
    Config.validate_llm()
    if Config.LLM_PROVIDER == "ollama":
        think = os.getenv("OLLAMA_THINK", "").strip().lower()
        return OllamaChat(
            Config.OLLAMA_API_URL,
            Config.OLLAMA_MODEL,
            Config.OLLAMA_NUM_CTX,
            think={"true": True, "false": False}.get(think),
        )
    return AnthropicChat(Config.ANTHROPIC_MODEL, Config.ANTHROPIC_API_KEY)
