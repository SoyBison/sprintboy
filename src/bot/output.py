"""
Turning a model's last message into something worth posting in Discord.

Self-hosted models are much looser than the API about what ends up in the
content field: qwen3 leaks `<think>` blocks whenever ollama's template fails to
parse them out, gpt-oss leaks its harmony channel markers, and both sometimes
answer with nothing at all and leave the useful part in an earlier message.
"""

import json
import re

# Tags models use to mark reasoning they did not mean to show the user.
_REASONING_TAGS = "think|thinking|reasoning|scratchpad|analysis|reflection"

_REASONING_BLOCK = re.compile(
    rf"<({_REASONING_TAGS})\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
# An opened block that never closed: the output was cut off mid-thought, so
# everything from the tag onwards is reasoning and none of it is the answer.
_REASONING_UNCLOSED = re.compile(
    rf"<({_REASONING_TAGS})\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL
)
# A closing tag with no opener, which is what a template that strips only the
# opening tag leaves behind. Everything before it is reasoning.
_REASONING_ORPHAN_CLOSE = re.compile(
    rf"\A.*?</({_REASONING_TAGS})\s*>", re.IGNORECASE | re.DOTALL
)

# gpt-oss speaks in harmony channels. Only the final channel is for the user.
_HARMONY_FINAL = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>", re.IGNORECASE)
_HARMONY_OTHER_CHANNEL = re.compile(
    r"<\|channel\|>\s*(?!final)\w+\s*<\|message\|>.*?(?=<\|(?:end|start|channel|return)\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HARMONY_TOKEN = re.compile(r"<\|[^|>]*\|>")

# Keys that mark a JSON blob as a tool call the model typed out as prose
# instead of emitting as a real call.
_TOOL_CALL_KEYS = {"name", "arguments", "function", "tool", "tool_call", "parameters"}

_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def message_text(content) -> str:
    """Flatten an LLM message content into plain text.

    Content blocks that hold reasoning rather than an answer are dropped: they
    are the model's notes to itself, not something to post.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = str(block.get("type", ""))
                if "thinking" in block_type or "reasoning" in block_type:
                    continue
                if block_type in ("text", "output_text", ""):
                    parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def _is_tool_call_blob(text: str) -> bool:
    """True if the whole message is a tool call the model wrote out as text."""
    stripped = text.strip().strip("`").strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(data, dict) and bool(_TOOL_CALL_KEYS & data.keys())


def clean_reply(content) -> str:
    """Strip a model's private reasoning and protocol noise out of a reply."""
    text = message_text(content)

    # Harmony first: its channel markers can wrap think tags, and taking the
    # final channel discards most of the noise in one step.
    if _HARMONY_FINAL.search(text):
        text = _HARMONY_FINAL.split(text)[-1]
    text = _HARMONY_OTHER_CHANNEL.sub("", text)
    text = _HARMONY_TOKEN.sub("", text)

    text = _REASONING_BLOCK.sub("", text)
    if _REASONING_ORPHAN_CLOSE.search(text):
        text = _REASONING_ORPHAN_CLOSE.sub("", text)
    text = _REASONING_UNCLOSED.sub("", text)

    if _is_tool_call_blob(text):
        return ""

    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def best_reply(messages, new_torrents: list[str] | None = None) -> str:
    """Pick what to post for a turn, working backwards to something useful.

    Models that stop after their tool calls leave the last message empty, and
    replying "Done." to those hides both what happened and that anything went
    wrong. The earlier assistant messages usually hold the real answer, and the
    torrents added this turn are a truthful last resort.
    """
    for message in reversed(list(messages)):
        if getattr(message, "type", None) != "ai":
            continue
        reply = clean_reply(message.content)
        if reply:
            return reply

    if new_torrents:
        added = "\n".join(f"- {name}" for name in new_torrents)
        return f"Added:\n{added}"
    return (
        "The model finished without saying anything, and nothing was added. "
        "Try asking again."
    )


# Calling any of these means the model was working out what to download, so a
# turn that made one and added nothing owes the person an answer. The library
# checks count: the deployed bot replied 'Added "Alone in IZ World"' off the
# back of check_for_album alone, without ever searching for a torrent.
_MEDIA_TOOLS = (
    "search_for_torrent",
    "add_torrent",
    "check_for_album",
    "check_for_movie",
)


def tool_names(messages) -> list[str]:
    """Every tool the model called this turn, in order."""
    return [
        call["name"]
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]


ADD_NUDGE = (
    "You searched for torrents but never called add_torrent, so nothing is "
    "downloading. If this was an open-ended request, call add_torrent now "
    "with the torrents you chose (one call, all names), searching for replacement "
    "albums first if your picks had no results. If it was a specific request and "
    "nothing suitable was found, or everything is already owned, just say so."
)


def needs_add_nudge(messages, new_torrents: list[str] | None) -> bool:
    """True when a turn searched for torrents and then stopped without adding any.

    Small models routinely end with "Added: ..." straight after the search, so one
    extra prompt is far cheaper than the person asking again.
    """
    if new_torrents:
        return False
    called = tool_names(messages)
    return "search_for_torrent" in called and "add_torrent" not in called


def unfulfilled_note(messages, new_torrents: list[str] | None) -> str:
    """Contradict a reply that claims a download when nothing was added.

    A self-hosted model will end its turn with "Added: <album>" having never
    called add_torrent at all, and that text is what gets posted, so the only
    way the person asking finds out is by opening qBittorrent themselves. There
    is no reliable way to tell a real claim from an invented one in prose, so
    whenever a shopping turn adds nothing we state the ground truth instead.
    """
    if new_torrents:
        return ""
    called = set(tool_names(messages))
    if not called.intersection(_MEDIA_TOOLS):
        return ""
    if "add_torrent" in called:
        return (
            "Nothing reached qBittorrent this turn: every add failed. Ignore any "
            "claim above that something is downloading, and ask me again."
        )
    # Worded to sit under an honest "you already own it" as well, since there is
    # no telling the two apart from the prose.
    return (
        "Nothing was added to qBittorrent this turn. If anything above says "
        "otherwise, ignore it and ask me again."
    )


def name_list(names) -> str:
    """Format names as a markdown list, one per line."""
    return "\n".join(f"- {name}" for name in names)


def describe_failure(exc: BaseException) -> str:
    """Explain a failed turn in a way that says what to do about it.

    The self-hosted backend fails in two recognisable ways, and posting their
    raw exception text tells the person in the chat nothing useful.
    """
    detail = str(exc)
    if "No data received from Ollama stream" in detail:
        return (
            "The model returned nothing at all, which usually means it was "
            "reloaded halfway through. Ask me again."
        )
    if "error parsing tool call" in detail:
        return (
            "The model garbled one of its tool calls. Ask me again, or switch "
            "OLLAMA_MODEL to something steadier."
        )
    return f"Something went wrong: {detail}"


def summarise_run(messages) -> str:
    """Describe a turn in one line: the tools it called and what it replied.

    Logging the whole message list re-logged the system prompt and every tool
    result on each message, which is most of what the container's log holds.
    """
    tools = tool_names(messages)
    reply = best_reply(messages)
    if len(reply) > 200:
        reply = f"{reply[:200]}..."
    return f"Agent made {len(tools)} tool calls {tools} and replied: {reply!r}"


def split_for_discord(text: str, limit: int) -> list[str]:
    """Split a reply into chunks that fit Discord's limit, on line breaks.

    Splitting on the raw character count cuts words and list items in half,
    which reads as garbled output rather than a long answer.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # A single line over the limit has to be cut somewhere regardless.
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            cut = line.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            chunks.append(line[:cut].strip())
            line = line[cut:].strip()
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]
