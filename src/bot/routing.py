"""
Work out what a message is asking for before the LLM sees it.

The chat model used to classify every request itself from a paragraph of
prompt, and a small model would treat "5 albums like X" as done after one
album, or give up the moment the first pick was already owned. A decision
model answers the same questions in ~0.2s with calibrated probabilities, and
the answer becomes a plain instruction plus a smaller tool set.
"""

import re
from dataclasses import dataclass

from bot.decide import decide
from bot.questions import ROUTE_QUESTIONS, ROUTE_QUESTIONS_NAME

# Below this the route is not trusted, and the agent gets every tool and
# classifies the request itself as it did before.
MIN_CONFIDENCE = 0.6

DEFAULT_OPEN_ENDED_COUNT = 3
MAX_COUNT = 20

_NUMBER_WORDS = {
    "a couple": 2, "couple": 2, "a few": 3, "few": 3, "some": None,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a dozen": 12, "dozen": 12,
}
_COUNT = re.compile(
    r"\b(\d{1,2}|" + "|".join(sorted((re.escape(w) for w in _NUMBER_WORDS), key=len, reverse=True)) + r")\b"
    r"(?:\s+(?:more|new|other|different|good|great))*"
    r"\s+(?:albums?|records?|lps?|eps?|movies?|films?|shows?|things?|artists?|of\s+(?:them|those))",
    re.IGNORECASE,
)


def requested_count(message: str) -> int | None:
    """How many things the message asks for, if it says: "5 albums", "a couple records"."""
    match = _COUNT.search(message)
    if not match:
        return None
    token = match.group(1).lower()
    count = int(token) if token.isdigit() else _NUMBER_WORDS.get(token)
    if count is None or count < 1:
        return None
    return min(count, MAX_COUNT)


@dataclass(frozen=True)
class Route:
    domain: str
    domain_p: float
    kind: str
    kind_p: float
    count: int | None

    @property
    def trusted(self) -> bool:
        return self.domain_p >= MIN_CONFIDENCE and self.kind_p >= MIN_CONFIDENCE

    def note(self) -> str:
        """The instruction handed to the agent alongside the conversation."""
        if not self.trusted:
            return ""
        media = {"music": "music", "movie": "a movie", "tv": "TV"}.get(self.domain, "")
        if self.kind == "open_ended":
            count = self.count or DEFAULT_OPEN_ENDED_COUNT
            return (
                f"Routing: this is an OPEN-ENDED {media or 'media'} request. Add {count} new "
                f"item(s) they do not already own. Owned items do not count towards "
                f"the {count}: skip them and pick replacements until you reach {count}."
            )
        if self.kind == "discography":
            return (
                "Routing: this is a DISCOGRAPHY request. Get every studio album they "
                "are missing and nothing they already own."
            )
        if self.kind == "specific":
            return (
                f"Routing: this is a SPECIFIC {media} request. Get exactly what was "
                f"named, never anything they already own."
            )
        return (
            "Routing: this is a question, not a download request. Answer it; do not "
            "search for or add torrents."
        )


# How much of the conversation the router sees: enough to resolve "that" and
# "it", little enough to stay a ~0.2s call. Bot replies can be long lists.
EARLIER_TURNS = 4
EARLIER_CHARS = 300


def route_state(message: str, earlier: list[dict] | None = None) -> dict:
    """The router's input: the newest message plus the last few turns before it."""
    state: dict = {"message": message}
    turns = [
        {
            "from": "bot" if entry.get("role") == "assistant" else "user",
            "text": _clip(str(entry.get("content", ""))),
        }
        for entry in (earlier or [])[-EARLIER_TURNS:]
        if str(entry.get("content", "")).strip()
    ]
    if turns:
        state["earlier"] = turns
    return state


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EARLIER_CHARS else text[: EARLIER_CHARS - 1] + "…"


async def route(
    message: str, run_id: str | None = None, earlier: list[dict] | None = None
) -> Route | None:
    """Classify the latest message in its conversation, or None if no backend answered.

    `earlier` is the history before the message, as {"role", "content"} dicts.
    """
    decision = await decide(
        ROUTE_QUESTIONS_NAME, route_state(message, earlier), ROUTE_QUESTIONS, run_id=run_id
    )
    if decision is None:
        return None
    domain, domain_p = decision.choice("domain")
    kind, kind_p = decision.choice("kind")
    return Route(domain, domain_p, kind, kind_p, requested_count(message))
