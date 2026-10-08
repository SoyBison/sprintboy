"""
Generate realistic Discord requests for distilling Jev's routing into Laya.

The local chat model writes the messages (free, and it runs on the box's GPU
before training needs it); which category a batch was asked for is only there
for diversity -- the training label is whatever Jev answers, see label.py.

    uv run python evals/distill/gen_messages.py --batches 220 --out data/distill/messages.jsonl
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from bot.config import Config  # noqa: E402
from bot.netcode import LastFMClient, PlexAPIClient, PLEX_CONTENT_TYPES  # noqa: E402
from bot.tags import tag_vocabulary  # noqa: E402

CATEGORIES = {
    # name: (weight, what to write)
    "music_specific": (14, "asks to download one or more particular albums by name (sometimes with the artist, sometimes not, sometimes describing it by a song on it)"),
    "music_discography": (7, "asks for an artist's whole or remaining discography, everything by them, or their newest album"),
    "music_open": (14, "asks for recommendations to be downloaded: albums like an artist or album, a genre, a mood, an era, 'surprise me', 'something new'"),
    "music_question": (6, "asks a question about music or their library without wanting a download: what an artist sounds like, whether they own something, who made an album"),
    "movie_specific": (6, "asks to download a particular film"),
    "movie_open": (4, "asks for some films to be picked and downloaded: a genre, a mood, 'something like X'"),
    "movie_question": (2, "asks a question about films without wanting a download"),
    "tv_specific": (5, "asks to download a particular show, season or episode"),
    "tv_open": (3, "asks for a show to be picked and downloaded"),
    "tv_question": (2, "asks a question about TV without wanting a download"),
    "chat": (5, "small talk or questions about the bot itself: thanks, status of a download, complaints, greetings"),
    "tracker_status": (4, "asks about their private tracker account: ratio, bonus points, how many freeleech tokens they have (often just 'tokens')"),
    "tracker_buy": (3, "asks to buy freeleech tokens with bonus points, a number of them or as many as possible"),
}

FOLLOWUP_WEIGHT = 25  # batches of conversations, out of the category weights' total

STYLES = [
    "all lowercase, terse, no punctuation",
    "full polite sentences",
    "casual slang, a bit rude or impatient",
    "with typos and misspelled names",
    "vague and half-remembered",
    "very short, three or four words",
    "long and chatty with extra context",
    "mentions a format preference such as CD, vinyl, SACD or 24 bit",
    "asks for several things in one message",
]

SCHEMA_MESSAGES = {
    "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
    "required": ["messages"],
}
SCHEMA_CONVERSATIONS = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "first": {"type": "string"},
                    "bot_reply": {"type": "string"},
                    "follow_up": {"type": "string"},
                },
                "required": ["first", "bot_reply", "follow_up"],
            },
        }
    },
    "required": ["conversations"],
}

BOT = (
    "The bot is a Discord assistant that downloads music, films and TV to the user's "
    "Plex server from private torrent trackers, and can report on their Orpheus tracker "
    "account (ratio, bonus points, freeleech tokens)."
)


async def seeds() -> tuple[list[str], list[str]]:
    """Real artist/album names, owned and not, plus genre tags, for grounding."""
    names: set[str] = set()
    async with PlexAPIClient() as plex:
        data = await plex.get_all_library_items({"type": PLEX_CONTENT_TYPES["album"]})
    for item in (data.get("MediaContainer") or {}).get("Metadata") or []:
        if item.get("parentTitle") and item.get("title"):
            names.add(f"{item['parentTitle']} - {item['title']}")
    tags = await tag_vocabulary()
    async with LastFMClient() as lastfm:
        for tag in random.sample(tags, min(30, len(tags))):
            try:
                for album in await lastfm.get_tag_top_albums(tag, limit=15):
                    if album.artist:
                        names.add(f"{album.artist} - {album.name}")
            except Exception as e:  # a missing tag is not worth stopping for
                print(f"tag {tag}: {e}")
    return sorted(names), tags


async def chat(session, prompt: str, schema: dict) -> dict:
    body = {
        "model": Config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 1.0, "num_ctx": 8192},
        "keep_alive": "30m",
    }
    async with session.post(
        f"{Config.OLLAMA_API_URL}/api/chat", json=body, timeout=aiohttp.ClientTimeout(total=600)
    ) as r:
        data = await r.json()
    return json.loads(data["message"]["content"])


def category_prompt(category: str, style: str, examples: list[str], tags: list[str]) -> str:
    what = CATEGORIES[category][1]
    return (
        f"{BOT}\n\nWrite 20 different messages a user might send it. Every message "
        f"{what}. Style: {style}. Vary the wording a lot; do not number them. Where a "
        f"message names music, use real releases such as these (artist - album): "
        f"{'; '.join(examples)}. Genres you may use: {', '.join(tags)}. Use real film "
        f"and TV titles where needed. Output JSON only."
    )


def followup_prompt(style: str, examples: list[str]) -> str:
    return (
        f"{BOT}\n\nWrite 10 short three-turn exchanges: the user's first message, the "
        f"bot's reply (what it added, what it could not find, or a question it answered), "
        f"and the user's follow-up. Make the follow-ups depend on the earlier turns: "
        f"corrections ('no, the other one', 'it's on X'), retries ('try the deluxe'), "
        f"'more like that', 'and the next season?', thanks, 'did that finish?', or a new "
        f"unrelated request. Mix music, film, TV and tracker-account topics. Style of the "
        f"follow-ups: {style}. Real releases you may use: {'; '.join(examples)}. JSON only."
    )


async def main(batches: int, out: Path, concurrency: int):
    names, tags = await seeds()
    print(f"{len(names)} seed releases, {len(tags)} tags")
    plan = []
    weights = {c: w for c, (w, _) in CATEGORIES.items()} | {"followup": FOLLOWUP_WEIGHT}
    for _ in range(batches):
        plan.append(random.choices(list(weights), weights=list(weights.values()))[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    written = 0

    async with aiohttp.ClientSession() as session:

        async def one(index: int, category: str):
            nonlocal written
            style = random.choice(STYLES)
            examples = random.sample(names, min(12, len(names)))
            async with semaphore:
                try:
                    if category == "followup":
                        data = await chat(session, followup_prompt(style, examples), SCHEMA_CONVERSATIONS)
                        rows = [
                            {
                                "category": "followup",
                                "style": style,
                                "message": c["follow_up"],
                                "earlier": [
                                    {"role": "user", "content": c["first"]},
                                    {"role": "assistant", "content": c["bot_reply"]},
                                ],
                            }
                            for c in data.get("conversations", [])
                            if c.get("follow_up", "").strip()
                        ]
                    else:
                        data = await chat(
                            session,
                            category_prompt(category, style, examples, random.sample(tags, 8)),
                            SCHEMA_MESSAGES,
                        )
                        rows = [
                            {"category": category, "style": style, "message": m, "earlier": []}
                            for m in data.get("messages", [])
                            if isinstance(m, str) and m.strip()
                        ]
                except Exception as e:
                    print(f"batch {index} ({category}) failed: {e}")
                    return
            with out.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += len(rows)
            if index % 10 == 0:
                print(f"batch {index}/{batches}: {written} messages so far", flush=True)

        await asyncio.gather(*[one(i, c) for i, c in enumerate(plan)])
    print(f"wrote {written} messages to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=220)
    parser.add_argument("--out", type=Path, default=Path("data/distill/messages.jsonl"))
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(args.batches, args.out, args.concurrency))
