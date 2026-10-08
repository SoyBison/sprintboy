"""
Build (candidate, owned) album pairs for the same_release question, from the
real library: every owned album against the artist's Last.fm releases and the
artist's other owned albums. Weighted towards the hard middle -- near-identical
titles that may or may not be the same album -- since exact matches and
obviously different titles are settled by code before the model is asked.

    uv run python evals/distill/gen_pairs.py --out data/distill/pairs.jsonl
"""

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from thefuzz import fuzz

load_dotenv()

from bot.netcode import LastFMClient, PlexAPIClient, PLEX_CONTENT_TYPES  # noqa: E402


async def library() -> dict[str, list[str]]:
    async with PlexAPIClient() as plex:
        data = await plex.get_all_library_items({"type": PLEX_CONTENT_TYPES["album"]})
    owned: dict[str, list[str]] = defaultdict(list)
    for item in (data.get("MediaContainer") or {}).get("Metadata") or []:
        if item.get("parentTitle") and item.get("title"):
            owned[item["parentTitle"]].append(item["title"])
    return owned


def band(a: str, b: str) -> str:
    score = fuzz.token_set_ratio(a.casefold(), b.casefold())
    if a.casefold() == b.casefold():
        return "identical"
    if score >= 75:
        return "close"
    if score >= 45:
        return "middling"
    return "far"


async def main(out: Path, max_artists: int, target: int):
    owned = await library()
    artists = [a for a in owned if a.strip()]
    random.shuffle(artists)
    artists = artists[:max_artists]
    print(f"{len(owned)} artists in the library, using {len(artists)}")
    semaphore = asyncio.Semaphore(4)

    async with LastFMClient() as lastfm:

        async def releases(artist: str) -> list[str]:
            async with semaphore:
                try:
                    return [a.name for a in await lastfm.get_artist_albums(artist, limit=30)]
                except Exception:
                    return []

        found = await asyncio.gather(*[releases(a) for a in artists])

    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for artist, lastfm_titles in zip(artists, found):
        mine = owned[artist]
        for title in mine:
            for other in set(lastfm_titles) | (set(mine) - {title}):
                pairs[band(other, title)].append((f"{artist} - {other}", f"{artist} - {title}"))
    # Hard cases first: most of the budget on close titles, some of each other kind.
    quota = {"close": 0.55, "middling": 0.2, "identical": 0.1, "far": 0.15}
    rows = []
    for name, share in quota.items():
        pool = list(dict.fromkeys(pairs[name]))
        random.shuffle(pool)
        rows += [
            {"candidate": c, "owned": o, "band": name}
            for c, o in pool[: int(target * share)]
        ]
    random.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in pairs.items()}, "->", len(rows), "pairs written to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/distill/pairs.jsonl"))
    parser.add_argument("--max-artists", type=int, default=600)
    parser.add_argument("--target", type=int, default=2500)
    args = parser.parse_args()
    asyncio.run(main(args.out, args.max_artists, args.target))
