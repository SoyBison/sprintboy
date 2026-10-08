"""
Compare decision models (hosted Jev, self-hosted Laya via ollaya) on the
judgements the bot needs to make.

Every request is one call with all its questions batched, and answers are
cached in evals/cache/ keyed by (model, request), so re-running costs nothing
unless a question set or case changes. Pass --refresh to ignore the cache.

    uv run python evals/decision_eval.py
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from bot.questions import ROUTE_QUESTIONS, SAME_RELEASE_QUESTIONS

load_dotenv()

CACHE = Path(__file__).parent / "cache"

BACKENDS = {
    "jev": ("https://api.typesafe.ai", "TYPESAFE_API_KEY", "jev-latest"),
    "laya": (os.getenv("OLLAYA_URL", "http://100.99.141.127:11435"), "OLLAYA_API_KEY", "laya"),
}

# ---------------------------------------------------------------------------
# Routing: what is this message asking for?
# ---------------------------------------------------------------------------

# (message, domain, kind)
ROUTE_CASES = [
    ("throw some new shoegaze at me", "music", "open_ended"),
    ("get me 5 albums similar to Khruangbin", "music", "open_ended"),
    ("I want something chill to work to", "music", "open_ended"),
    ("surprise me", "music", "open_ended"),
    ("more stuff like the last thing you got me", "music", "open_ended"),
    ("introduce me to some japanese city pop", "music", "open_ended"),
    ("got any good jazz from the 60s?", "music", "open_ended"),
    ("what's a good starting point for Aphex Twin", "music", "open_ended"),
    ("some sad girl music pls", "music", "open_ended"),
    ("find me a couple albums like Mordechai but heavier", "music", "open_ended"),
    ("get me Mordechai by Khruangbin", "music", "specific"),
    ("download Loveless and Souvlaki", "music", "specific"),
    ("can you grab In Rainbows", "music", "specific"),
    ("add Kendrick's GNX", "music", "specific"),
    ("get the deluxe edition of Djesse Vol. 4", "music", "specific"),
    ("get me the rest of the Khruangbin discography", "music", "discography"),
    ("Update my Jacob Collier discography to include his newest album.", "music", "discography"),
    ("grab everything by Boards of Canada", "music", "discography"),
    ("fill in my Radiohead collection", "music", "discography"),
    ("download Dune Part Two", "movie", "specific"),
    ("get me the new Paul Thomas Anderson movie", "movie", "specific"),
    ("I want a good horror movie for tonight", "movie", "open_ended"),
    ("grab some 80s action flicks", "movie", "open_ended"),
    ("get season 2 of Severance", "tv", "specific"),
    ("download the latest episode of The Bear", "tv", "specific"),
    ("find me a new sitcom to binge", "tv", "open_ended"),
    ("what does Khruangbin sound like?", "chat", "question"),
    ("do I have any Slowdive?", "chat", "question"),
    ("thanks!", "chat", "question"),
    ("why didn't that download work", "chat", "question"),
]

# ---------------------------------------------------------------------------
# Same release: the "maybe owned as" cases check_for_album hands the LLM today
# ---------------------------------------------------------------------------

# (candidate, owned, same?)
RELEASE_CASES = [
    ("Khruangbin - Mordechai", "Khruangbin - Mordechai", True),
    ("Jacob Collier - Djesse Vol. 4 (Deluxe)", "Jacob Collier - Djesse, Vol. 4", True),
    ("Madvillain - MM..FOOD", "MF DOOM - MM.. FOOD", True),
    ("Slowdive - Souvlaki (Remastered 2005)", "Slowdive - Souvlaki", True),
    ("Khruangbin - Con Todo El Mundo", "Khruangbin - Con Todo El Mundo (Excluding N & S America)", True),
    ("Radiohead - OK Computer OKNOTOK 1997 2017", "Radiohead - OK Computer", True),
    ("Metallica - Metallica Through the Never", "Metallica - Metallica", False),
    ("Hamid El Kasri performed by Various Artists - Djesse Vol.1", "Jacob Collier - Djesse, Vol. 1", False),
    ("Jacob Collier - Djesse Vol. 2", "Jacob Collier - Djesse, Vol. 1", False),
    ("Khruangbin - Texas Sun", "Khruangbin - Texas Moon", False),
    ("Jacob Collier - Piano Ballads - Live from the Djesse World Tour 2022", "Jacob Collier - Djesse, Vol. 3", False),
    ("Khruangbin - The Universe Smiles Upon You ii", "Khruangbin - The Universe Smiles Upon You", False),
]


def _cache_path(backend: str, body: dict) -> Path:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE / f"{backend}-{digest}.json"


async def ask(session, backend: str, state, questions: dict, refresh: bool) -> dict:
    url, key_var, model = BACKENDS[backend]
    body = {"model": model, "state": state, "questions": questions}
    path = _cache_path(backend, body)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    start = time.perf_counter()
    async with session.post(
        f"{url}/v1/systemone",
        json=body,
        headers={"Authorization": f"Bearer {os.environ[key_var]}"},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as response:
        result = await response.json()
        if response.status != 200:
            raise RuntimeError(f"{backend} {response.status}: {result}")
    result["_latency"] = time.perf_counter() - start
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


async def run(backend: str, refresh: bool):
    # Jev in parallel is fine; serial for the CPU-bound local server so the
    # latencies mean something.
    limit = asyncio.Semaphore(8 if backend == "jev" else 1)
    async with aiohttp.ClientSession() as session:

        async def one(state, questions):
            async with limit:
                return await ask(session, backend, state, questions, refresh)

        routes = await asyncio.gather(
            *[one({"message": m}, ROUTE_QUESTIONS) for m, _, _ in ROUTE_CASES]
        )
        releases = await asyncio.gather(
            *[
                one({"candidate": c, "owned": o}, SAME_RELEASE_QUESTIONS)
                for c, o, _ in RELEASE_CASES
            ]
        )
    return routes, releases


def report(name: str, routes, releases):
    domain_ok = kind_ok = open_ok = 0
    misses = []
    for (message, domain, kind), result in zip(ROUTE_CASES, routes):
        a = result["answers"]
        want_open = kind == "open_ended"
        got_open = a["kind"]["probabilities"].get("open_ended", 0.0)
        domain_ok += a["domain"]["choice"] == domain
        kind_ok += a["kind"]["choice"] == kind
        open_ok += (got_open >= 0.5) == want_open
        if a["kind"]["choice"] != kind or (got_open >= 0.5) != want_open or a["domain"]["choice"] != domain:
            misses.append(
                f"  {message!r}: domain {a['domain']['choice']} (want {domain}), "
                f"kind {a['kind']['choice']} {a['kind']['confidence']:.2f} (want {kind}), "
                f"open {got_open:.2f}"
            )
    release_ok = 0
    for (candidate, owned, same), result in zip(RELEASE_CASES, releases):
        p = result["answers"]["same_release"]["noul"]
        release_ok += (p >= 0.5) == same
        if (p >= 0.5) != same:
            misses.append(f"  same? {candidate!r} vs {owned!r}: {p:.2f} (want {same})")

    latencies = sorted(r.get("_latency", 0) for r in [*routes, *releases])
    tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in [*routes, *releases])
    n, m = len(ROUTE_CASES), len(RELEASE_CASES)
    print(f"\n== {name}")
    print(f"domain       {domain_ok}/{n}")
    print(f"kind         {kind_ok}/{n}")
    print(f"open_ended   {open_ok}/{n}")
    print(f"same_release {release_ok}/{m}")
    print(f"latency p50 {latencies[len(latencies) // 2]:.2f}s  max {latencies[-1]:.2f}s  ({tokens} input tokens)")
    if misses:
        print("misses:")
        print("\n".join(misses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backends", nargs="*", default=["laya", "jev"])
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = parser.parse_args()
    for backend in args.backends:
        routes, releases = asyncio.run(run(backend, args.refresh))
        report(backend, routes, releases)


if __name__ == "__main__":
    main()
