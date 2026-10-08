"""
Label distillation states with Jev and write laya-train rows.

Each output row is {"state", "questions", "gold"}, where gold holds Jev's full
probability distribution per question (the soft targets RLCD/soft-CE train on).
Answers are cached by request hash, like evals/decision_eval.py, so a re-run
after a crash never pays twice.

    uv run python evals/distill/label.py route data/distill/messages.jsonl data/distill/route.jsonl
    uv run python evals/distill/label.py same_release data/distill/pairs.jsonl data/distill/same_release.jsonl
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from bot.questions import ROUTE_QUESTIONS, SAME_RELEASE_QUESTIONS  # noqa: E402
from bot.routing import route_state  # noqa: E402

CACHE = Path("data/distill/jev_cache")
URL = "https://api.typesafe.ai/v1/systemone"


def build(kind: str, row: dict) -> tuple[dict, dict]:
    if kind == "route":
        return route_state(row["message"], row.get("earlier") or []), ROUTE_QUESTIONS
    return {"candidate": row["candidate"], "owned": row["owned"]}, SAME_RELEASE_QUESTIONS


def gold(questions: dict, answers: dict) -> dict:
    """Jev's answers as laya-train gold: probabilities keyed the way target_from_gold reads them."""
    out = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "noul":
            p = float(a["noul"])
            out[qid] = {"probabilities": {"true": p, "false": 1.0 - p}}
        elif q["type"] == "choice":
            out[qid] = {"probabilities": {k: float(v) for k, v in a["probabilities"].items()}}
        else:  # score: Jev returns per-level probabilities as a list or dict
            probs = a.get("probabilities")
            if isinstance(probs, list):
                probs = {str(i): float(p) for i, p in enumerate(probs)}
            out[qid] = {"probabilities": probs}
    return out


async def ask(session, body: dict) -> dict:
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        async with session.post(
            URL,
            json=body,
            headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            data = await r.json(content_type=None)
        if r.status == 200 and "answers" in data:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return data
        if r.status in (429, 500, 502, 503, 504):
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"jev {r.status}: {data}")
    raise RuntimeError("jev kept failing")


async def main(kind: str, src: Path, dst: Path, concurrency: int, limit: int | None):
    rows = [json.loads(line) for line in src.open()]
    if limit:
        rows = rows[:limit]
    semaphore = asyncio.Semaphore(concurrency)
    tokens = 0
    done = 0
    results: list[dict | None] = [None] * len(rows)

    async with aiohttp.ClientSession() as session:

        async def one(i: int, row: dict):
            nonlocal tokens, done
            state, questions = build(kind, row)
            async with semaphore:
                try:
                    data = await ask(session, {"model": "jev-latest", "state": state, "questions": questions})
                except Exception as e:
                    print(f"row {i}: {e}")
                    return
            tokens += data.get("usage", {}).get("input_tokens", 0)
            results[i] = {
                "state": state,
                "questions": questions,
                "gold": gold(questions, data["answers"]),
                "meta": {k: v for k, v in row.items() if k not in ("message", "earlier", "candidate", "owned")},
            }
            done += 1
            if done % 250 == 0:
                print(f"{done}/{len(rows)} labelled, {tokens:,} input tokens", flush=True)

        await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = sum(r is not None for r in results)
    print(f"wrote {kept} rows to {dst}; {tokens:,} input tokens (~${tokens * 0.042 / 1e6:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["route", "same_release"])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.kind, args.src, args.dst, args.concurrency, args.limit))
