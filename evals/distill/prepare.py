"""
Merge labelled distillation rows into laya-train's train/eval files.

Holds out 10% of each source (stratified by the category the message was
generated for, or the pair's similarity band) as eval.jsonl: Jev-labelled,
never trained on, so `laya-train --eval` reports teacher agreement before and
after. Messages that appear in evals/decision_eval.py are dropped from both,
so the hand-labelled eval stays untouched.

    uv run python evals/distill/prepare.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decision_eval import FOLLOWUP_CASES, ROUTE_CASES  # noqa: E402

ROOT = Path("data/distill")
SOURCES = {"route": ROOT / "route.jsonl", "same_release": ROOT / "same_release.jsonl"}
HOLDOUT = 0.1


def main():
    random.seed(20261008)
    reserved = {m.casefold().strip() for m, *_ in ROUTE_CASES} | {
        m.casefold().strip() for m, *_ in FOLLOWUP_CASES
    }
    train, held = [], []
    for source, path in SOURCES.items():
        if not path.exists():
            print(f"skipping {source}: {path} missing")
            continue
        groups: dict[str, list[dict]] = defaultdict(list)
        seen = set()
        for line in path.open():
            row = json.loads(line)
            key = json.dumps(row["state"], sort_keys=True)
            message = str(row["state"].get("message", "")).casefold().strip()
            if key in seen or message in reserved:
                continue
            seen.add(key)
            meta = row.pop("meta", {})
            groups[meta.get("category") or meta.get("band") or source].append(row)
        for name, rows in groups.items():
            random.shuffle(rows)
            cut = max(1, int(len(rows) * HOLDOUT))
            held += rows[:cut]
            train += rows[cut:]
            print(f"{source}/{name}: {len(rows) - cut} train, {cut} held out")
    random.shuffle(train)
    random.shuffle(held)
    for name, rows in (("train", train), ("eval", held)):
        with (ROOT / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train {len(train)}, eval {len(held)}")


if __name__ == "__main__":
    main()
