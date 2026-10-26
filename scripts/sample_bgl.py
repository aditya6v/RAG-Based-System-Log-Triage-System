"""
Usage: python -I scripts/sample_bgl.py <path/to/BGL.log> [--seed 7]

One streaming pass over BGL.log (never loaded into memory), then writes:
  sample_data/bgl_index.log  -- raw BGL lines (label kept) to ingest + embed
  eval/bgl_eval.jsonl        -- held-out lines with the label stripped off

How the split works:
  * Every line gets a "pattern key": its category plus the component, level
    and message with numbers/hex masked. Lines with the same key are the same
    kind of event.
  * Some keys are held out entirely (about 10% of normal keys, about 20% of
    alert keys per category, never a category's most common key). Lines with
    those keys go only to the eval's "unseen" subset, to test events whose
    exact pattern was never ingested.
  * The "seen" eval subset is a random, frequency-weighted sample per
    category (reservoir sampling), so it reflects what the log really
    contains. Half of the normal lines are hard negatives: normal lines
    logged at FATAL/ERROR/SEVERE/FAILURE.
  * The index gets up to 300 lines per alert category, 2,000 normal lines,
    plus one example of every non-held-out pattern so retrieval has seen
    each kind of event at least once.
  * No line number appears in both the index and the eval (asserted).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # -I drops the script dir; we only import our own code

from bgl.categories import category_for_code  # noqa: E402
from bgl.lines import has_message, level_of, split_line  # noqa: E402

MASK = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b|\d+")
HARD_LEVELS = {"FATAL", "ERROR", "SEVERE", "FAILURE"}

RESERVOIR = {"alert": 600, "normal_hard": 2000, "normal_soft": 2000}
EVAL_SEEN = {"alert": 40, "normal_hard": 100, "normal_soft": 100}
INDEX = {"alert": 300, "normal_hard": 1000, "normal_soft": 1000}
EVAL_UNSEEN_NORMAL = 100
UNSEEN_LINES_PER_ALERT_KEY = 3


def reservoir_add(bucket: list, size: int, seen: int, item, rng: random.Random):
    """Classic reservoir sampling: after `seen` items, each has equal odds of
    being kept. `seen` counts this item too."""
    if len(bucket) < size:
        bucket.append(item)
    else:
        j = rng.randrange(seen)
        if j < size:
            bucket[j] = item


def kind(stratum: str) -> str:
    return stratum if stratum.startswith("normal") else "alert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bgl_log")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    strata = defaultdict(list)  # stratum -> reservoir of (line_no, line, key)
    strata_seen = Counter()
    keys = {}  # key -> {"stratum", "count", "examples": reservoir of (line_no, line)}
    skipped_no_msg = skipped_unknown = 0

    with open(args.bgl_log, "rb") as f:
        for n, raw in enumerate(f, 1):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            sp = split_line(line)
            if sp["label"] is None:
                continue
            if not has_message(sp["content"]):
                skipped_no_msg += 1
                continue
            cat = category_for_code(sp["label"])
            if cat is None:
                skipped_unknown += 1
                continue
            if cat == "normal":
                stratum = "normal_hard" if level_of(sp["content"]) in HARD_LEVELS else "normal_soft"
            else:
                stratum = cat
            key = stratum + "|" + MASK.sub("<N>", sp["content"])[:200]

            strata_seen[stratum] += 1
            reservoir_add(strata[stratum], RESERVOIR[kind(stratum)], strata_seen[stratum], (n, line, key), rng)

            k = keys.setdefault(key, {"stratum": stratum, "count": 0, "examples": []})
            k["count"] += 1
            reservoir_add(k["examples"], UNSEEN_LINES_PER_ALERT_KEY, k["count"], (n, line), rng)

    # --- choose held-out keys -------------------------------------------------
    by_cat = defaultdict(list)
    for key, k in keys.items():
        by_cat["normal" if k["stratum"].startswith("normal") else k["stratum"]].append(key)
    held_out = set()
    for cat, ks in sorted(by_cat.items()):
        ks.sort(key=lambda x: (-keys[x]["count"], x))
        candidates = ks[1:]  # never hold out the most common pattern
        share = 0.10 if cat == "normal" else 0.20
        n_hold = round(len(ks) * share) if len(ks) >= 3 else 0
        held_out.update(rng.sample(candidates, min(n_hold, len(candidates))))

    # --- eval -------------------------------------------------------------------
    eval_rows, used = [], set()

    def add_eval(n, line, subset):
        sp = split_line(line)
        cat = category_for_code(sp["label"])
        eval_rows.append({
            "line_no": n,
            "raw_log": sp["text"],
            "true_label": cat,
            "code": sp["label"],
            "subset": subset,
            "hard_negative": cat == "normal" and level_of(sp["content"]) in HARD_LEVELS,
        })
        used.add(n)

    for stratum, bucket in sorted(strata.items()):
        pool = [it for it in bucket if it[2] not in held_out]
        rng.shuffle(pool)
        want = EVAL_SEEN[kind(stratum)]
        if kind(stratum) == "alert" and strata_seen[stratum] < RESERVOIR["alert"]:
            want = min(want, max(1, round(len(pool) * 0.15)))  # small category: ~85/15 split
        for n, line, _ in pool[:want]:
            add_eval(n, line, "seen")

    normal_held = sorted(k for k in held_out if keys[k]["stratum"].startswith("normal"))
    for key in rng.sample(normal_held, min(EVAL_UNSEEN_NORMAL, len(normal_held))):
        n, line = keys[key]["examples"][0]
        add_eval(n, line, "unseen")
    for key in sorted(k for k in held_out if not keys[k]["stratum"].startswith("normal")):
        for n, line in keys[key]["examples"]:
            add_eval(n, line, "unseen")

    # --- index ------------------------------------------------------------------
    index = {}
    for stratum, bucket in sorted(strata.items()):
        pool = [it for it in bucket if it[2] not in held_out and it[0] not in used]
        for n, line, _ in pool[:INDEX[kind(stratum)]]:
            index[n] = line
    for key, k in keys.items():
        if key in held_out:
            continue
        for n, line in k["examples"]:
            if n not in used:
                index.setdefault(n, line)
                break

    overlap = used & set(index)
    assert not overlap, f"{len(overlap)} lines in both index and eval"
    index_keys = {keys_key for keys_key in keys if keys_key not in held_out}
    assert not (held_out & index_keys)

    # --- write ------------------------------------------------------------------
    idx_path = ROOT / "sample_data" / "bgl_index.log"
    eval_path = ROOT / "eval" / "bgl_eval.jsonl"
    with open(idx_path, "w", encoding="utf-8", newline="\n") as f:
        for n in sorted(index):
            f.write(index[n] + "\n")
    eval_rows.sort(key=lambda r: r["line_no"])
    with open(eval_path, "w", encoding="utf-8", newline="\n") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    # --- report -----------------------------------------------------------------
    idx_cats = Counter(category_for_code(split_line(l)["label"]) for l in index.values())
    ev = Counter((r["true_label"], r["subset"]) for r in eval_rows)
    print(f"skipped: {skipped_no_msg:,} lines with no message, {skipped_unknown} unknown codes")
    print(f"distinct patterns: {len(keys):,}  held out: {len(held_out):,}")
    print(f"index: {len(index):,} lines -> {idx_path.relative_to(ROOT)}")
    print(f"eval:  {len(eval_rows):,} lines -> {eval_path.relative_to(ROOT)}  "
          f"(hard negatives: {sum(r['hard_negative'] for r in eval_rows)})")
    print(f"\n{'category':28} {'index':>6} {'eval seen':>10} {'eval unseen':>12}")
    for cat in sorted(set(idx_cats) | {c for c, _ in ev}):
        print(f"{cat:28} {idx_cats[cat]:>6} {ev[(cat, 'seen')]:>10} {ev[(cat, 'unseen')]:>12}")
    print("\nno line in both index and eval: OK")


if __name__ == "__main__":
    main()
