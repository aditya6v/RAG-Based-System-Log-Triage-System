"""
Usage: python -I scripts/sample_bgl_time.py <path/to/BGL.log> [--seed 7]

Time-ordered split of BGL. The log is in time order, so line position is
time. The first 60% of lines (2005-06-03 to 2005-07-28) is the INDEX period,
the next 10% (to 2005-09-02) VALIDATION, the last 30% (to 2006-01-04) TEST.
Thresholds may be tuned on validation only; test is used once.

Writes:
  sample_data/bgl_time_index.log   lines to ingest (label kept)
  eval/bgl_time_val.jsonl          validation lines (label stripped)
  eval/bgl_time_test.jsonl         test lines (label stripped)

Sampling inside each period works like scripts/sample_bgl.py: a reservoir
sample per category (so frequent messages stay frequent), half of the normal
lines being hard negatives, and the index also gets one example of every
message pattern seen in the index period. A val/test line is "seen" if its
pattern (category + component + level + message with numbers masked)
occurred anywhere in the index period, otherwise "unseen".

Known in advance (from a scan): filesystem mount failures never occur in the
index period, so no retrieval-based classifier can name them on this split.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bgl.categories import category_for_code  # noqa: E402
from bgl.lines import has_message, level_of, split_line  # noqa: E402
from sample_bgl import HARD_LEVELS, MASK, reservoir_add  # noqa: E402

CUTS = (0.60, 0.70)
PERIODS = ("index", "val", "test")
RESERVOIR = {"index": {"alert": 600, "normal_hard": 2000, "normal_soft": 2000},
             "val": {"alert": 30, "normal_hard": 100, "normal_soft": 100},
             "test": {"alert": 50, "normal_hard": 150, "normal_soft": 150}}
INDEX_CAP = {"alert": 300, "normal_hard": 1000, "normal_soft": 1000}


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def kind(stratum: str) -> str:
    return stratum if stratum.startswith("normal") else "alert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bgl_log")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    total = count_lines(args.bgl_log)
    cut1, cut2 = int(total * CUTS[0]), int(total * CUTS[1])

    res = {p: defaultdict(list) for p in PERIODS}       # period -> stratum -> reservoir
    seen_n = {p: Counter() for p in PERIODS}
    index_keys = {}                                      # key -> reservoir of 1 example
    first_date, last_date = {}, {}

    with open(args.bgl_log, "rb") as f:
        for n, raw in enumerate(f, 1):
            period = "index" if n <= cut1 else "val" if n <= cut2 else "test"
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            sp = split_line(line)
            if sp["label"] is None or not has_message(sp["content"]):
                continue
            cat = category_for_code(sp["label"])
            if cat is None:
                continue
            day = sp["text"].split(" ")[1]
            first_date.setdefault(period, day)
            last_date[period] = day
            stratum = cat if cat != "normal" else (
                "normal_hard" if level_of(sp["content"]) in HARD_LEVELS else "normal_soft")
            key = stratum + "|" + MASK.sub("<N>", sp["content"])[:200]
            seen_n[period][stratum] += 1
            reservoir_add(res[period][stratum], RESERVOIR[period][kind(stratum)], seen_n[period][stratum],
                          (n, line, key), rng)
            if period == "index":
                k = index_keys.setdefault(key, {"count": 0, "ex": []})
                k["count"] += 1
                reservoir_add(k["ex"], 1, k["count"], (n, line), rng)

    # index: capped reservoirs + one example of every pattern in the index period
    index = {}
    for stratum, bucket in res["index"].items():
        for n, line, _ in bucket[:INDEX_CAP[kind(stratum)]]:
            index[n] = line
    for k in index_keys.values():
        n, line = k["ex"][0]
        index.setdefault(n, line)

    def rows(period):
        out = []
        for stratum, bucket in sorted(res[period].items()):
            for n, line, key in bucket:
                sp = split_line(line)
                cat = category_for_code(sp["label"])
                out.append({"line_no": n, "raw_log": sp["text"], "true_label": cat, "code": sp["label"],
                            "subset": "seen" if key in index_keys else "unseen", "period": period,
                            "hard_negative": cat == "normal" and level_of(sp["content"]) in HARD_LEVELS})
        return sorted(out, key=lambda r: r["line_no"])

    val, test = rows("val"), rows("test")
    assert max(index) <= cut1 < min(r["line_no"] for r in val), "index must come before validation"
    assert max(r["line_no"] for r in val) <= cut2 < min(r["line_no"] for r in test), "validation must come before test"

    (ROOT / "sample_data" / "bgl_time_index.log").write_text(
        "".join(index[n] + "\n" for n in sorted(index)), encoding="utf-8", newline="\n")
    for name, rs in (("val", val), ("test", test)):
        with open(ROOT / "eval" / f"bgl_time_{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)

    print(f"{total:,} lines; cuts after line {cut1:,} and {cut2:,}")
    for p in PERIODS:
        print(f"  {p:5} {first_date.get(p)} to {last_date.get(p)}")
    idx_cats = Counter(category_for_code(split_line(l)["label"]) for l in index.values())
    vc = Counter((r["true_label"], r["subset"]) for r in val)
    tc = Counter((r["true_label"], r["subset"]) for r in test)
    print(f"\nindex {len(index):,} lines, {len(index_keys):,} patterns; val {len(val)}; test {len(test)} "
          f"(hard negatives: val {sum(r['hard_negative'] for r in val)}, test {sum(r['hard_negative'] for r in test)})")
    print(f"\n{'category':26} {'index':>6} {'val seen':>9} {'val new':>8} {'test seen':>10} {'test new':>9}")
    cats = sorted(set(idx_cats) | {c for c, _ in vc} | {c for c, _ in tc})
    for c in cats:
        print(f"{c:26} {idx_cats[c]:>6} {vc[(c, 'seen')]:>9} {vc[(c, 'unseen')]:>8} {tc[(c, 'seen')]:>10} {tc[(c, 'unseen')]:>9}")


if __name__ == "__main__":
    main()
