"""
Usage: python -I scripts/sample_thunderbird.py [--archive data/tbird/Thunderbird.tar.gz] [--seed 7]

One pass over the full Loghub Thunderbird log [1, 2], decompressed in memory
(the 30 GB log is never written to disk). With --archive it reads a local
copy of the 2 GB .tar.gz; without it, it streams from Zenodo. Streaming
failed once with "unexpected end of data" (the connection dropped), and a
gzip stream can't resume, so a resumable download first is more reliable:
  curl -L -C - --retry 20 -o data/tbird/Thunderbird.tar.gz "<URL below>" Same time-ordered split as scripts/sample_bgl_time.py:
the first 60% of lines is the INDEX period, the next 10% VALIDATION, the last
30% TEST. Line position is time order. Loghub states 211,212,192 lines; the
actual count is printed at the end.

Writes (raw alert codes kept; grouping codes into categories is a separate
step, decided from the counts this script prints):
  sample_data/tbird_index.log    index lines, label kept
  eval/tbird_val.raw.jsonl       validation lines with their code
  eval/tbird_test.raw.jsonl      test lines with their code
  eval/results/tbird_scan.json   counts per code and period, first/last line

Thunderbird line: label, unix time, date, host, month, day, time, source,
message. The message (from the 9th field) is what gets templated. A normal
line counts as a hard negative if its message contains error-like words
(error, fail, fatal, critical, panic, abort): Thunderbird has no level field.
"""
import argparse
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1"
STATED_TOTAL = 211_212_192
CUTS = (0.60, 0.70)
MASK = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b|\d+")
HARD = re.compile(r"\b(error|fail\w*|fatal|critical|panic|abort\w*)\b", re.I)
RESERVOIR = {"index": {"alert": 600, "normal_hard": 2000, "normal_soft": 2000},
             "val": {"alert": 30, "normal_hard": 100, "normal_soft": 100},
             "test": {"alert": 50, "normal_hard": 150, "normal_soft": 150}}
INDEX_CAP = {"alert": 300, "normal_hard": 1000, "normal_soft": 1000}
MAX_TRACKED_KEYS = 300_000      # index-period patterns tracked for coverage / seen flags
COVERAGE_KEYS = 3_000           # most frequent normal patterns added to the index


def reservoir_add(bucket, size, seen, item, rng):
    if len(bucket) < size:
        bucket.append(item)
    else:
        j = rng.randrange(seen)
        if j < size:
            bucket[j] = item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-lines", type=int, default=0, help="stop early (for testing the script)")
    ap.add_argument("--archive", help="local Thunderbird.tar.gz; streams from Zenodo if omitted")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    total = args.max_lines or STATED_TOTAL
    cut1, cut2 = int(total * CUTS[0]), int(total * CUTS[1])

    res = {p: defaultdict(list) for p in ("index", "val", "test")}    # period -> stratum -> reservoir
    seen_n = {p: Counter() for p in res}
    codes = {p: Counter() for p in res}
    first, last = {}, {}
    keys = {}                     # index-period key -> [count, example (n, line)]
    alert_keys = set()
    t0, n = time.time(), 0
    source = open(args.archive, "rb") if args.archive else urllib.request.urlopen(URL, timeout=120)
    with source as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            for n, raw in enumerate(tar.extractfile(member), 1):
                period = "index" if n <= cut1 else "val" if n <= cut2 else "test"
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                parts = line.split(" ", 8)
                if len(parts) < 9 or not parts[8].strip():
                    continue
                code, msg = parts[0], parts[8]
                codes[period][code] += 1
                first.setdefault(code, n)
                last[code] = n
                stratum = code if code != "-" else ("normal_hard" if HARD.search(msg) else "normal_soft")
                kind = "alert" if code != "-" else stratum
                key = code + "|" + MASK.sub("<N>", msg)[:200]
                seen_n[period][stratum] += 1
                reservoir_add(res[period][stratum], RESERVOIR[period][kind], seen_n[period][stratum], (n, line, key), rng)
                if period == "index":
                    k = keys.get(key)
                    if k is not None:
                        k[0] += 1
                    elif len(keys) < MAX_TRACKED_KEYS:
                        keys[key] = [1, (n, line)]
                    if code != "-":
                        alert_keys.add(key)
                if n % 5_000_000 == 0:
                    print(f"  {n:,} lines, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
                if args.max_lines and n >= args.max_lines:
                    break
            break

    # index: capped reservoirs + one example of every alert pattern and of the most frequent normal patterns
    index = {}
    for stratum, bucket in res["index"].items():
        cap = INDEX_CAP["alert" if not stratum.startswith("normal") else stratum]
        for ln, line, _ in bucket[:cap]:
            index[ln] = line
    normal_keys = sorted((k for k in keys if k not in alert_keys), key=lambda k: -keys[k][0])[:COVERAGE_KEYS]
    for k in list(alert_keys) + normal_keys:
        ln, line = keys[k][1]
        index.setdefault(ln, line)

    def rows(period):
        out = []
        for stratum, bucket in res[period].items():
            for ln, line, key in bucket:
                code, msg = line.split(" ", 1)[0], line.split(" ", 8)[8]
                out.append({"line_no": ln, "raw_log": line.split(" ", 1)[1], "code": code, "period": period,
                            "subset": "seen" if key in keys else "unseen",
                            "hard_negative": code == "-" and bool(HARD.search(msg))})
        return sorted(out, key=lambda r: r["line_no"])

    val, test = rows("val"), rows("test")
    (ROOT / "sample_data" / "tbird_index.log").write_text("".join(index[k] + "\n" for k in sorted(index)),
                                                          encoding="utf-8", newline="\n")
    for name, rs in (("val", val), ("test", test)):
        with open(ROOT / "eval" / f"tbird_{name}.raw.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    scan = {"lines": n, "stated_total": STATED_TOTAL, "cuts": [cut1, cut2], "seconds": round(time.time() - t0),
            "codes_by_period": {p: dict(c.most_common()) for p, c in codes.items()},
            "first_line": first, "last_line": last, "index_patterns_tracked": len(keys),
            "index_lines": len(index), "val_lines": len(val), "test_lines": len(test)}
    (ROOT / "eval" / "results").mkdir(exist_ok=True)
    (ROOT / "eval" / "results" / "tbird_scan.json").write_text(json.dumps(scan, indent=1), encoding="utf-8")

    print(f"{n:,} lines in {time.time() - t0:.0f}s (Loghub states {STATED_TOTAL:,}); cuts at {cut1:,} / {cut2:,}")
    print(f"index {len(index):,} lines ({len(keys):,} patterns tracked); val {len(val)}; test {len(test)}")
    print(f"\n{'code':>12} {'index':>11} {'val':>10} {'test':>10}  first-last line")
    for code in sorted(first, key=lambda c: -sum(codes[p][c] for p in codes)):
        print(f"{code:>12} {codes['index'][code]:>11,} {codes['val'][code]:>10,} {codes['test'][code]:>10,}  "
              f"{first[code]:,}-{last[code]:,}")


if __name__ == "__main__":
    main()
