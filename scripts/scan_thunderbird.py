"""
Usage: python -I scripts/scan_thunderbird.py [max_lines] [out.json]

Profiles a prefix of the Loghub Thunderbird log [1, 2] by streaming it straight
from Zenodo: the 2 GB .tar.gz is read over HTTP and decompressed in memory, and
the connection is closed after `max_lines` lines. Nothing is written to disk
except the small JSON summary.

Thunderbird lines start with a label like BGL ('-' = normal, otherwise an
alert tag), then: unix time, date, host, month, day, time, host/facility,
message.
"""
import json
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict

URL = "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1"
MASK = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b|\d+")

max_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
out = sys.argv[2] if len(sys.argv) > 2 else None

labels, first, last = Counter(), {}, {}
keys = defaultdict(Counter)
examples = {}
t0 = time.time()
with urllib.request.urlopen(URL, timeout=60) as resp:
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            print("member:", member.name, f"{member.size / 1e9:.1f} GB")
            f = tar.extractfile(member)
            for n, raw in enumerate(f, 1):
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                parts = line.split(" ", 8)
                lab = parts[0]
                labels[lab] += 1
                first.setdefault(lab, n)
                last[lab] = n
                msg = parts[8] if len(parts) > 8 else ""
                if len(keys[lab]) < 5000 or msg in keys[lab]:
                    keys[lab][MASK.sub("<N>", msg)[:160]] += 1
                examples.setdefault(lab, line[:220])
                if n % 1_000_000 == 0:
                    print(f"  {n:,} lines ({time.time() - t0:.0f}s)", file=sys.stderr)
                if n >= max_lines:
                    break
            break

alerts = sum(c for l, c in labels.items() if l != "-")
print(f"\n{n:,} lines read in {time.time() - t0:.0f}s; {alerts:,} alert lines ({alerts / n:.2%}); {len(labels)} labels")
for lab, c in labels.most_common():
    top = ", ".join(f"{m[:60]!r} x{k}" for m, k in keys[lab].most_common(2))
    print(f"{lab:>10} {c:>9,}  lines {first[lab]:,}-{last[lab]:,}  patterns {len(keys[lab])}  {top}")
print("\nexamples:")
for lab in list(labels)[:12]:
    print("  ", examples[lab])
if out:
    json.dump({"lines": n, "labels": labels, "first": first, "last": last,
               "patterns": {l: len(k) for l, k in keys.items()}, "examples": examples},
              open(out, "w", encoding="utf-8"), indent=1)
