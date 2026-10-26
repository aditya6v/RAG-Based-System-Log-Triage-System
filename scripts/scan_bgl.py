"""Stream-scan BGL.log once. Usage: python -I scripts/scan_bgl.py <BGL.log> <out.json>"""
import json
import re
import sys
from collections import Counter, defaultdict

MASK = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b|\d+")
MAX_DISTINCT = 20000  # per label, caps memory on the normal class

path, out = sys.argv[1], sys.argv[2]
total = bad_format = decode_err = 0
labels = Counter()
first_line = {}
last_line = {}
levels = defaultdict(Counter)
components = defaultdict(Counter)
msgs = defaultdict(Counter)
msg_overflow = Counter()
days = defaultdict(set)

with open(path, "rb") as f:
    for n, raw in enumerate(f, 1):
        total += 1
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            decode_err += 1
            line = raw.decode("utf-8", "replace")
        parts = line.rstrip("\r\n").split(" ", 9)
        if len(parts) < 10:
            bad_format += 1
            lab = parts[0] if parts else ""
            labels[lab] += 1
            continue
        lab, _ts, day, _node, _t, _node2, _typ, comp, lvl, msg = parts
        labels[lab] += 1
        first_line.setdefault(lab, n)
        last_line[lab] = n
        levels[lab][lvl] += 1
        components[lab][comp] += 1
        days[lab].add(day)
        key = MASK.sub("<N>", msg)[:160]
        if key in msgs[lab] or len(msgs[lab]) < MAX_DISTINCT:
            msgs[lab][key] += 1
        else:
            msg_overflow[lab] += 1
        if n % 1_000_000 == 0:
            print(f"  {n:,} lines...", file=sys.stderr)

report = {
    "total_lines": total,
    "bad_format_lines": bad_format,
    "decode_error_lines": decode_err,
    "labels": {
        lab: {
            "count": c,
            "first_line": first_line.get(lab),
            "last_line": last_line.get(lab),
            "distinct_days": len(days[lab]),
            "levels": dict(levels[lab].most_common()),
            "components": dict(components[lab].most_common(5)),
            "distinct_masked_msgs": len(msgs[lab]),
            "distinct_capped": msg_overflow[lab] > 0,
            "top_msgs": msgs[lab].most_common(4),
        }
        for lab, c in labels.most_common()
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print(f"done: {total:,} lines, {len(labels)} labels -> {out}")
