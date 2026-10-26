"""Second pass: normal FATAL/ERROR messages, bad-format lines, and text shared
between normal and alert labels. Usage: python -I scripts/scan_bgl_overlap.py <BGL.log>"""
import re
import sys
from collections import Counter, defaultdict

MASK = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,}\b|\d+")
normal_bad_lvl = defaultdict(Counter)
bad_fmt = Counter()
bad_fmt_examples = []
msg_labels = defaultdict(Counter)  # masked msg -> label counts (alert msgs only tracked fully)
alert_keys = set()
lines_cache = []

with open(sys.argv[1], "rb") as f:
    for raw in f:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        parts = line.split(" ", 9)
        if len(parts) < 10:
            bad_fmt[parts[0]] += 1
            if len(bad_fmt_examples) < 5:
                bad_fmt_examples.append(line[:160])
            continue
        lab, lvl, msg = parts[0], parts[8], parts[9]
        key = MASK.sub("<N>", msg)[:160]
        if lab != "-":
            alert_keys.add(key)
        elif lvl in ("FATAL", "ERROR", "SEVERE", "FAILURE"):
            if key in normal_bad_lvl[lvl] or len(normal_bad_lvl[lvl]) < 5000:
                normal_bad_lvl[lvl][key] += 1
        msg_labels[key][lab] += 1 if (lab != "-" or key in alert_keys) else 0

print("bad-format lines by label:", dict(bad_fmt.most_common(8)))
for e in bad_fmt_examples:
    print("   ", e)
for lvl, c in normal_bad_lvl.items():
    print(f"\nnormal-labelled {lvl}: top messages")
    for m, n in c.most_common(6):
        print(f"  {n:>8,}  {m[:110]}")

# Second look for overlap needs normal counts for alert keys seen before the
# first alert occurrence too, so re-stream counting only alert keys.
overlap = Counter()
with open(sys.argv[1], "rb") as f:
    for raw in f:
        parts = raw.decode("utf-8", "replace").rstrip("\r\n").split(" ", 9)
        if len(parts) == 10 and parts[0] == "-":
            key = MASK.sub("<N>", parts[9])[:160]
            if key in alert_keys:
                overlap[key] += 1
print(f"\nalert message texts that ALSO appear as normal: {len(overlap)} of {len(alert_keys)}")
for m, n in overlap.most_common(10):
    al = {k: v for k, v in msg_labels[m].items() if k != "-"}
    print(f"  normal={n:>7,} alert={al}  {m[:90]}")
