"""
Usage:
  python -m eval.groundedness build <eval.jsonl> <out.jsonl> [--per-category 5] [--seed 7]
  python -m eval.groundedness check <rubric.jsonl> [<rubric.jsonl> ...]

Does the generated root cause and fix state things the evidence doesn't?

build  Picks a fixed set of held-out lines (N per alert category, plus the
       N least confident), runs each through the real pipeline, and saves the
       log line, the evidence the root-cause step saw, and the generated text.

check  Splits the root cause and fix into sentences and flags a sentence when
       it names a SPECIFIC that appears nowhere in the context the model was
       given (log line, category definition, evidence templates):
         - file or directory paths      /etc/lustre/lustre.conf
         - file names with extensions   params.lustre, mtio.conf
         - numbers with 2+ digits       (port, size, count, error code)
         - IPs and hex values
         - code-like identifiers        CamelCase, snake_case, ALLCAPS (3+)
         - host and device names        bglio318, sda5, eth0
       Sentences with no specific are counted as GENERIC (advice like "check
       the network configuration"): not wrong, but not evidence-based either.

The specifics check catches the failure seen in hand reading (invented paths,
config files, acronym expansions with capitals). It cannot judge whether a
general claim is true; an LLM judge and hand checks cover that.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATH = re.compile(r"(?<![\w.])/(?:[\w.<>*-]+/)*[\w.<>*-]+")
FILE = re.compile(r"\b[\w-]+\.(?:conf|cfg|ini|log|xml|yaml|yml|json|sh|py|map|mapfile|lustre|rts|so|ko|d)\b", re.I)
NUM = re.compile(r"\b\d{2,}\b")
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
IDENT = re.compile(r"\b(?:[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*|\w+_\w+|[A-Z]{3,}\d*)\b")
ALNUM = re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b")   # host and device names: bglio318, sda5, eth0
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Common words that the IDENT pattern would otherwise flag.
ALLOWED = {"RAS", "CPU", "TLB", "DDR", "API", "LLM", "FATAL", "ERROR", "INFO", "WARNING", "SEVERE", "FAILURE",
           "KERNEL", "APP", "BGL", "NOTE", "IMPORTANT", "JSON", "TCP", "UDP", "DNS", "MPI", "RAM", "IO", "NFS"}


def specifics(sentence: str) -> set[str]:
    found = set()
    for pat in (PATH, FILE, IP, HEX, NUM, IDENT, ALNUM):
        for m in pat.finditer(sentence):
            tok = m.group(0).strip(".,;:()'\"")
            if tok and tok not in ALLOWED:
                found.add(tok)
    # a path also matches its parts; keep the longest form only
    return {t for t in found if not any(t != o and t in o for o in found)}


def supported(token: str, context: str) -> bool:
    t = token.lower().rstrip("/")
    if t in context:
        return True
    # allow masked forms: "point /p/gb1" vs template "/p/gb<NUM>"
    stem = re.sub(r"\d+", "", t)
    return len(stem) >= 4 and stem in re.sub(r"<num>|<\*>|\d+", "", context)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text or "") if len(s.strip()) > 3]


def check_one(d: dict) -> list[dict]:
    context = " ".join([d["text"], d.get("definition", ""), *d.get("evidence", [])]).lower()
    out = []
    for field in ("root_cause", "remediation"):
        for s in sentences(d.get(field, "")):
            spec = specifics(s)
            bad = sorted(t for t in spec if not supported(t, context))
            out.append({"field": field, "sentence": s, "specifics": sorted(spec), "unsupported": bad,
                        "verdict": "unsupported" if bad else "grounded" if spec else "generic"})
    return out


# --- build -------------------------------------------------------------------------

def build(eval_path: str, out_path: str, per_category: int, seed: int) -> None:
    import config
    from agents.graph import build_graph, select_evidence
    from agents.llm import start_recording
    from bgl import categories

    config.require_all()
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")]
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        if r["true_label"] != "normal":
            by_cat[r["true_label"]].append(r)
    picked = []
    for cat in sorted(by_cat):
        picked += rng.sample(by_cat[cat], min(per_category, len(by_cat[cat])))
    graph = build_graph()
    out = []
    for i, r in enumerate(picked, 1):
        calls = start_recording()
        s = graph.invoke({"raw_log": r["raw_log"], "notes": []})
        if s.get("route") != "full":
            continue
        ev = select_evidence(s.get("retrieved", []), s.get("classification"))
        cat = categories.get(s.get("classification", "")) or {}
        out.append({
            "line_no": r["line_no"], "true_label": r["true_label"], "classification": s["classification"],
            "text": s["text"], "definition": cat.get("definition", ""),
            "evidence": [e["template"] for e in ([ev["primary"]] if ev["primary"] else []) + ev["supporting"]],
            "root_cause": s.get("root_cause", ""), "remediation": s.get("remediation", ""),
            "providers": sorted({c["provider"] for c in calls}),
        })
        print(f"  {i}/{len(picked)} {r['true_label']}", file=sys.stderr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(d) + "\n" for d in out)
    print(f"saved {len(out)} diagnoses to {out_path}")


# --- check -------------------------------------------------------------------------

def report(path: str) -> dict:
    ds = [json.loads(l) for l in open(path, encoding="utf-8")]
    sents = [dict(s, line_no=d["line_no"]) for d in ds for s in check_one(d)]
    v = Counter(s["verdict"] for s in sents)
    any_bad = sum(any(s["verdict"] == "unsupported" for s in check_one(d)) for d in ds)
    n = len(sents)
    print(f"\n{path}: {len(ds)} diagnoses, {n} sentences")
    for k in ("grounded", "generic", "unsupported"):
        print(f"  {k:12} {v[k]:4}  {v[k] / n:.1%}")
    print(f"  diagnoses with at least one unsupported specific: {any_bad}/{len(ds)} = {any_bad / len(ds):.1%}")
    tokens = Counter(t for s in sents for t in s["unsupported"])
    print("  most common unsupported specifics:", ", ".join(f"{t} x{c}" for t, c in tokens.most_common(10)))
    return {"diagnoses": len(ds), "sentences": n, **v, "diagnoses_with_unsupported": any_bad}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("eval_file")
    b.add_argument("out")
    b.add_argument("--per-category", type=int, default=5)
    b.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.eval_file, a.out, a.per_category, a.seed)
    else:
        for f in a.files:
            report(f)


if __name__ == "__main__":
    main()
