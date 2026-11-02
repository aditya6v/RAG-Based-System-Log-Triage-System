"""
Usage:
  python -m eval.evaluate eval/bgl_eval.jsonl [--limit N] [--no-llm]
  python -m eval.evaluate --report eval/results/<run>.json

Runs the classification step on the held-out BGL eval set (built by
scripts/sample_bgl.py; none of these lines are in the retrieval index) and
compares four classifiers on exactly the same retrieval results:

  level_rule  FATAL/FAILURE means alert (binary only, no retrieval)
  knn_vote    similarity-weighted vote of the retrieved incidents' labels
  llm_only    the LLM with just the category definitions
  llm_rag     the LLM shown the retrieved incidents and their labels
  triage      what the app does: knn_vote if confident, else llm_rag

Every example's predictions are saved to eval/results/<timestamp>.json, and a
markdown report is written next to it. --report rebuilds the report from a
saved run without calling anything again.

Reports: per-class precision/recall/F1, macro and weighted averages, a
confusion matrix, binary alert-vs-normal scores (so level_rule can be
compared), and the same split by subset: "seen" (pattern was ingested),
"unseen" (exact pattern held out of the index) and hard negatives (normal
lines logged at FATAL/ERROR/SEVERE/FAILURE).

Slow part: two LLM calls per example. About 600 examples on local Ollama is
tens of minutes; use --limit for a quick check or --no-llm for the
retrieval-only classifiers.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from bgl import categories

RESULTS_DIR = Path(__file__).with_name("results")
MULTICLASS = ["knn_vote", "llm_only", "llm_rag", "triage"]


def _retry(fn, *args, tries: int = 3):
    """Transient failures (a dropped Qdrant connection, a memory hiccup while
    decompressing a response) shouldn't kill a 25-minute run. Real errors
    still surface after the last try."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retrying after {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))


def run(path: str, limit: int | None, use_llm: bool, resume: str | None = None) -> Path:
    import config
    from agents import classify, llm
    from agents.retrieval import retrieve

    config.require_all()
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if limit:
        examples = examples[:limit]

    # Every finished example is appended to a .partial.jsonl straight away, so
    # a crash loses at most one example; --resume continues from that file.
    RESULTS_DIR.mkdir(exist_ok=True)
    partial = Path(resume) if resume else RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.partial.jsonl"
    rows = []
    if partial.exists():
        rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"  resuming: {len(rows)} examples already done in {partial.name}", file=sys.stderr)
    done = {r["line_no"] for r in rows}
    t_start = time.perf_counter()
    for i, ex in enumerate(examples, 1):
        if ex["line_no"] in done:
            continue
        t0 = time.perf_counter()
        r = _retry(retrieve, ex["raw_log"])
        t_retrieval = time.perf_counter() - t0
        vote = classify.knn_vote(r["retrieved"])
        row = {
            **{k: ex[k] for k in ("line_no", "true_label", "code", "subset", "hard_negative")},
            "matched_cluster": r["cluster_id"],
            "top_score": vote["top_score"],
            "vote_share": vote["share"],
            "pred": {
                "level_rule": classify.level_rule(r["content"]),
                "knn_vote": vote["label"],
            },
            "seconds": {"retrieval": round(t_retrieval, 3)},
        }
        if use_llm:
            calls = llm.start_recording()
            t0 = time.perf_counter()
            row["pred"]["llm_only"] = _retry(classify.llm_classify, r["text"])
            row["seconds"]["llm_only"] = round(time.perf_counter() - t0, 3)
            t0 = time.perf_counter()
            row["pred"]["llm_rag"] = _retry(classify.llm_classify, r["text"], r["retrieved"])
            row["seconds"]["llm_rag"] = round(time.perf_counter() - t0, 3)
            row["llm_calls"] = calls  # which provider answered each call, tokens
        rows.append(row)
        with open(partial, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if i % 25 == 0 or i == len(examples):
            print(f"  {i}/{len(examples)}  ({time.perf_counter() - t_start:.0f}s)", file=sys.stderr)

    out = partial.with_name(partial.name.replace(".partial.jsonl", ".json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "eval_file": path,
            "n": len(rows),
            "used_llm": use_llm,
            "llm_provider_chain": llm.provider_chain(),
            "chat_model": ", ".join(sorted({c["model"] for r in rows for c in r.get("llm_calls", [])})) or None,
            "embed_model": config.OLLAMA_EMBED_MODEL if config.EMBED_PROVIDER == "ollama" else config.GEMINI_EMBED_MODEL,
            "qdrant_collection": config.QDRANT_COLLECTION,
            "total_seconds": round(sum(sum(r["seconds"].values()) for r in rows), 1),  # summed per example, so resumed runs add up
            "rows": rows,
        }, f, indent=1)
    return out


def _with_triage(rows: list[dict]) -> None:
    """triage = vote when confident, else llm_rag. Derived from saved fields so
    the confidence thresholds can be tuned without re-running the LLM."""
    from agents import classify
    for r in rows:
        if "llm_rag" not in r["pred"]:
            continue
        confident = classify.is_confident({"top_score": r["top_score"], "share": r["vote_share"]})
        r["pred"]["triage"] = r["pred"]["knn_vote"] if confident else r["pred"]["llm_rag"]
        r["triage_method"] = "vote" if confident else "llm"


def _binary(label: str) -> str:
    return "normal" if label == "normal" else "alert"


def _score_table(rows, classifiers, labels_fn=lambda x: x):
    out = ["| classifier | accuracy | macro F1 | weighted F1 |", "|---|---|---|---|"]
    for clf in classifiers:
        rs = [r for r in rows if clf in r["pred"]]
        if not rs:
            continue
        y = [labels_fn(r["true_label"]) for r in rs]
        p = [labels_fn(r["pred"][clf]) if clf != "level_rule" else r["pred"][clf] for r in rs]
        acc = sum(a == b for a, b in zip(y, p)) / len(y)
        _, _, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        _, _, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        out.append(f"| {clf} | {acc:.3f} | {mf:.3f} | {wf:.3f} |")
    return "\n".join(out)


def report(results_path: Path) -> str:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    _with_triage(rows)
    present = [c for c in MULTICLASS if any(c in r["pred"] for r in rows)]
    ids = categories.ids()
    md = [f"# Eval run `{results_path.name}`", ""]
    md.append(f"{data['n']} examples from `{data['eval_file']}`; chat model `{data['chat_model']}`, "
              f"embeddings `{data['embed_model']}`, collection `{data['qdrant_collection']}`; "
              f"total {data['total_seconds']}s.")
    from collections import Counter
    md.append("")
    md.append("Examples per subset: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["subset"] for r in rows).items()))
              + f", hard negatives={sum(r['hard_negative'] for r in rows)}.")

    md += ["", "## Multiclass (8 categories), all examples", "", _score_table(rows, present)]
    for subset in ("seen", "unseen"):
        rs = [r for r in rows if r["subset"] == subset]
        if rs:
            md += ["", f"### Subset: {subset} ({len(rs)} examples)", "", _score_table(rs, present)]

    md += ["", "## Binary: alert vs normal, all examples", "",
           _score_table(rows, ["level_rule"] + present, _binary)]
    hn = [r for r in rows if r["hard_negative"]]
    if hn:
        md += ["", f"### Hard negatives ({len(hn)} normal lines at FATAL/ERROR/SEVERE/FAILURE)", "",
               "Share correctly called normal:", ""]
        for clf in ["level_rule"] + present:
            rs = [r for r in hn if clf in r["pred"]]
            if rs:
                ok = sum((r["pred"][clf] if clf == "level_rule" else _binary(r["pred"][clf])) == "normal" for r in rs)
                md.append(f"- {clf}: {ok}/{len(rs)} = {ok / len(rs):.3f}")

    for clf in present:
        rs = [r for r in rows if clf in r["pred"]]
        y = [r["true_label"] for r in rs]
        p = [r["pred"][clf] for r in rs]
        labels = ids + (["unknown"] if "unknown" in p else [])
        md += ["", f"## Per-class report: {clf}", "", "```",
               classification_report(y, p, labels=labels, zero_division=0, digits=3), "```"]
        cm = confusion_matrix(y, p, labels=labels)
        short = [l[:10] for l in labels]
        md += ["", f"Confusion matrix ({clf}; rows = true, columns = predicted):", "", "```",
               " " * 26 + " ".join(f"{s:>10}" for s in short)]
        for lab, row in zip(labels, cm):
            md.append(f"{lab:26}" + " ".join(f"{v:>10}" for v in row))
        md.append("```")

    if any("triage_method" in r for r in rows):
        methods = Counter(r["triage_method"] for r in rows if "triage_method" in r)
        md += ["", f"triage used the vote for {methods['vote']} examples and the LLM for {methods['llm']}."]

    secs = [r["seconds"] for r in rows]
    md += ["", "## Latency per example (seconds)", ""]
    for k in ("retrieval", "llm_only", "llm_rag"):
        vals = sorted(s[k] for s in secs if k in s)
        if vals:
            md.append(f"- {k}: median {vals[len(vals) // 2]:.2f}, p90 {vals[int(len(vals) * 0.9)]:.2f}, max {vals[-1]:.2f}")

    calls = [c for r in rows for c in r.get("llm_calls", [])]
    if calls:
        md += ["", "## LLM calls by provider", "",
               "| provider | model | calls | after a failed provider | prompt tokens | completion tokens |",
               "|---|---|---|---|---|---|"]
        for prov, model in sorted({(c["provider"], c["model"]) for c in calls}, key=str):
            mine = [c for c in calls if (c["provider"], c["model"]) == (prov, model)]
            fell = sum(1 for c in mine if c.get("failed_before"))
            pt = sum(c.get("prompt_tokens") or 0 for c in mine)
            ct = sum(c.get("completion_tokens") or 0 for c in mine)
            md.append(f"| {prov} | {model} | {len(mine)} | {fell} | {pt} | {ct} |")

    text = "\n".join(md) + "\n"
    results_path.with_suffix(".md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_file", nargs="?")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--report", help="rebuild the report from a saved results .json")
    ap.add_argument("--resume", help="continue a crashed run from its eval/results/<run>.partial.jsonl")
    args = ap.parse_args()
    if args.report:
        print(report(Path(args.report)))
    elif args.eval_file:
        out = run(args.eval_file, args.limit, not args.no_llm, args.resume)
        print(report(out))
        print(f"saved: {out} (+ .md)")
    else:
        ap.print_help()
        sys.exit(1)
