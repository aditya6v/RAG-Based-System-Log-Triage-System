"""
Usage: python -m eval.check_report eval/results/<run>.json

Recomputes the headline numbers quoted in README.md and docs/EVALUATION.md
from saved predictions, without calling any model or database, and exits 1 if
any of them differs. Run in CI: it catches a change to the triage rule or the
scoring code that would make the published numbers stale.
"""
import json
import sys
from pathlib import Path

from eval.evaluate import _binary, _with_triage

# (classifier, subset, metric) -> value quoted in README.md, rounded to 3 places
EXPECTED = {
    ("knn_vote", "all", "accuracy"): 0.981,
    ("knn_vote", "unseen", "accuracy"): 0.921,
    ("llm_only", "all", "accuracy"): 0.425,
    ("llm_rag", "all", "accuracy"): 0.639,
    ("vote_llm_fallback", "all", "accuracy"): 0.982,
    ("vote_llm_fallback", "unseen", "accuracy"): 0.927,
    ("vote_llm_fallback", "all", "binary_accuracy"): 0.997,
    ("vote_llm_fallback", "hard_negative", "called_normal"): 0.981,
    ("triage", "all", "answered_precision"): 0.993,
    ("triage", "all", "coverage"): 0.976,
    ("level_rule", "all", "binary_accuracy"): 0.876,
    ("knn_vote", "hard_negative", "called_normal"): 1.000,
    ("level_rule", "hard_negative", "called_normal"): 0.287,
}


def compute(rows: list[dict], clf: str, subset: str, metric: str) -> float:
    if subset == "all":
        rs = rows
    elif subset == "hard_negative":
        rs = [r for r in rows if r["hard_negative"]]
    else:
        rs = [r for r in rows if r["subset"] == subset]
    rs = [r for r in rs if clf in r["pred"]]

    def pred(r):
        p = r["pred"][clf]
        return p if clf == "level_rule" else _binary(p) if metric != "accuracy" else p

    if metric == "accuracy":
        return sum(r["pred"][clf] == r["true_label"] for r in rs) / len(rs)
    if metric == "binary_accuracy":
        return sum(pred(r) == _binary(r["true_label"]) for r in rs) / len(rs)
    if metric == "called_normal":
        return sum(pred(r) == "normal" for r in rs) / len(rs)
    if metric in ("answered_precision", "coverage"):
        answered = [r for r in rs if r["pred"][clf] != "unknown"]
        if metric == "coverage":
            return len(answered) / len(rs)
        return sum(r["pred"][clf] == r["true_label"] for r in answered) / len(answered)
    raise ValueError(metric)


def main(path: str) -> int:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    _with_triage(rows)
    bad = 0
    for (clf, subset, metric), expected in EXPECTED.items():
        got = round(compute(rows, clf, subset, metric), 3)
        ok = abs(got - expected) < 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {clf:18} {subset:13} {metric:18} expected {expected:.3f} got {got:.3f}")
    print(f"\n{len(EXPECTED) - bad}/{len(EXPECTED)} headline numbers match")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m eval.check_report <results.json>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
