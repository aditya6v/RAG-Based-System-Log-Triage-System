"""
Usage: python -m eval.evaluate path/to/labeled.json

Expects a JSON file that's a list of {"raw_log": "...", "true_label": "..."}
objects. Runs the triage classification step (not the full pipeline --
root cause/remediation text can't be scored against a fixed label the same
way) on each, and reports precision/recall/F1 using scikit-learn.

This is deliberately a loose string-match comparison between the model's
classification and your true_label, which means wording matters (the model
saying "network allocation failure" won't match a true_label of "network
error" even though a human would call that correct). For a real number
worth putting on a slide, either normalize both sides to a small fixed set
of category names, or have the LLM pick from an explicit list of allowed
categories instead of free-text classification -- a 15-minute change in
graph.py's classify_prompt if you decide it's worth it before the demo.

Breaks-and-fixes:
  - sklearn warns about labels with no predicted samples -> normal if your
    label set is small and the model never predicted one of them; doesn't
    mean the script is broken.
  - This is slow for more than a few dozen examples -> each one is a real
    Gemini call plus a Qdrant search. Fine for a 30-50 example eval set,
    not meant for the full labeled dataset under time pressure.
"""
import json
import sys

from sklearn.metrics import classification_report

from agents.graph import triage_and_retrieve
import config


def run(path: str):
    config.require_all()
    with open(path) as f:
        examples = json.load(f)

    y_true, y_pred = [], []
    for ex in examples:
        state = triage_and_retrieve({"raw_log": ex["raw_log"], "notes": []})
        y_true.append(ex["true_label"].strip().lower())
        y_pred.append(state["classification"].strip().lower().rstrip("."))
        print(f"  true={ex['true_label']!r:35} pred={state['classification']!r}")

    print()
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m eval.evaluate <path-to-labeled.json>")
        sys.exit(1)
    run(sys.argv[1])
