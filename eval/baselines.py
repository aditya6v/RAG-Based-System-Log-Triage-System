"""
Usage: python -m eval.baselines <index.log> <eval.jsonl> <results.json>

Text-classifier baselines trained on exactly the lines that went into the
retrieval index, then added to an existing results file so they appear in the
same report as the retrieval and LLM classifiers. No database or LLM needed.

  majority  -- always the most common category in the index lines
  tfidf_lr  -- TF-IDF on "component level message" (word 1-2 grams plus
               character 3-5 grams, so masked numbers and odd tokens still
               count) and logistic regression with balanced class weights
               [scikit-learn, ref 5]

The classifiers see the same text as Drain3 (bgl/lines.py), never the label.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from bgl.categories import category_for_code
from bgl.lines import has_message, split_line


def load_index(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            sp = split_line(line.rstrip("\n"))
            if sp["label"] and has_message(sp["content"]):
                texts.append(sp["content"])
                labels.append(category_for_code(sp["label"]))
    return texts, labels


def train(texts: list[str], labels: list[str]):
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    model = make_pipeline(features, LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(texts, labels)
    return model


def main(index_path: str, eval_path: str, results_path: str) -> None:
    texts, labels = load_index(index_path)
    majority = Counter(labels).most_common(1)[0][0]
    model = train(texts, labels)
    examples = {}
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples[ex["line_no"]] = split_line(ex["raw_log"])["content"]

    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = data["rows"]
    preds = model.predict([examples[r["line_no"]] for r in rows])
    for r, p in zip(rows, preds):
        r["pred"]["majority"] = majority
        r["pred"]["tfidf_lr"] = str(p)
    data["baselines"] = {"index_file": index_path, "index_lines": len(texts), "majority_class": majority}
    Path(results_path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    acc = sum(r["pred"]["tfidf_lr"] == r["true_label"] for r in rows) / len(rows)
    print(f"trained on {len(texts)} index lines; majority class '{majority}'; tfidf_lr accuracy {acc:.3f} on {len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
