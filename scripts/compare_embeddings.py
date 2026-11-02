"""
Usage: python scripts/compare_embeddings.py [fastembed-model] [n-test-lines]

Checks whether an in-process embedding model (fastembed, ONNX) can stand in
for Ollama's nomic-embed-text against the EXISTING Qdrant index, which was
built with Ollama. Two checks:

  1. Same strings, both runtimes: cosine similarity of the two vectors for
     every stored template (357 on the BGL index).
  2. Same retrieval: for test lines, is the top-1 and top-5 result in Qdrant
     the same when the query vector comes from fastembed instead of Ollama?

Use it against the existing index only if both are near-perfect. Otherwise
an in-process deployment needs its own index, re-embedded with that model.
Model files are cached in data/fastembed_cache (not in git).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

import config  # noqa: E402
from agents.llm import embed_text  # noqa: E402
from bgl.lines import split_line  # noqa: E402
from db.mongo_client import incidents_collection  # noqa: E402
from db.qdrant_client import get_client  # noqa: E402
from ingestion.parser import match_line  # noqa: E402

model_name = sys.argv[1] if len(sys.argv) > 1 else "nomic-ai/nomic-embed-text-v1.5"
n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


t0 = time.perf_counter()
fe = TextEmbedding(model_name=model_name, cache_dir=str(ROOT / "data" / "fastembed_cache"))
print(f"loaded {model_name} in {time.perf_counter() - t0:.1f}s")

# 1. same strings
templates = [d["template"] for d in incidents_collection().find({}, {"template": 1})]
t0 = time.perf_counter()
fe_vecs = [unit(v) for v in fe.embed(templates)]
fe_s = time.perf_counter() - t0
t0 = time.perf_counter()
ol_vecs = [unit(embed_text(t)) for t in templates]
ol_s = time.perf_counter() - t0
cos = np.array([float(a @ b) for a, b in zip(fe_vecs, ol_vecs)])
print(f"\n1. {len(templates)} templates, cosine(fastembed, ollama): mean {cos.mean():.4f}  min {cos.min():.4f}  "
      f"p5 {np.percentile(cos, 5):.4f}  share >= 0.99: {(cos >= 0.99).mean():.1%}")
print(f"   time: fastembed {fe_s:.1f}s, ollama {ol_s:.1f}s for {len(templates)} strings")

# 2. same retrieval against the existing index
lines = [json.loads(l) for l in open(ROOT / "eval" / "bgl_eval.jsonl", encoding="utf-8")][:n_test]
queries = [match_line(split_line(l["raw_log"])["content"])["template"] for l in lines]
q = get_client()
top1 = top5 = 0
for text, fv in zip(queries, fe.embed(queries)):
    a = [p.id for p in q.query_points(config.QDRANT_COLLECTION, query=list(fv), limit=5).points]
    b = [p.id for p in q.query_points(config.QDRANT_COLLECTION, query=embed_text(text), limit=5).points]
    top1 += a[:1] == b[:1]
    top5 += set(a) == set(b)
print(f"\n2. {len(queries)} test lines against collection '{config.QDRANT_COLLECTION}':")
print(f"   same top-1 incident: {top1}/{len(queries)}   same top-5 set: {top5}/{len(queries)}")
