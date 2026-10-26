"""
Retrieval: raw log line in, the most similar past incidents out.

    raw line -> strip BGL label -> Drain3 match (read-only) -> embed template
             -> Qdrant top-k -> one Mongo lookup for counts and labels

Shared by the graph's retrieval agent and eval/evaluate.py, so the eval
measures exactly what the app does.
"""
from agents.llm import embed_text
from bgl.categories import majority
from bgl.lines import split_line
from db.mongo_client import incidents_collection
from db.qdrant_client import get_client
from ingestion.parser import match_line
import config

TOP_K = 5


def retrieve(raw_log: str, limit: int = TOP_K) -> dict:
    sp = split_line(raw_log)
    parsed = match_line(sp["content"])

    hits = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_text(parsed["template"]),
        limit=limit,
        with_payload=True,
    ).points

    ids = [str(h.id) for h in hits]
    docs = {d["_id"]: d for d in incidents_collection().find({"_id": {"$in": ids}})}
    retrieved = []
    for hit in hits:
        doc = docs.get(str(hit.id), {})
        retrieved.append({
            "cluster_id": str(hit.id),
            "score": round(hit.score, 3),
            "template": hit.payload.get("template", ""),
            "example_raw": hit.payload.get("example_raw", ""),
            "occurrence_count": doc.get("occurrence_count"),
            "label": majority(doc.get("label_counts")) or hit.payload.get("label"),
        })

    return {
        "text": sp["text"],
        "content": sp["content"],
        "cluster_id": parsed["cluster_id"],
        "template": parsed["template"],
        "retrieved": retrieved,
    }
