"""
Usage: python -m embeddings.embed [--rebuild]

Finds every incident in Mongo that doesn't have a vector in Qdrant yet,
embeds its template text, and upserts it in batches. Run this after
ingestion.ingest, and re-run it any time ingestion adds new templates --
it only processes what's missing, so it's safe to run repeatedly.

--rebuild re-embeds everything. Use it after changing the embedding model
(vectors from different models don't share a space, scores collapse) or
after re-ingesting, since Drain3 may have widened templates that were
already embedded.

Breaks-and-fixes:
  - 429 / rate-limit errors (Gemini only) -> the free tier caps requests per
    minute. The sleep below is a blunt fix; if you still hit limits, increase it.
  - Dimension mismatch on upsert -> see db/qdrant_client.py docstring.
  - Nothing gets embedded, script says "0 incidents to embed" -> you
    probably haven't run ingestion.ingest yet, or Qdrant already has
    points for every current incident (check the Qdrant Cloud dashboard
    for the collection's point count).
"""
import sys
import time

import config
from agents.llm import embed_text
from bgl.categories import majority
from db.mongo_client import incidents_collection
from db.qdrant_client import ensure_collection
from qdrant_client.models import PointStruct

BATCH_SIZE = 64


def already_embedded_ids(client) -> set[str]:
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_payload=False,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        ids.update(p.id for p in points)
        if offset is None:
            break
    return ids


def run(rebuild: bool = False):
    client = ensure_collection()
    # existing_ids comes back from Qdrant as ints; Mongo _id is the string
    # form of the same cluster_id, so compare as strings.
    existing_ids = set() if rebuild else {str(i) for i in already_embedded_ids(client)}
    to_embed = list(incidents_collection().find({"_id": {"$nin": list(existing_ids)}}))
    print(f"{len(to_embed)} incidents to embed")

    batch = []
    for i, doc in enumerate(to_embed, 1):
        # Qdrant point IDs must be an unsigned int or a UUID string -- an
        # arbitrary string like our Mongo _id ("3") is NOT accepted by the
        # server. Drain3's cluster_id is already an int, so we use that
        # directly as the Qdrant id and keep the string form only in Mongo.
        batch.append(PointStruct(
            id=int(doc["_id"]),
            vector=embed_text(doc["template"]),
            payload={
                "template": doc["template"],
                "example_raw": doc.get("example_raw", ""),
                "label": majority(doc.get("label_counts")),
            },
        ))
        if config.LLM_PROVIDER == "gemini":
            time.sleep(0.5)  # stay under Gemini free-tier rate limits
        if len(batch) >= BATCH_SIZE or i == len(to_embed):
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=batch)
            print(f"  embedded {i}/{len(to_embed)}")
            batch = []


if __name__ == "__main__":
    run(rebuild="--rebuild" in sys.argv)
