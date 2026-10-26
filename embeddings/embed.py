"""
Usage: python -m embeddings.embed

Finds every incident in Mongo that doesn't have a vector in Qdrant yet,
embeds its template text with Gemini, and upserts it. Run this after
ingestion.ingest, and re-run it any time ingestion adds new templates --
it only processes what's missing, so it's safe to run repeatedly.

Breaks-and-fixes:
  - 429 / rate-limit errors -> the free tier caps requests per minute. The
    sleep below is a blunt fix; if you still hit limits, increase it.
  - Dimension mismatch on upsert -> see db/qdrant_client.py docstring.
  - Nothing gets embedded, script says "0 incidents to embed" -> you
    probably haven't run ingestion.ingest yet, or Qdrant already has
    points for every current incident (check the Qdrant Cloud dashboard
    for the collection's point count).
"""
import time

import config
from db.mongo_client import incidents_collection
from db.qdrant_client import ensure_collection
from agents.llm import embed_text
from qdrant_client.models import PointStruct


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


def run():
    client = ensure_collection()
    existing_ids = already_embedded_ids(client)

    # existing_ids comes back from Qdrant as ints; Mongo _id is the string
    # form of the same cluster_id, so compare as strings.
    existing_ids = {str(i) for i in existing_ids}
    to_embed = list(incidents_collection().find({"_id": {"$nin": list(existing_ids)}}))
    print(f"{len(to_embed)} incidents to embed")

    for doc in to_embed:
        vector = embed_text(doc["template"])
        # Qdrant point IDs must be an unsigned int or a UUID string -- an
        # arbitrary string like our Mongo _id ("3") is NOT accepted by the
        # server even though the client won't complain until it's too
        # late. Drain3's cluster_id is already an int, so we use that
        # directly as the Qdrant id and keep the string form only in
        # Mongo. graph.py converts back with str(point.id) when it needs
        # to look an incident up in Mongo after a Qdrant search hit.
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=[PointStruct(
                id=int(doc["_id"]),
                vector=vector,
                payload={"template": doc["template"], "example_raw": doc.get("example_raw", "")},
            )],
        )
        print(f"  embedded {doc['_id']}: {doc['template'][:70]}")
        time.sleep(0.5)  # stay comfortably under free-tier rate limits


if __name__ == "__main__":
    run()
