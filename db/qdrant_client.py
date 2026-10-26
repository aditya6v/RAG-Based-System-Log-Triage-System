"""
Thin wrapper around qdrant-client. One collection, "incidents" by default,
storing one vector per unique template (same unit as the Mongo `incidents`
collection -- the Qdrant point ID and the Mongo _id for a given incident
are the same string so you can hop from a search hit straight to the full
Mongo doc).

Breaks-and-fixes:
  - "Vector dimension error" on upsert -> config.EMBEDDING_DIM doesn't match
    what your embedding model actually returns. Check this first if you
    ever change GEMINI_EMBED_MODEL.
  - Collection already exists with a different config -> ensure_collection
    is idempotent for *creation* but won't migrate an existing mismatched
    collection. Easiest fix during development: delete the collection from
    the Qdrant Cloud dashboard and re-run embeddings/embed.py.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return _client


def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return client
