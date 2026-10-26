"""
Usage: python -m ingestion.ingest sample_data/bgl_index.log

Reads a BGL log file line by line, templates each line with Drain3, and
writes two things to Mongo:
  - one `events` doc per raw line (the full history, for audit/debugging)
  - one `incidents` doc per *unique* cluster_id, upserted so re-running
    this script on overlapping data doesn't create duplicates -- it just
    bumps occurrence_count and updates last_seen.

The BGL label token at the start of each line ('-' or an alert code) is
split off before Drain3 sees the line (see bgl/lines.py). It is stored on
the event, and each incident keeps `label_counts` -- how many of its lines
fell in each category. The majority of those is the incident's label, which
the nearest-neighbour vote in agents/classify.py uses.

To start over from scratch: delete drain3_state.bin, point MONGO_DB_NAME and
QDRANT_COLLECTION at empty ones (or drop them), then ingest and embed again.
"""
import sys
from datetime import datetime, timezone

from pymongo import UpdateOne

from bgl.categories import category_for_code
from bgl.lines import has_message, split_line
from db.mongo_client import events_collection, incidents_collection
from ingestion.parser import parse_line

BATCH_SIZE = 1000


def ingest_file(path: str) -> None:
    events = events_collection()
    incidents = incidents_collection()
    count = skipped = 0
    event_batch = []
    incident_ops = []

    def flush():
        if event_batch:
            events.insert_many(event_batch)
            event_batch.clear()
        if incident_ops:
            incidents.bulk_write(incident_ops, ordered=True)
            incident_ops.clear()

    # Explicit utf-8: Windows defaults to cp1252, which crashes on real log files.
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            sp = split_line(raw_line.rstrip("\n"))
            if not sp["text"] or not has_message(sp["content"]):
                skipped += 1
                continue
            category = category_for_code(sp["label"]) if sp["label"] else None

            parsed = parse_line(sp["content"])
            now = datetime.now(timezone.utc)

            event_batch.append({
                "raw": sp["text"],
                "content": sp["content"],
                "code": sp["label"],
                "category": category,
                "cluster_id": parsed["cluster_id"],
                "ingested_at": now,
            })
            update = {
                "$set": {"template": parsed["template"], "last_seen": now},
                "$setOnInsert": {"first_seen": now, "example_raw": sp["text"]},
                "$inc": {"occurrence_count": 1},
            }
            if category:
                update["$inc"][f"label_counts.{category}"] = 1
            incident_ops.append(UpdateOne({"_id": parsed["cluster_id"]}, update, upsert=True))
            count += 1

            if len(event_batch) >= BATCH_SIZE:
                flush()

    flush()  # final partial batch

    print(f"Ingested {count} log lines from {path} (skipped {skipped} empty/no-message lines)")
    print(f"Distinct incident templates so far: {incidents.count_documents({})}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m ingestion.ingest <log file>")
        sys.exit(1)
    ingest_file(sys.argv[1])
