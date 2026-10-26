"""
Usage: python -m ingestion.ingest sample_data/sample_logs.txt

Reads a raw log file line by line, templates each line with Drain3, and
writes two things to Mongo:
  - one `events` doc per raw line (the full history, for audit/debugging)
  - one `incidents` doc per *unique* cluster_id, upserted so re-running
    this script on overlapping data doesn't create duplicates -- it just
    bumps occurrence_count and updates last_seen.

This is intentionally the only ingestion path for both "historical
incidents to learn from" and "new incident to diagnose" -- there is no
separate code path for live traffic yet. When you wire the real `/diagnose`
endpoint to new incoming logs later, it should call `parse_line` the same
way this script does, so templates stay consistent between what's stored
and what's queried.
"""
from pymongo import UpdateOne

import sys
from datetime import datetime, timezone

from ingestion.parser import parse_line
from db.mongo_client import events_collection, incidents_collection


BATCH_SIZE = 1000


def ingest_file(path: str) -> None:
    events = events_collection()
    incidents = incidents_collection()
    count = 0
    event_batch = []
    incident_ops = []

    def flush():
        if event_batch:
            events.insert_many(event_batch)
            event_batch.clear()
        if incident_ops:
            incidents.bulk_write(incident_ops)
            incident_ops.clear()

    # Explicit utf-8: Windows defaults to cp1252, which crashes on real log files.
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip() or raw_line.lstrip().startswith(";"):
                continue  # skip blank lines and comment lines like the ones in sample_logs.txt

            parsed = parse_line(raw_line)
            now = datetime.now(timezone.utc)

            event_batch.append({
                "raw": parsed["raw"],
                "cluster_id": parsed["cluster_id"],
                "ingested_at": now,
            })
            incident_ops.append(UpdateOne(
                {"_id": parsed["cluster_id"]},
                {
                    "$set": {"template": parsed["template"], "last_seen": now},
                    "$setOnInsert": {"first_seen": now, "example_raw": parsed["raw"]},
                    "$inc": {"occurrence_count": 1},
                },
                upsert=True,
            ))
            count += 1

            if len(event_batch) >= BATCH_SIZE:
                flush()

    flush()  # final partial batch

    print(f"Ingested {count} log lines from {path}")
    print(f"Distinct incident templates so far: {incidents.count_documents({})}")