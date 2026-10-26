"""
Thin wrapper around pymongo. Two collections, both created lazily (Mongo
creates a collection on first insert, no migration step needed):

  events    -- every raw log line we've ever ingested, one doc each.
  incidents -- one doc per *unique template* (i.e. per Drain3 cluster).
               This is the thing that gets embedded and retrieved, not raw
               events -- you want to match on "kind of failure", not on
               one specific instance ID.

If connections hang instead of erroring: check (1) your current IP is
allow-listed in Atlas -> Network Access, (2) the password in MONGO_URI
doesn't contain characters that need URL-encoding (@, :, / in a password
will break the URI silently).
"""
from pymongo import MongoClient
from pymongo.database import Database

import config

_client: MongoClient | None = None


def get_db() -> Database:
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=8000)
        _client.admin.command("ping")  # fails fast with a clear error if unreachable
    return _client[config.MONGO_DB_NAME]


def events_collection():
    return get_db()["events"]


def incidents_collection():
    return get_db()["incidents"]
