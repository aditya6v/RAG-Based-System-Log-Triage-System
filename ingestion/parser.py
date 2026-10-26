"""
Drain3 wrapper: turns a raw log line into a (cluster_id, template) pair.

How to read a Drain3 result: `add_log_message` returns a dict; the two
fields we care about are `cluster_id` (an int identifying which template
this line matched, stable across calls) and `template_mined` (the line
with variable parts replaced by <*>). Two raw lines that differ only in,
say, an instance ID or a timestamp should produce the same cluster_id and
template -- that's the whole point of templating.

Breaks-and-fixes:
  - Every line gets its own cluster_id (nothing is collapsing) -> sim_th in
    drain3.ini is too high, or your logs have more incidental variation
    (extra whitespace, inconsistent field order) than Drain3's default
    tokenizer handles well. Try lowering sim_th first.
  - Unrelated errors sharing a cluster_id -> sim_th is too low.
  - A cluster_id that meant one thing during ingestion means something
    else (or nothing) when the live Flask app diagnoses a new log line --
    this WILL happen if ingestion and the app run as separate processes
    without shared state, because Drain3's learned clusters live in
    memory by default. We avoid this by persisting state to
    drain3_state.bin (below): both ingestion.ingest and the running app
    load the same learned clusters on startup. Run ingestion at least
    once before starting app.py, so the file exists with real clusters in
    it, not an empty state.
"""
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

_config = TemplateMinerConfig()
_config.load("drain3.ini")
_persistence = FilePersistence("drain3_state.bin")
_miner = TemplateMiner(persistence_handler=_persistence, config=_config)


def parse_line(raw_line: str) -> dict:
    result = _miner.add_log_message(raw_line)
    return {
        "raw": raw_line,
        "cluster_id": str(result["cluster_id"]),
        "template": result["template_mined"],
    }
