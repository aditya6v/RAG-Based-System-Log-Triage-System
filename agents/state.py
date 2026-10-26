"""
The one object that flows through the whole graph. Every node reads from
this and returns a dict of only the fields it's changing -- LangGraph
merges that dict into the running state, it does not replace it.

`notes` is an audit trail, not application logic: every node appends a
short string to it. It's the first thing to print when something
downstream looks wrong, since it tells you how far the state got before
breaking. `timings_ms` records each node's wall time.
"""
from typing import TypedDict


class IncidentState(TypedDict, total=False):
    raw_log: str            # input: the log line to diagnose
    text: str               # retrieval: raw_log without the BGL label token
    content: str            # retrieval: "component level message" (what Drain3 sees)
    cluster_id: str | None  # retrieval: matched Drain3 cluster, None if no exact match
    template: str           # retrieval: the cluster's template (or the content)
    retrieved: list[dict]   # retrieval: similar past incidents with score and label
    classification: str     # triage: category id from bgl/categories.json
    triage_method: str      # triage: "vote" or "llm"
    vote: dict              # triage: nearest-neighbour vote details
    severity: str           # triage: none/low/medium/high, from categories.json
    route: str              # "full" or "no_action"
    evidence: dict          # root_cause: {"primary": hit|None, "supporting": [hits]}
    root_cause: str         # root_cause (or no_action)
    remediation: str        # remediation (or no_action)
    notes: list[str]        # audit trail, appended to by every node
    timings_ms: dict        # per-node wall time
