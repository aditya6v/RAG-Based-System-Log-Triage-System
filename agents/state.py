"""
The one object that flows through the whole graph. Every node reads from
this and returns a dict of only the fields it's changing -- LangGraph
merges that dict into the running state, it does not replace it.

`notes` is an audit trail, not application logic: every node appends a
short string to it. This is what you show on a slide/demo to prove the
pipeline actually did multi-step reasoning rather than one opaque LLM
call -- and it's the first thing to print when something downstream looks
wrong, since it tells you how far the state got before breaking.
"""
from typing import TypedDict


class IncidentState(TypedDict, total=False):
    raw_log: str          # input: the log line to diagnose
    cluster_id: str        # filled by triage_and_retrieve
    template: str           # filled by triage_and_retrieve
    retrieved: list[dict]    # filled by triage_and_retrieve: similar past incidents
    classification: str       # filled by triage_and_retrieve
    root_cause: str            # filled by diagnose_and_remediate
    remediation: str            # filled by diagnose_and_remediate
    notes: list[str]              # audit trail, appended to by every node
