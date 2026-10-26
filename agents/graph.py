"""
The actual multi-agent pipeline, deliberately linear for the midterm (no
conditional routing yet -- see the architecture discussion: that's a
phase-2 addition, not something the demo needs). Two combined nodes
standing in for the full 4-agent target:

    START -> triage_and_retrieve -> diagnose_and_remediate -> END

Run this file directly (`python -m agents.graph`) to sanity-check the
whole pipeline against one hardcoded log line without needing the Flask
app running -- this is the fastest debug loop while you're iterating on
prompts, since Flask adds nothing useful while you're still fixing agent
logic.

Breaks-and-fixes:
  - "KeyError" on a state field -> a node returned a dict missing a key
    some later node expects. Print state["notes"] first -- it tells you
    exactly which nodes ran before the failure.
  - Gemini response isn't valid JSON when you asked for structured output
    -> see _parse_json_response below; never let one bad response crash
    the request, especially live during a demo.
  - Retrieval returns nothing / irrelevant incidents -> either Qdrant is
    empty (did you run embeddings/embed.py?) or the live log's Drain3
    template diverged from the stored ones because drain3_state.bin is
    stale/missing (see ingestion/parser.py docstring).
"""
import json

from agents.state import IncidentState
from agents.llm import chat_text, embed_text
from ingestion.parser import parse_line
from db.mongo_client import incidents_collection
from db.qdrant_client import get_client
import config

from langgraph.graph import StateGraph, START, END


def _parse_json_response(text: str) -> dict:
    """LLMs asked for JSON sometimes wrap it in ```json fences or add a
    sentence before/after. Strip the common cases; if it still won't
    parse, fall back to treating the raw text as a single field rather
    than raising -- a malformed-but-present answer beats a crashed
    request during a live demo."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"_unparsed": text}


def triage_and_retrieve(state: IncidentState) -> dict:
    parsed = parse_line(state["raw_log"])
    query_vector = embed_text(parsed["template"])

    hits = get_client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=3,
        with_payload=True,
    ).points

    retrieved = []
    for hit in hits:
        mongo_doc = incidents_collection().find_one({"_id": str(hit.id)})
        retrieved.append({
            "score": round(hit.score, 3),
            "template": hit.payload.get("template", ""),
            "example_raw": hit.payload.get("example_raw", ""),
            "occurrence_count": mongo_doc.get("occurrence_count") if mongo_doc else None,
        })

    classify_prompt = f"""You are a log triage assistant. Classify this log line as EXACTLY ONE of the following categories (copy it exactly, verbatim):

- network allocation failure
- network binding failure
- disk space cleanup failure
- normal operation

Log line: {state["raw_log"]}
Extracted template: {parsed["template"]}

Respond with ONLY the category text from the list above, nothing else -- no punctuation, no explanation."""
    classification = chat_text(classify_prompt).strip()

    return {
        "cluster_id": parsed["cluster_id"],
        "template": parsed["template"],
        "retrieved": retrieved,
        "classification": classification,
        "notes": state.get("notes", []) + [
            f"triage_and_retrieve: classified as '{classification}', found {len(retrieved)} similar past incidents"
        ],
    }

def diagnose_and_remediate(state: IncidentState) -> dict:
    if "normal" in state.get("classification", "").lower():
        return {
            "root_cause": "No root cause -- classified as normal operation.",
            "remediation": "No action needed.",
            "notes": state.get("notes", []) + ["diagnose_and_remediate: skipped, classified as normal operation"],
        }

    evidence = ...  # existing code continues unchanged

def diagnose_and_remediate(state: IncidentState) -> dict:
    evidence = "\n".join(
        f"- (similarity {r['score']}, seen {r['occurrence_count']}x before) {r['template']}"
        for r in state.get("retrieved", [])
    ) or "No similar past incidents were found in the retrieval step."

    prompt = f"""You are a root-cause analysis assistant for infrastructure
incidents. Ground your answer in the retrieved past incidents below --
do not invent causes that aren't supported by either the current log or
the retrieved evidence.

Current incident:
  log line: {state["raw_log"]}
  classification: {state.get("classification")}

Similar past incidents (retrieved via semantic search):
{evidence}

Respond with ONLY a JSON object, no markdown fences, no extra text, with
exactly these two keys:
{{"root_cause": "<1-3 sentence explanation>", "remediation": "<1-3 sentence suggested fix>"}}"""

    raw_response = chat_text(prompt)
    parsed = _parse_json_response(raw_response)

    root_cause = parsed.get("root_cause", parsed.get("_unparsed", raw_response))
    remediation = parsed.get("remediation", "")

    return {
        "root_cause": root_cause,
        "remediation": remediation,
        "notes": state.get("notes", []) + ["diagnose_and_remediate: produced root cause + remediation"],
    }


def build_graph():
    g = StateGraph(IncidentState)
    g.add_node("triage_and_retrieve", triage_and_retrieve)
    g.add_node("diagnose_and_remediate", diagnose_and_remediate)
    g.add_edge(START, "triage_and_retrieve")
    g.add_edge("triage_and_retrieve", "diagnose_and_remediate")
    g.add_edge("diagnose_and_remediate", END)
    return g.compile()


if __name__ == "__main__":
    config.require_all()
    app = build_graph()
    result = app.invoke({
        "raw_log": "nova-compute.log.1 2017-05-16 03:12:09.112 2931 ERROR nova.compute.manager [instance: aa21] Failed to allocate network for instance, no fixed IP addresses available.",
        "notes": [],
    })
    print(json.dumps(result, indent=2, default=str))
