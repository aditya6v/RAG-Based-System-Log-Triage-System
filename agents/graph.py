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
from agents.classify import triage
from agents.llm import chat_text
from agents.retrieval import retrieve
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
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last try: the outermost {...}, for "Here is the JSON: {...}" replies.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        try:
            parsed = json.loads(cleaned[start:end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            parsed = None
    return parsed if isinstance(parsed, dict) else {"_unparsed": text}


def triage_and_retrieve(state: IncidentState) -> dict:
    r = retrieve(state["raw_log"])
    t = triage(r["text"], r["retrieved"])

    template_note = (
        f"matched template cluster {r['cluster_id']}" if r["cluster_id"] is not None
        else "no matching template from ingestion, searched with the raw line"
    )
    return {
        "cluster_id": r["cluster_id"],
        "template": r["template"],
        "retrieved": r["retrieved"],
        "classification": t["classification"],
        "notes": state.get("notes", []) + [
            f"triage_and_retrieve: {template_note}; classified as '{t['classification']}' "
            f"by {t['method']}, found {len(r['retrieved'])} similar past incidents"
        ],
    }


def diagnose_and_remediate(state: IncidentState) -> dict:
    if "normal" in state.get("classification", "").lower():
        return {
            "root_cause": "No root cause -- classified as normal operation.",
            "remediation": "No action needed.",
            "notes": state.get("notes", []) + ["diagnose_and_remediate: skipped, classified as normal operation"],
        }

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
        "raw_log": "1125552602 2005.08.31 R23-M1-N8-I:J18-U01 2005-08-31-22.30.02.453491 R23-M1-N8-I:J18-U01 RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1",
        "notes": [],
    })
    print(json.dumps(result, indent=2, default=str))
