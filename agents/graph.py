"""
The multi-agent pipeline: four agents and one deterministic router.

    START -> retrieval -> triage --(router)--> root_cause -> remediation -> END
                                       \\
                                        +--> no_action -> END

  retrieval    Drain3 template match + Qdrant top-5 + Mongo counts/labels
  triage       category + severity. Nearest-neighbour vote when confident,
               otherwise the LLM (see agents/classify.py)
  router       plain function, no LLM: severity from bgl/categories.json.
               'none'/'low' -> no_action, anything else -> root_cause
  root_cause   LLM, grounded: the top hit is primary evidence, other hits
               above MIN_EVIDENCE_SCORE are weak support, the rest are dropped
  remediation  LLM, given the root cause and the category definition

Retrieval runs before triage (the plan had it the other way round) because
triage votes with the retrieved incidents' labels.

Run this file directly (`python -m agents.graph`) to push one log line
through the whole pipeline without the API in the way.

Breaks-and-fixes:
  - "KeyError" on a state field -> a node returned a dict missing a key
    some later node expects. Print state["notes"] first -- it tells you
    exactly which nodes ran before the failure.
  - LLM response isn't valid JSON -> see _parse_json_response below; never
    let one bad response crash the request, especially live during a demo.
  - Retrieval returns nothing / irrelevant incidents -> either Qdrant is
    empty (did you run embeddings/embed.py?) or drain3_state.bin doesn't
    match what was embedded (re-run ingestion, then embed --rebuild).
"""
import json
import time

from agents.classify import triage
from agents.llm import chat_text
from agents.retrieval import retrieve
from agents.state import IncidentState
from bgl import categories
import config

from langgraph.graph import StateGraph, START, END

# Hits below this similarity are not shown to the root-cause agent at all.
# Calibrated for nomic-embed-text on the 357 stored BGL incidents (not the
# eval set): among each incident's top-5 neighbours, 0.85 keeps 40% of
# same-label pairs and 25% of different-label pairs; see docs/EVALUATION.md.
# Scores from a different embedding model are not comparable; recalibrate.
MIN_EVIDENCE_SCORE = 0.85

NO_ACTION_SEVERITIES = {"none", "low"}

# A stricter prompt ("only name files, paths, hosts... that appear in the log
# or the evidence") was tried and measured on 42 diagnoses: sentences naming an
# invented specific went from 5.4% to 9.6%, so it was reverted. See
# docs/EVALUATION.md (groundedness) and eval/rubric/.


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


def _timed(name, fn):
    """Wrap a node so it records its own wall time in state["timings_ms"]."""
    def node(state: IncidentState) -> dict:
        t0 = time.perf_counter()
        out = fn(state)
        ms = round((time.perf_counter() - t0) * 1000)
        out["timings_ms"] = {**state.get("timings_ms", {}), name: ms}
        return out
    node.__name__ = name
    return node


def _note(state, text):
    return state.get("notes", []) + [text]


# --- agents -------------------------------------------------------------------

def retrieval_agent(state: IncidentState) -> dict:
    r = retrieve(state["raw_log"])
    how = (f"matched template cluster {r['cluster_id']}" if r["cluster_id"] is not None
           else "no exact template match, searched with the raw line")
    return {
        "text": r["text"],
        "content": r["content"],
        "cluster_id": r["cluster_id"],
        "template": r["template"],
        "retrieved": r["retrieved"],
        "notes": _note(state, f"retrieval: {how}; {len(r['retrieved'])} similar incidents, "
                              f"top similarity {r['retrieved'][0]['score'] if r['retrieved'] else 'n/a'}"),
    }


def triage_agent(state: IncidentState) -> dict:
    t = triage(state["text"], state.get("retrieved", []))
    # 'unknown' isn't in categories.json, so it gets severity 'medium' and a
    # full diagnosis: an unrecognised line is never silently dropped.
    sev = categories.severity(t["classification"])
    how = {"vote": "nearest-neighbour vote", "llm": "LLM",
           "abstain": f"abstained (LLM suggests '{t['suggestion']}', unverified)"}[t["method"]]
    return {
        "classification": t["classification"],
        "triage_method": t["method"],
        "vote": t["vote"],
        "confidence": t["confidence"],
        "suggestion": t["suggestion"],
        "severity": sev,
        "notes": _note(state, f"triage: '{t['classification']}' (severity {sev}) by {how}, "
                              f"confidence {t['confidence']}"),
    }


def _category_line(state: IncidentState, cat: dict) -> str:
    if state.get("classification") == "unknown":
        hint = f" A language model suggests '{state['suggestion']}', unverified." if state.get("suggestion") else ""
        return ("unknown: the past incidents did not match closely enough to name a category."
                f"{hint} Say that the category is uncertain.")
    return f"{cat.get('name', state.get('classification'))} -- {cat.get('definition', '')}"


def route_after_triage(state: IncidentState) -> str:
    """Deterministic: severity comes from categories.json, not from the LLM."""
    return "no_action" if state.get("severity") in NO_ACTION_SEVERITIES else "root_cause"


def no_action(state: IncidentState) -> dict:
    return {
        "route": "no_action",
        "evidence": {"primary": None, "supporting": []},
        "root_cause": f"No root cause needed -- classified as {state.get('classification')}.",
        "remediation": "No action needed.",
        "notes": _note(state, "router: severity "
                              f"{state.get('severity')} -> skipped root cause and remediation"),
    }


def select_evidence(retrieved: list[dict], classification: str | None = None) -> dict:
    """Primary: the best hit, if it clears MIN_EVIDENCE_SCORE. Supporting:
    other hits that clear it AND carry the same label as the triage result.
    Similarity alone is a weak filter here -- among the stored incidents' own
    top-5 neighbours, 25% of different-label pairs still score >= 0.85 -- so
    a lookalike from another category would otherwise leak into the
    explanation (the midterm's "blamed disk space" problem)."""
    kept = [r for r in retrieved if r["score"] >= MIN_EVIDENCE_SCORE]
    if not kept:
        return {"primary": None, "supporting": []}
    supporting = [r for r in kept[1:] if classification is None or r.get("label") == classification]
    return {"primary": kept[0], "supporting": supporting}


def root_cause_agent(state: IncidentState) -> dict:
    ev = select_evidence(state.get("retrieved", []), state.get("classification"))
    cat = categories.get(state.get("classification", "")) or {}

    def describe(r):
        return (f"(similarity {r['score']}, seen {r['occurrence_count']}x, "
                f"labelled {r.get('label')}) {r['template']}")

    if ev["primary"]:
        evidence = f"PRIMARY evidence (closest match, rely on this most):\n- {describe(ev['primary'])}"
        if ev["supporting"]:
            evidence += ("\nWEAK supporting evidence (less similar; use only if consistent "
                         "with the primary):\n" + "\n".join(f"- {describe(r)}" for r in ev["supporting"]))
    else:
        evidence = ("No sufficiently similar past incident was found. Reason only from the "
                    "log line itself and say that the evidence is limited.")

    prompt = f"""You are a root-cause analysis assistant for the BlueGene/L supercomputer.

Current incident:
  log line: {state.get("text", state["raw_log"])}
  category: {_category_line(state, cat)}

{evidence}

Explain the most likely root cause of THIS log line. Do not mention failures
that appear only in weak evidence and are unrelated to this line's message.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"root_cause": "<1-3 sentence explanation>"}}"""

    parsed = _parse_json_response(chat_text(prompt))
    root_cause = parsed.get("root_cause") or parsed.get("_unparsed", "")
    return {
        "route": "full",
        "evidence": ev,
        "root_cause": root_cause,
        "notes": _note(state, f"root_cause: used {1 if ev['primary'] else 0} primary + "
                              f"{len(ev['supporting'])} weak incidents "
                              f"(dropped {len(state.get('retrieved', [])) - (1 if ev['primary'] else 0) - len(ev['supporting'])} "
                              f"below {MIN_EVIDENCE_SCORE})"),
    }


def remediation_agent(state: IncidentState) -> dict:
    cat = categories.get(state.get("classification", "")) or {}
    prompt = f"""You are an operations assistant for the BlueGene/L supercomputer.

Incident:
  log line: {state.get("text", state["raw_log"])}
  category: {cat.get("name", state.get("classification"))} (severity {state.get("severity")})
  root cause: {state.get("root_cause")}

Suggest a concrete remediation an operator could carry out: what to check
first and what to do. Stay consistent with the root cause above.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"remediation": "<1-3 sentence suggested fix>"}}"""

    parsed = _parse_json_response(chat_text(prompt))
    return {
        "remediation": parsed.get("remediation") or parsed.get("_unparsed", ""),
        "notes": _note(state, "remediation: produced a suggested fix"),
    }


def build_graph():
    g = StateGraph(IncidentState)
    g.add_node("retrieval", _timed("retrieval", retrieval_agent))
    g.add_node("triage", _timed("triage", triage_agent))
    g.add_node("no_action", _timed("no_action", no_action))
    g.add_node("root_cause", _timed("root_cause", root_cause_agent))
    g.add_node("remediation", _timed("remediation", remediation_agent))
    g.add_edge(START, "retrieval")
    g.add_edge("retrieval", "triage")
    g.add_conditional_edges("triage", route_after_triage, {"no_action": "no_action", "root_cause": "root_cause"})
    g.add_edge("root_cause", "remediation")
    g.add_edge("remediation", END)
    g.add_edge("no_action", END)
    return g.compile()


if __name__ == "__main__":
    config.require_all()
    app = build_graph()
    result = app.invoke({
        "raw_log": "1125552602 2005.08.31 R23-M1-N8-I:J18-U01 2005-08-31-22.30.02.453491 R23-M1-N8-I:J18-U01 RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1",
        "notes": [],
    })
    print(json.dumps(result, indent=2, default=str))
