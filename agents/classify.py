"""
The classifiers compared in the eval, and used by the triage agent.

  level_rule  -- baseline: FATAL/FAILURE level means alert. Binary only.
  knn_vote    -- nearest-neighbour vote: retrieved incidents vote with their
                 majority label, weighted by similarity. No LLM call.
  llm_classify -- the LLM picks a category from bgl/categories.json, either
                 on its own or shown the retrieved incidents and their labels.
  triage      -- what the app uses: the vote when it is confident, the LLM
                 (with retrieved incidents) when it isn't.
"""
from collections import defaultdict

from agents.llm import chat_text
from bgl import categories
from bgl.lines import level_of

ALERT_LEVELS = {"FATAL", "FAILURE"}

# Vote is trusted when the best match is this similar and this share of the
# (similarity-weighted) votes agree. Set from eval results -- see PROGRESS.md.
CONFIDENT_TOP_SCORE = 0.9
CONFIDENT_SHARE = 0.6


def level_rule(content: str) -> str:
    """'alert' or 'normal'. Every BGL alert is FATAL or FAILURE, so this has
    perfect alert recall; the question is how many normal lines it flags."""
    return "alert" if level_of(content) in ALERT_LEVELS else "normal"


# Only neighbours within this much of the best match get a vote. Without it,
# an exact match (1.0, right label) is outvoted by three ~0.75 normal
# incidents, because 312 of the 357 BGL incidents are normal -- the first
# eval run lost 136 alert lines that way (see PROGRESS.md).
VOTE_MARGIN = 0.05


def knn_vote(retrieved: list[dict]) -> dict:
    best = max((r["score"] for r in retrieved), default=0.0)
    weights = defaultdict(float)
    for r in retrieved:
        if r.get("label") and r["score"] >= best - VOTE_MARGIN:
            weights[r["label"]] += max(r["score"], 0.0)
    total = sum(weights.values())
    if not total:
        return {"label": "unknown", "share": 0.0, "top_score": 0.0}
    label = max(weights, key=weights.get)
    return {
        "label": label,
        "share": round(weights[label] / total, 3),
        "top_score": max((r["score"] for r in retrieved), default=0.0),
    }


def is_confident(vote: dict) -> bool:
    return vote["top_score"] >= CONFIDENT_TOP_SCORE and vote["share"] >= CONFIDENT_SHARE


def _category_block() -> str:
    return "\n".join(f"- {c['id']}: {c['definition']}" for c in categories.all_categories())


def llm_classify(text: str, retrieved: list[dict] | None = None) -> str:
    evidence = ""
    if retrieved:
        rows = "\n".join(
            f"- similarity {r['score']}, labelled {r.get('label') or 'unknown'}: {r['template']}"
            for r in retrieved
        )
        evidence = f"\nSimilar past log events from this system, with their known category:\n{rows}\n"

    prompt = f"""You are a log triage assistant for the BlueGene/L supercomputer.
Classify the log line into EXACTLY ONE of these category ids:

{_category_block()}

Important: many normal lines are logged at FATAL or ERROR level. The level
alone does not make a line an alert -- judge by what the message describes.
{evidence}
Log line: {text}

Respond with ONLY the category id from the list above, nothing else."""
    return categories.parse_llm_answer(chat_text(prompt))


def triage(text: str, retrieved: list[dict]) -> dict:
    """Returns {"classification", "method", "vote"}. method is 'vote' or 'llm'."""
    vote = knn_vote(retrieved)
    if is_confident(vote):
        return {"classification": vote["label"], "method": "vote", "vote": vote}
    return {"classification": llm_classify(text, retrieved), "method": "llm", "vote": vote}
