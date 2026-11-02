"""
Loads the category file -- the single source of truth for the category list:
bgl/categories.json for BGL, another file per dataset (config.CATEGORIES_FILE).
Nothing else should hard-code category names: prompts, the eval, the router
and the UI (via the API) all ask this module.
"""
import json
from functools import lru_cache
from pathlib import Path

import config

_PATH = Path(__file__).resolve().parents[1] / config.CATEGORIES_FILE


@lru_cache(maxsize=1)
def all_categories() -> list[dict]:
    with open(_PATH, encoding="utf-8") as f:
        return json.load(f)["categories"]


def ids() -> list[str]:
    return [c["id"] for c in all_categories()]


def get(category_id: str) -> dict | None:
    return next((c for c in all_categories() if c["id"] == category_id), None)


def severity(category_id: str) -> str:
    """'none', 'low', 'medium' or 'high'. Unknown categories count as
    'medium' so they still get a diagnosis instead of being silently dropped."""
    c = get(category_id)
    return c["severity"] if c else "medium"


@lru_cache(maxsize=1)
def _code_map() -> dict[str, str]:
    return {code: c["id"] for c in all_categories() for code in c["codes"]}


def category_for_code(code: str) -> str | None:
    """BGL label token ('-', 'KERNDTLB', ...) -> category id, or None if the
    code isn't in categories.json."""
    return _code_map().get(code)


def majority(label_counts: dict | None) -> str | None:
    """An incident's label: the category most of its ingested lines had."""
    if not label_counts:
        return None
    return max(label_counts, key=label_counts.get)


def parse_llm_answer(text: str) -> str:
    """Map a free-text LLM reply to a category id. Accepts the id or the
    display name, with stray quotes/punctuation; 'unknown' if nothing fits.
    Longest match wins so 'normal' can't shadow a longer name."""
    t = text.strip().strip("`'\".").lower()
    by_key = {}
    for c in all_categories():
        by_key[c["id"].lower()] = c["id"]
        by_key[c["name"].lower()] = c["id"]
    if t in by_key:
        return by_key[t]
    for key in sorted(by_key, key=len, reverse=True):
        if key in t:
            return by_key[key]
    return "unknown"
