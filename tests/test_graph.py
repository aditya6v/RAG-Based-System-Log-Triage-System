"""Pipeline pieces that don't need a database or an LLM."""
import pytest

from agents import graph
from agents.graph import _parse_json_response


@pytest.mark.parametrize("text", [
    '{"root_cause": "a", "remediation": "b"}',
    '```json\n{"root_cause": "a", "remediation": "b"}\n```',
    '```\n{"root_cause": "a", "remediation": "b"}\n```',
    'Here is the JSON:\n{"root_cause": "a", "remediation": "b"}\nHope that helps.',
])
def test_parse_json_handles_common_llm_wrappings(text):
    assert _parse_json_response(text) == {"root_cause": "a", "remediation": "b"}


@pytest.mark.parametrize("text", ["not json at all", "{broken", "[1, 2]", ""])
def test_parse_json_falls_back_to_raw_text(text):
    assert _parse_json_response(text) == {"_unparsed": text}


def test_normal_logs_skip_diagnosis_without_calling_the_llm(monkeypatch):
    def no_llm(prompt):
        raise AssertionError("LLM should not be called for a normal log")
    monkeypatch.setattr(graph, "chat_text", no_llm)

    out = graph.diagnose_and_remediate({"raw_log": "x", "classification": "normal operation", "notes": []})
    assert out["remediation"] == "No action needed."
    assert "skipped" in out["notes"][-1]


def test_failures_get_a_diagnosis_from_the_llm(monkeypatch):
    monkeypatch.setattr(graph, "chat_text", lambda p: '{"root_cause": "rc", "remediation": "fix"}')
    out = graph.diagnose_and_remediate({
        "raw_log": "x", "classification": "network allocation failure", "notes": [],
        "retrieved": [{"score": 0.9, "occurrence_count": 3, "template": "t"}],
    })
    assert (out["root_cause"], out["remediation"]) == ("rc", "fix")
