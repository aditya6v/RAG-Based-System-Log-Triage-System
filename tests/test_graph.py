"""Pipeline pieces and the whole compiled graph, with retrieval and the LLM
replaced -- no database or model needed."""
import pytest

from agents import classify, graph
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


@pytest.mark.parametrize("severity,route", [
    ("none", "no_action"), ("low", "no_action"), ("medium", "root_cause"), ("high", "root_cause"),
    (None, "root_cause"),
])
def test_router_is_deterministic_on_severity(severity, route):
    assert graph.route_after_triage({"severity": severity}) == route


def hit(score, label, cid="1"):
    return {"cluster_id": cid, "score": score, "label": label, "template": f"t{cid}",
            "example_raw": "", "occurrence_count": 3}


def test_evidence_keeps_top_hit_as_primary_and_drops_weak_ones():
    ev = graph.select_evidence([hit(1.0, "a", "1"), hit(0.9, "b", "2"), hit(0.75, "c", "3")])
    assert ev["primary"]["cluster_id"] == "1"
    assert [r["cluster_id"] for r in ev["supporting"]] == ["2"]


def test_supporting_evidence_must_share_the_triage_label():
    retrieved = [hit(1.0, "kernel_crash", "1"), hit(0.95, "memory_cpu_fault", "2"), hit(0.9, "kernel_crash", "3")]
    ev = graph.select_evidence(retrieved, "kernel_crash")
    assert ev["primary"]["cluster_id"] == "1"
    assert [r["cluster_id"] for r in ev["supporting"]] == ["3"]


def test_evidence_empty_when_nothing_is_similar_enough():
    assert graph.select_evidence([hit(0.7, "a")]) == {"primary": None, "supporting": []}


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace retrieval and every LLM call; record the prompts sent."""
    prompts = []

    def fake_retrieve(raw_log):
        label = "normal" if "normal" in raw_log else "filesystem_mount_failure"
        return {"text": raw_log, "content": raw_log, "cluster_id": "7", "template": "t7",
                "retrieved": [hit(1.0, label, "7"), hit(0.9, label, "6"),
                              hit(0.88, "normal", "8"), hit(0.7, "normal", "9")]}

    def fake_llm(prompt):
        prompts.append(prompt)
        return '{"root_cause": "rc", "remediation": "fix"}'

    monkeypatch.setattr(graph, "retrieve", fake_retrieve)
    monkeypatch.setattr(graph, "chat_text", fake_llm)
    monkeypatch.setattr(classify, "chat_text", fake_llm)
    return prompts


def test_normal_log_skips_root_cause_and_remediation(fake_pipeline):
    out = graph.build_graph().invoke({"raw_log": "a normal line", "notes": []})
    assert out["classification"] == "normal" and out["route"] == "no_action"
    assert out["remediation"] == "No action needed."
    assert fake_pipeline == []  # vote was confident, no LLM call at all
    assert set(out["timings_ms"]) == {"retrieval", "triage", "no_action"}


def test_failure_goes_through_all_four_agents(fake_pipeline):
    out = graph.build_graph().invoke({"raw_log": "lustre mount failed", "notes": []})
    assert out["classification"] == "filesystem_mount_failure"
    assert out["severity"] == "high" and out["route"] == "full"
    assert (out["root_cause"], out["remediation"]) == ("rc", "fix")
    assert set(out["timings_ms"]) == {"retrieval", "triage", "root_cause", "remediation"}
    # root cause prompt: primary vs weak evidence; drops the 0.7 hit (too far)
    # and the 0.88 normal hit (different label from the triage result)
    rc_prompt = fake_pipeline[0]
    assert "PRIMARY evidence" in rc_prompt and "WEAK supporting" in rc_prompt
    assert "t6" in rc_prompt and "t8" not in rc_prompt and "t9" not in rc_prompt
    # remediation sees the root cause
    assert "root cause: rc" in fake_pipeline[1]
    assert len(out["notes"]) == 4
