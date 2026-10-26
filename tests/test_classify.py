"""Classifiers, with the LLM replaced. No database or model needed."""
from agents import classify


def hit(score, label):
    return {"score": score, "label": label, "template": "t"}


def test_exact_match_is_not_outvoted_by_weaker_normal_neighbours():
    # The case that broke the first eval run: one exact match, three ~0.75 normals.
    retrieved = [hit(1.0, "filesystem_mount_failure"), hit(0.753, "normal"),
                 hit(0.750, "normal"), hit(0.731, "filesystem_mount_failure"), hit(0.717, "normal")]
    vote = classify.knn_vote(retrieved)
    assert vote["label"] == "filesystem_mount_failure"
    assert vote["share"] == 1.0 and vote["top_score"] == 1.0


def test_close_neighbours_split_the_vote():
    vote = classify.knn_vote([hit(0.92, "normal"), hit(0.91, "kernel_crash"), hit(0.90, "kernel_crash")])
    assert vote["label"] == "kernel_crash"
    assert 0.6 < vote["share"] < 0.7


def test_no_labelled_neighbours():
    assert classify.knn_vote([])["label"] == "unknown"
    assert classify.knn_vote([hit(0.9, None)])["label"] == "unknown"


def test_level_rule():
    assert classify.level_rule("KERNEL FATAL data TLB error interrupt") == "alert"
    assert classify.level_rule("MONITOR FAILURE Temperature Over Limit") == "alert"
    assert classify.level_rule("KERNEL INFO generating core.123") == "normal"


def test_triage_uses_vote_when_confident(monkeypatch):
    monkeypatch.setattr(classify, "chat_text", lambda p: (_ for _ in ()).throw(AssertionError("no LLM")))
    out = classify.triage("line", [hit(1.0, "kernel_crash"), hit(0.7, "normal")])
    assert (out["classification"], out["method"]) == ("kernel_crash", "vote")


def test_triage_falls_back_to_llm_when_unsure(monkeypatch):
    prompts = []
    monkeypatch.setattr(classify, "chat_text", lambda p: prompts.append(p) or "power_control_failure")
    out = classify.triage("line", [hit(0.8, "normal"), hit(0.79, "kernel_crash")])
    assert (out["classification"], out["method"]) == ("power_control_failure", "llm")
    # the LLM is shown every category definition and the retrieved evidence
    assert "memory_cpu_fault:" in prompts[0] and "labelled kernel_crash" in prompts[0]
