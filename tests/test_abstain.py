"""Abstain threshold: chosen on validation, applied to test."""
from eval.abstain import at, choose


def row(conf, right):
    return {"conf": conf, "true_label": "a", "pred": {"knn_vote": "a" if right else "b"}}


ROWS = [row(1.0, True), row(0.95, True), row(0.9, True), row(0.8, False), row(0.7, True), row(0.6, False)]


def test_precision_and_coverage_at_a_threshold():
    assert at(ROWS, 0.0) == (4 / 6, 1.0, 6)
    assert at(ROWS, 0.85) == (1.0, 0.5, 3)


def test_lowest_threshold_reaching_the_target_is_chosen():
    assert choose(ROWS, 0.95) == 0.9     # 0.8 would let a wrong answer in
    assert choose(ROWS, 0.75) == 0.7     # 0.7: 4 of 5 answered right = 0.8


def test_nothing_answered_counts_as_full_precision_zero_coverage():
    assert at(ROWS, 1.01) == (1.0, 0.0, 0)
