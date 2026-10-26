"""Drain3 wrapper, on BGL content ("component level message"). Each test gets
a fresh in-memory miner, so nothing here reads or writes drain3_state.bin."""
import pytest
from drain3 import TemplateMiner

from ingestion import parser

LUSTRE = "KERNEL FATAL Lustre mount FAILED : bglio{n} : point /p/gb1"
SEVERED = ("APP FATAL ciod: Error reading message prefix on CioStream socket to "
           "172.16.96.{n}:5337{n}, Link has been severed")
TLB = "KERNEL FATAL data TLB error interrupt"


@pytest.fixture(autouse=True)
def fresh_miner(monkeypatch):
    miner = TemplateMiner(config=parser._config)  # no persistence handler
    monkeypatch.setattr(parser, "_miner", miner)
    return miner


def test_lines_differing_only_in_numbers_share_a_cluster():
    a = parser.parse_line(LUSTRE.format(n=318))
    b = parser.parse_line(LUSTRE.format(n=42))
    assert a["cluster_id"] == b["cluster_id"]


def test_ips_and_numbers_are_masked():
    t = parser.parse_line(SEVERED.format(n=1))["template"]
    assert "<IP>" in t and "172.16" not in t


def test_different_events_get_different_clusters():
    a = parser.parse_line(LUSTRE.format(n=318))
    b = parser.parse_line(TLB)
    assert a["cluster_id"] != b["cluster_id"]


def test_match_finds_learned_cluster_for_new_values():
    learned = parser.parse_line(SEVERED.format(n=1))
    hit = parser.match_line(SEVERED.format(n=9))
    assert hit["cluster_id"] == learned["cluster_id"]
    assert hit["template"] == learned["template"]


def test_match_never_changes_the_miner(fresh_miner):
    parser.parse_line(LUSTRE.format(n=318))
    before = [(c.cluster_id, c.get_template(), c.size) for c in fresh_miner.drain.clusters]
    parser.match_line(LUSTRE.format(n=1))
    parser.match_line("something completely different happened here")
    after = [(c.cluster_id, c.get_template(), c.size) for c in fresh_miner.drain.clusters]
    assert before == after


def test_match_on_unknown_line_falls_back_to_raw_text():
    parser.parse_line(TLB)
    line = "kernel panic on node seven"
    hit = parser.match_line(line)
    assert hit["cluster_id"] is None
    assert hit["template"] == line
