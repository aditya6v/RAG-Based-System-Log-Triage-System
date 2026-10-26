"""Drain3 wrapper. Each test gets a fresh in-memory miner, so nothing here
reads or writes drain3_state.bin."""
import pytest
from drain3 import TemplateMiner

from ingestion import parser

ALLOC = ("nova-compute.log.1 2017-05-16 00:01:12.009 2931 ERROR nova.compute.manager "
         "[instance: {iid}] Failed to allocate network for instance, no fixed IP addresses available.")
SPAWN = ("nova-compute.log.1 2017-05-16 00:00:04.500 2931 INFO nova.compute.manager "
         "[instance: {iid}] Took 0.26 seconds to spawn the instance on the hypervisor.")


@pytest.fixture(autouse=True)
def fresh_miner(monkeypatch):
    miner = TemplateMiner(config=parser._config)  # no persistence handler
    monkeypatch.setattr(parser, "_miner", miner)
    return miner


def test_lines_differing_only_in_ids_share_a_cluster():
    a = parser.parse_line(ALLOC.format(iid="2c88"))
    b = parser.parse_line(ALLOC.format(iid="44ab"))
    assert a["cluster_id"] == b["cluster_id"]
    assert "<*>" in b["template"]


def test_different_events_get_different_clusters():
    a = parser.parse_line(ALLOC.format(iid="2c88"))
    b = parser.parse_line(SPAWN.format(iid="2c88"))
    assert a["cluster_id"] != b["cluster_id"]


def test_match_finds_learned_cluster_for_a_new_instance_id():
    parser.parse_line(ALLOC.format(iid="2c88"))
    learned = parser.parse_line(ALLOC.format(iid="44ab"))
    hit = parser.match_line(ALLOC.format(iid="ffff"))
    assert hit["cluster_id"] == learned["cluster_id"]
    assert hit["template"] == learned["template"]


def test_match_never_changes_the_miner(fresh_miner):
    parser.parse_line(ALLOC.format(iid="2c88"))
    before = [(c.cluster_id, c.get_template(), c.size) for c in fresh_miner.drain.clusters]
    parser.match_line(ALLOC.format(iid="2c88"))
    parser.match_line("something completely different happened here")
    after = [(c.cluster_id, c.get_template(), c.size) for c in fresh_miner.drain.clusters]
    assert before == after


def test_match_on_unknown_line_falls_back_to_raw_text():
    parser.parse_line(ALLOC.format(iid="2c88"))
    line = "kernel panic on node 7"
    hit = parser.match_line(line)
    assert hit["cluster_id"] is None
    assert hit["template"] == line
