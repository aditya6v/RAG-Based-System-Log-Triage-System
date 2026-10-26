"""HTTP contract, with the graph replaced so no database or LLM is touched."""
import pytest

import app as app_module


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, state):
        if self.error:
            raise self.error
        return {**state, **self.result}


@pytest.fixture
def client():
    return app_module.app.test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_diagnose_requires_raw_log(client):
    r = client.post("/diagnose", json={})
    assert r.status_code == 400
    assert "raw_log" in r.get_json()["error"]


def test_diagnose_returns_the_graph_state(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_graph", lambda: FakeGraph(result={"classification": "normal operation"}))
    r = client.post("/diagnose", json={"raw_log": "line"})
    assert r.status_code == 200
    assert r.get_json()["classification"] == "normal operation"
    assert r.get_json()["raw_log"] == "line"


def test_pipeline_failure_comes_back_as_json(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_graph", lambda: FakeGraph(error=ConnectionError("qdrant down")))
    r = client.post("/diagnose", json={"raw_log": "line"})
    assert r.status_code == 500
    assert "qdrant down" in r.get_json()["error"]
