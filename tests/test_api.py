"""HTTP contract for the FastAPI service, with the graph replaced so no
database or LLM is touched. TestClient is not used as a context manager, so
the startup hook (which needs .env) doesn't run."""
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_graph


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result, self.error = result or {}, error

    def invoke(self, state):
        if self.error:
            raise self.error
        return {**state, **self.result}


FULL = {
    "text": "line", "cluster_id": "7", "template": "t",
    "classification": "filesystem_mount_failure", "severity": "high",
    "triage_method": "vote", "vote": {"label": "filesystem_mount_failure", "share": 1.0, "top_score": 1.0},
    "route": "full",
    "retrieved": [{"cluster_id": "7", "score": 1.0, "template": "t", "example_raw": "e",
                   "occurrence_count": 3, "label": "filesystem_mount_failure"}],
    "evidence": {"primary": {"cluster_id": "7", "score": 1.0, "template": "t"}, "supporting": []},
    "root_cause": "rc", "remediation": "fix",
    "notes": ["a", "b"], "timings_ms": {"retrieval": 5, "triage": 1},
}


@pytest.fixture
def client():
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def use_graph(graph):
    app.dependency_overrides[get_graph] = lambda: graph


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_categories_come_from_the_category_file(client):
    cats = client.get("/categories").json()
    assert len(cats) == 8 and cats[0]["id"] == "normal"
    assert {"id", "name", "severity", "definition"} <= set(cats[0])


@pytest.mark.parametrize("body", [{}, {"raw_log": ""}, {"raw_log": 5}, {"wrong": "x"}])
def test_bad_request_is_400_json(client, body):
    use_graph(FakeGraph())
    r = client.post("/diagnose", json=body)
    assert r.status_code == 400
    assert "raw_log" in r.json()["error"]


def test_diagnose_returns_the_full_contract(client):
    use_graph(FakeGraph(result=FULL))
    r = client.post("/diagnose", json={"raw_log": "line"})
    assert r.status_code == 200
    d = r.json()
    assert d["classification"] == "filesystem_mount_failure"
    assert d["category_name"] == "filesystem mount failure"
    assert d["route"] == "full" and d["severity"] == "high"
    assert d["retrieved"][0]["score"] == 1.0
    assert d["evidence"]["primary"]["cluster_id"] == "7"
    assert d["timings_ms"] == {"retrieval": 5, "triage": 1}
    assert isinstance(d["total_ms"], int) and d["raw_log"] == "line"


def test_pipeline_failure_is_500_json(client):
    use_graph(FakeGraph(error=ConnectionError("qdrant down")))
    r = client.post("/diagnose", json={"raw_log": "line"})
    assert r.status_code == 500
    assert "qdrant down" in r.json()["error"]


def test_front_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Log Triage" in r.text


def test_openapi_docs_exist(client):
    assert client.get("/docs").status_code == 200
    assert "/diagnose" in client.get("/openapi.json").json()["paths"]
