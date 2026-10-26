"""
The HTTP service. Run with:

    uvicorn api.main:app --port 8000          (or: python -m api.main)

Then open http://localhost:8000 for the page, http://localhost:8000/docs for
the auto-generated API docs.

Deliberately thin: this file knows nothing about Drain3, Mongo, Qdrant or the
LLM. It validates the request, hands raw_log to the compiled graph (built
once at startup), and shapes the result into DiagnoseResponse. Debug
pipeline logic in agents/graph.py (python -m agents.graph), not through HTTP.

Errors always come back as JSON {"error": "..."}: 400 for a bad request,
500 when something in the pipeline fails (most often a database or the LLM
being unreachable -- the message says which).
"""
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from bgl import categories

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


# --- models -------------------------------------------------------------------

class DiagnoseRequest(BaseModel):
    raw_log: str = Field(min_length=1, description="One BGL log line, with or without the leading label token",
                         examples=["1118538627 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-18.10.27.692687 "
                                   "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt"])


class Incident(BaseModel):
    cluster_id: str | None = None
    score: float
    template: str = ""
    example_raw: str = ""
    occurrence_count: int | None = None
    label: str | None = None


class Evidence(BaseModel):
    primary: Incident | None = None
    supporting: list[Incident] = []


class Vote(BaseModel):
    label: str
    share: float
    top_score: float


class DiagnoseResponse(BaseModel):
    raw_log: str
    text: str = ""
    cluster_id: str | None = None
    template: str = ""
    classification: str
    category_name: str
    severity: str
    triage_method: str = ""
    vote: Vote | None = None
    route: str
    retrieved: list[Incident] = []
    evidence: Evidence = Evidence()
    root_cause: str = ""
    remediation: str = ""
    notes: list[str] = []
    timings_ms: dict[str, int] = {}
    total_ms: int


class Category(BaseModel):
    id: str
    name: str
    severity: str
    definition: str


# --- app ----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once, not per request. Fails at startup (not on the first
    # request) if .env is missing a key.
    config.require_all()
    from agents.graph import build_graph
    app.state.graph = build_graph()
    yield


app = FastAPI(
    title="Incident Copilot",
    description="RAG-based multi-agent triage for BlueGene/L logs: retrieval, triage, root cause, remediation.",
    version="2.0",
    lifespan=lifespan,
)


def get_graph(request: Request):
    return request.app.state.graph


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=400, content={"error": f"expected JSON body with a non-empty 'raw_log' string ({msg})"})


@app.exception_handler(Exception)
async def server_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories", response_model=list[Category])
def list_categories():
    return [Category(**{k: c[k] for k in ("id", "name", "severity", "definition")}) for c in categories.all_categories()]


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(body: DiagnoseRequest, graph=Depends(get_graph)):
    # Plain `def`: FastAPI runs it in a worker thread, so the blocking graph
    # call doesn't stall the event loop.
    t0 = time.perf_counter()
    state = graph.invoke({"raw_log": body.raw_log, "notes": []})
    cat = categories.get(state.get("classification", "")) or {}
    return DiagnoseResponse(
        **{k: v for k, v in state.items() if k in DiagnoseResponse.model_fields},
        category_name=cat.get("name", state.get("classification", "unknown")),
        total_ms=round((time.perf_counter() - t0) * 1000),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
