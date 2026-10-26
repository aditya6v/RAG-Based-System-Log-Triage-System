# Incident Copilot

RAG-based multi-agent triage for supercomputer logs. Paste one log line from
the BlueGene/L (BGL) system and four agents work through it:

```
START -> retrieval -> triage --(router)--> root_cause -> remediation -> END
                                  \
                                   +--> no_action -> END   (normal / low severity)
```

| Agent | What it does | LLM? |
|---|---|---|
| retrieval | Drain3 template match (read-only), embed, Qdrant top-5, Mongo counts and labels | embedding only |
| triage | Category + severity. Nearest-neighbour vote of the retrieved incidents when it is confident, otherwise the LLM | only when the vote is unsure |
| router | Plain function: severity from `bgl/categories.json` decides the path | no |
| root_cause | Grounded explanation: the top hit is primary evidence, hits above 0.85 similarity are weak support, the rest are dropped | yes |
| remediation | Concrete fix, given the root cause and the category definition | yes |

Data: the Loghub BGL dataset (4,747,963 lines, 41 alert codes grouped into 8
categories). Evaluation results, design decisions and limitations are in
`docs/ETE_NOTES.md`; build history in `PROGRESS.md`.

## Setup (once)

1. **Python env** (3.12): `python -m venv venv`, then `.\venv\Scripts\Activate.ps1`
   (if scripts are blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`),
   then `pip install -r requirements.txt`.
2. **Ollama** (default LLM, local): install it, then
   `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`.
3. **MongoDB Atlas** free cluster: create a database user, allow your IP under
   Network Access, copy the `mongodb+srv://...` string.
4. **Qdrant Cloud** free cluster: copy the cluster URL and API key. Free
   clusters are **suspended after a few weeks unused**; open the dashboard a
   day before a demo.
5. Copy `.env.example` to `.env` and fill in the values in `<...>`.
   Gemini is optional (`LLM_PROVIDER=gemini`).

## Build the index (once, or after changing data or Drain3 settings)

Download `BGL.zip` from https://zenodo.org/records/8196385 and extract
`BGL.log` to `data/bgl/` (ignored by git, about 710 MB).

```
python -I scripts/sample_bgl.py data/bgl/BGL.log   # -> sample_data/bgl_index.log, eval/bgl_eval.jsonl
python -m ingestion.ingest sample_data/bgl_index.log  # Drain3 templates -> Mongo
python -m embeddings.embed                            # templates -> Qdrant
```

The sample and eval files are committed, so the first step is only needed to
regenerate them. To start from scratch, delete `drain3_state.bin` and use an
empty `MONGO_DB_NAME` / `QDRANT_COLLECTION` first. After re-ingesting, run
`python -m embeddings.embed --rebuild`.

## Run

```
uvicorn api.main:app --port 8000      # or: python -m api.main
```

- http://localhost:8000: the page (sample buttons, history, per-agent timings)
- http://localhost:8000/docs: interactive API docs
- `POST /diagnose` with `{"raw_log": "<one BGL line>"}`; `GET /categories`; `GET /health`

```
curl -X POST http://localhost:8000/diagnose -H "Content-Type: application/json" \
  -d '{"raw_log": "1125552602 2005.08.31 R23-M1-N8-I:J18-U01 2005-08-31-22.30.02.453491 R23-M1-N8-I:J18-U01 RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1"}'
```

### Docker

```
docker compose up --build             # API in a container, Ollama on the host
docker compose --profile ollama up    # also run Ollama in a container (slower, no GPU)
```

MongoDB Atlas and Qdrant Cloud stay external; the container reads `.env`.

## Test and evaluate

```
python -m pytest -q                                  # unit + API contract tests, no DB or LLM needed
python -m eval.evaluate eval/bgl_eval.jsonl          # full eval, about 25 min on local Ollama
python -m eval.evaluate eval/bgl_eval.jsonl --no-llm # retrieval-only classifiers, about 3 min
python -m eval.evaluate --report eval/results/<run>.json   # rebuild a report without re-running
python -m agents.graph                               # one line through the whole pipeline
```

## Code map

- `bgl/categories.json`: the 8 categories, definitions, severities, alert codes. The one place they are defined.
- `bgl/lines.py`: splits a BGL line; the ground-truth label token never reaches Drain3, embeddings or the LLM.
- `ingestion/parser.py`: Drain3 wrapper. `parse_line` learns (ingestion only); `match_line` only looks up (diagnose, eval).
- `ingestion/ingest.py`, `embeddings/embed.py`: batch jobs that build the index.
- `agents/retrieval.py`, `agents/classify.py`: shared by the graph and the eval.
- `agents/graph.py`: the four agents, router, evidence selection, per-agent timing.
- `agents/llm.py`: the only file that calls an LLM (Ollama or Gemini).
- `api/main.py`: FastAPI app, Pydantic models, JSON errors.
- `frontend/index.html`: single page, no dependencies.
- `eval/evaluate.py`: compares the FATAL rule, nearest-neighbour vote, LLM-only, LLM+RAG and triage.
- `scripts/`: BGL scan and sampling scripts. Run them with `python -I`.

## When something breaks

- **Request hangs, then a 500 about Mongo**: your IP isn't allowed in Atlas, or the
  password in `MONGO_URI` has an unescaped `@ : /`.
- **500 "Connection was reset" from Qdrant**: the free cluster is suspended. Resume it in the dashboard.
- **Connection refused on 11434**: Ollama isn't running. **"model not found"**: `ollama pull` it.
- **Similarity scores all low / retrieval nonsense**: embeddings were made with a
  different model, or Drain3 state doesn't match what was embedded. Re-run
  ingest, then `embed --rebuild`.
- **First request slow**: Ollama is loading the model into memory; later calls are faster.
