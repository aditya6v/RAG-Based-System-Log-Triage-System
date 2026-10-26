# Incident Copilot -- midterm scaffold

RAG-based multi-agent log triage, phase 1 of the 4-agent target: two
combined agents (triage+retrieval, root-cause+remediation) instead of
four, single Python service instead of a Node+Python split -- both
deliberate scope cuts for a 3-4 day build, not the final architecture.
See the file-by-file notes below for what each piece does, and the
"likely to break" section at the end before you demo.

## One-time setup (do this first)

1. **Python env**: `python3 -m venv venv && source venv/bin/activate`
   (Windows: `venv\Scripts\activate`), then `pip install -r requirements.txt`.
   Use a venv, not your system Python -- some of these packages have had
   build i  ssues on systems with unusual default Python setups.

2. **MongoDB Atlas** (free tier): create a cluster at mongodb.com/atlas,
   create a database user, and under **Network Access** add your current
   IP (or `0.0.0.0/0` if you'll be switching networks before the demo --
   fine for a throwaway midterm project, not something you'd do for
   anything real). Copy the `mongodb+srv://...` connection string.

3. **Qdrant Cloud** (free tier): create a cluster at cloud.qdrant.io. The
   dashboard shows both the cluster URL and an API key.

4. **Gemini API key**: aistudio.google.com -> Get API key. Free tier
   covers both the chat model and the embedding model, which is why
   everything here goes through one provider instead of two.

5. Copy `.env.example` to `.env` and fill in all five values.

## Run order

```
python -m ingestion.ingest sample_data/sample_logs.txt   # parse + store
python -m embeddings.embed                                 # embed + index
python -m agents.graph                                       # sanity-check the pipeline alone
python app.py                                                  # start the HTTP service
```

Then in another terminal:
```
curl -X POST http://localhost:5000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"raw_log": "nova-compute.log.1 2017-05-16 03:12:09.112 2931 ERROR nova.compute.manager [instance: aa21] Failed to allocate network for instance, no fixed IP addresses available."}'
```

Swap `sample_data/sample_logs.txt` for a real slice of the Loghub
OpenStack dataset (github.com/logpai/loghub) once the pipeline runs
end-to-end on the fake data -- don't point at real data until the fake
data path works, so you know any new failure is about the real data, not
the plumbing.

## What each file does

- **config.py** -- every env var, read once, in one place. `require_all()`
  fails loudly at startup if a key's missing, instead of failing
  confusingly three calls deep.
- **drain3.ini** -- Drain3's tuning knobs. `sim_th` is the one you'll
  actually touch (see comments in the file).
- **ingestion/parser.py** -- wraps Drain3: raw log line in, `{cluster_id,
  template}` out. Owns the one shared `TemplateMiner` instance, persisted
  to `drain3_state.bin` so ingestion and the live app agree on what a
  given template looks like.
- **ingestion/ingest.py** -- batch script. Reads a log file, writes one
  `events` doc per line and upserts one `incidents` doc per unique
  template into Mongo.
- **db/mongo_client.py**, **db/qdrant_client.py** -- connection wrappers,
  nothing else in the codebase talks to Mongo or Qdrant directly except
  through these.
- **embeddings/embed.py** -- batch script. Embeds every incident that
  doesn't have a Qdrant vector yet. Safe to re-run.
- **agents/llm.py** -- the only file that calls ollama(primary) and gemin(alternative) . Two functions:
  `chat_text`, `embed_text`.
- **agents/state.py** -- the `IncidentState` TypedDict that flows through
  the graph.
- **agents/graph.py** -- the pipeline itself. `triage_and_retrieve`
  (classify + semantic search), `diagnose_and_remediate` (grounded root
  cause + fix). Run directly for fast iteration without Flask in the way.
- **app.py** -- `/diagnose` HTTP endpoint, thin wrapper around the
  compiled graph. `/health` for a quick "is it up" check.
- **eval/evaluate.py** -- precision/recall of the classification step
  against a small hand-labeled set, via scikit-learn.

## What's likely to break, and how to fix it

- **Mongo hangs instead of erroring**: your current IP isn't allow-listed
  in Atlas, or the connection string's password has an unescaped special
  character. This is the single most common "it worked an hour ago" bug.
- **Qdrant upsert fails with a dimension error**: you changed
  `GEMINI_EMBED_MODEL` without updating `config.EMBEDDING_DIM` to match.
- **Retrieval returns nothing relevant**: either `embeddings/embed.py`
  hasn't been run since the last ingestion, or `drain3_state.bin` is
  missing/stale so the live app's templates don't line up with what was
  embedded. Re-run ingestion, then embedding, in order.
- **Gemini 429 errors**: free-tier rate limit. `embeddings/embed.py`
  already sleeps between calls; raise the sleep if you still hit it, or
  don't run ingestion+embedding back-to-back on a large batch right
  before you're about to demo.
- **The diagnosis response has a `root_cause` that's just raw text instead
  of the expected fields**: Gemini didn't return clean JSON. This is
  handled (see `_parse_json_response` in `agents/graph.py`) by falling
  back to showing the raw text rather than crashing the request --
  acceptable for a demo, but if it happens often, tighten the prompt or
  lower temperature.
- **A node seems to silently not run / state looks stale**: print
  `state["notes"]` -- every node appends to it, so it tells you exactly
  how far execution got.

## Deliberately deferred (don't build these before the demo)

- Conditional routing / branching in the graph -- the two nodes are
  wired linearly; a router node that picks different downstream paths by
  severity is a natural phase-2 addition, not needed to demonstrate the
  RAG-grounding concept.
- FastAPI migration -- Flask is fine for a demo; FastAPI is worth it
  once you have time to learn it properly, not during a 3-4 day crunch.
- Splitting into 4 separate agents -- explicitly framed as "phase 1 of 4"
  in the presentation, not a hidden shortcut.
