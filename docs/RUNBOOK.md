# Demo runbook and backup plan

What to do before a demo, what can go wrong, and what to fall back to. Every timing
here was measured. The last full check was 2026-11-02: all services up, 6/6 demo requests
correct, 54 tests passing.

## Known-good versions

| What | How to get it |
|---|---|
| ETE version (as presented) | `git checkout ete-final` |
| Midterm version | `git checkout midterm` |
| Work in progress | branch `resume-phase` |

If a change breaks something right before a demo, run `git stash` (or commit it) and then
`git checkout ete-final`. Nothing else needs rebuilding: the databases didn't change
between these versions.

## Pre-demo checklist

**The day before**
1. Open cloud.qdrant.io and make sure the cluster is **running**. Free clusters
   suspend after a few weeks unused, and resuming needs the dashboard.
2. In MongoDB Atlas, under Network Access, check that the network you'll demo from is
   allowed (a new Wi-Fi network means a new IP).
3. `python -m pytest -q`: expect 95 passed.

**30 minutes before**
1. Start the Ollama app.
2. `.\venv\Scripts\Activate.ps1`, then `uvicorn api.main:app --port 8000`.
3. Open http://localhost:8000 and click **"FATAL but normal"**, then **Diagnose**.
   The first request takes about 10 s while Ollama loads the models. Then click
   **"Lustre mount failed"** once, which loads the chat model (another ~8 s).
   After that: a normal line takes about 0.2–0.7 s, and a failure 5.5–7.5 s.
4. Models stay loaded for 30 minutes after the last request (`OLLAMA_KEEP_ALIVE`). If
   there's a long gap, send one more request just before presenting.

## Faults: symptom, cause, fix

| Symptom | Likely cause | Fix (time) |
|---|---|---|
| Page says **"API offline"** / "Could not reach the API" | Server not running or crashed | Restart `uvicorn api.main:app --port 8000` (5 s) |
| 500: `Connection was reset` mentioning Qdrant | Qdrant free cluster suspended | Resume it in the Qdrant dashboard (1–2 min). If the cluster was **deleted**, see "Rebuild the databases" |
| 500 or a hang mentioning Mongo / `ServerSelectionTimeout` | Your IP isn't allowed in Atlas (new network) | Atlas, then Network Access, then add the current IP (1 min). Or switch to a hotspot that's already allowed |
| 500: connection refused on 11434 | Ollama isn't running | Start the Ollama app (10 s), then send a warm-up request |
| 500: `model not found` | Model not pulled on this machine | `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text` (a few minutes, needs internet) |
| Every request takes 8–12 s | Models were unloaded after an idle gap | Expected once. Send a warm-up request |
| Port 8000 in use | An old server is still running | `uvicorn api.main:app --port 8001` and open that port |
| Laptop very slow or out-of-memory errors | RAM is short (this laptop has had 1.9 GB free) | Close browsers and VS Code windows you don't need; restart Ollama |
| Results look wrong (everything "normal", low similarity scores) | Index and Drain3 state out of step | `python -m ingestion.ingest sample_data/bgl_index.log`, then `python -m embeddings.embed --rebuild` |

## Fallback levels

Use the first level that works.

1. **Live, on the laptop.** The normal path, as above.
2. **Live, in Docker.** `docker compose up --build` (the first build took about 10 min, so do it
   ahead of time). It still needs Atlas, Qdrant Cloud and Ollama on the host.
3. **Live, on another machine.** Clone from GitHub, create the venv, copy `.env` (keep a copy
   of it somewhere safe, such as a password manager, because it's **not** in git), install
   and pull the Ollama models. The databases are in the cloud, so nothing needs rebuilding.
4. **No live system at all.** Show the recorded evidence. Everything in it came from real runs:
   - `docs/img/diagnosis.png`: a full diagnosis on the page (I/O node connection lost)
   - `docs/img/normal-line.png`: a normal FATAL-level line, with root cause and fix skipped
   - `docs/img/api-unreachable.png`: what the page shows when the API is down
   - `eval/results/20261026-153603.md`: the full 623-line evaluation report
   - `eval/results/samples-stage3-http.json`, `samples-stage4-docker.json`: demo requests
     and their full responses
   - `docs/EVALUATION.md` and `docs/ETE_NOTES.md`: all numbers and how they were produced

## Rebuild the databases (if a cloud cluster is lost)

Everything needed is in the repo. Measured on 2026-10-26: ingest 2m05s, embed 1m28s.

1. Create a new free Qdrant cluster (and/or Atlas cluster), and put the new URL/key in `.env`.
2. `python -m ingestion.ingest sample_data/bgl_index.log`
3. `python -m embeddings.embed`
4. Check: Mongo `incidents` and Qdrant both show **357**. Then run the warm-up request.

You do *not* need `BGL.log` for this. It's only needed to re-sample (`scripts/sample_bgl.py`).

## What is and isn't backed up

| Thing | Where | If lost |
|---|---|---|
| Code, eval sets, index sample, Drain3 state, results, docs | git (GitHub once pushed) | Clone |
| `.env` secrets | your machine only | Keep a copy in a password manager; keys can be regenerated in each console |
| Mongo + Qdrant contents | cloud free tiers | Rebuild from the repo (above), about 4 min |
| `data/bgl/BGL.log` (743 MB) | your machine only | Re-download from https://zenodo.org/records/8196385 |
| Ollama models | your machine | `ollama pull` again |

## Not covered yet (known gap)

The system has **no fully offline mode**: Atlas and Qdrant Cloud must be reachable. A
fallback to local storage (embedded Qdrant on disk, with labels and counts read from the
vector payload instead of Mongo) is possible but not built. It's listed as a candidate
next step, not something to promise in a demo.
