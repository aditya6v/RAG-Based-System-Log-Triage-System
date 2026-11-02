# Evaluation

All numbers on this page come from commands that were run. Each result names the
command and the file it was saved to. Hardware: a Windows laptop with 16 GB RAM, CPU only.
Models: `llama3.2:3b` for chat and `nomic-embed-text` (768-dim) for embeddings, both served
by Ollama.

## 1. Data

**Source:** the BGL (BlueGene/L) log from Loghub [1, 2]. A full streaming scan
(`scripts/scan_bgl.py`, `scripts/scan_bgl_overlap.py`) found:

| Fact | Value |
|---|---|
| Lines | 4,747,963 (0 decode errors) |
| Alert lines (label token is not `-`) | 348,460 (7.3%) |
| Alert codes | 41, over 215 days |
| Normal lines with no message text | 34,470 (skipped) |
| Distinct alert messages after masking numbers | 88 in total; 26 codes have exactly one |
| Normal lines logged at FATAL / ERROR / SEVERE / FAILURE | about 618,000 |

Two things follow:
- Alert text barely varies, so recognising a known alert is close to a lookup.
- Many normal lines look alarming, so a rule based on the log level is not enough. These
  are called **hard negatives** below.

**Categories.** The 41 codes are grouped by meaning into 7 alert categories plus `normal`
(`bgl/categories.json`). KERNPROG is mapped to `normal` because its only message also
appears 3,662 times labelled normal (5 times as KERNPROG).

**Sample and split** (`scripts/sample_bgl.py`, seed 7, one streaming pass):
- Each line gets a pattern key: category + component + level + message with numbers masked.
- About 10% of normal patterns and 20% of alert patterns per category (never the most
  common one) are held out of the index. Their lines form the **unseen** test subset.
- The **seen** subset is a reservoir sample [9] per category, so it follows the real
  frequency of each message.
- Index and test share no line; the script asserts this.

| Category | Index lines | Test: seen | Test: unseen |
|---|---|---|---|
| normal | 15,520 | 200 | 100 |
| memory_cpu_fault | 311 | 40 | 9 |
| kernel_crash | 311 | 40 | 8 |
| io_connection_lost | 309 | 40 | 6 |
| filesystem_mount_failure | 303 | 40 | 3 |
| job_launch_failure | 305 | 40 | 6 |
| link_hardware_failure | 183 | 32 | 5 |
| power_control_failure | 310 | 40 | 14 |

The test set has 623 lines in total, including 108 hard negatives. The label token is
removed before the parser, the embedding model or the LLM see a line (`bgl/lines.py`,
covered by tests).

**Template parsing.** Drain [3] via Drain3 [4], with masking for hex values, IPs and numbers.
`sim_th` was set by measuring, on the index, the share of lines whose cluster's majority
label matches their own:

| sim_th | Clusters | Purity |
|---|---|---|
| 0.4 | 237 | 98.3% |
| 0.5 | 268 | 99.5% |
| 0.6 | 309 | 99.6% |
| 0.7 | 357 | 99.97% |
| 0.8 | 410 | 99.98% |

0.7 is used. At 0.4, short messages merge on the words "KERNEL FATAL" alone.

## 2. Classification results

Command: `python -m eval.evaluate eval/bgl_eval.jsonl`. Results:
`eval/results/20261026-153603.json` (every prediction) and `.md` (full report,
including per-class scores and confusion matrices). Run time: 1,609 s.

All classifiers see exactly the same retrieval results:

| Classifier | What it does |
|---|---|
| level_rule | FATAL or FAILURE means alert (binary only) |
| knn_vote | Similarity-weighted vote of the 5 retrieved incidents' labels; only hits within 0.05 of the best match vote |
| llm_only | The LLM picks a category from the definitions alone |
| llm_rag | The LLM also sees the 5 retrieved incidents and their labels |
| triage | Used by the service: knn_vote when the top score is at least 0.9 and the vote share at least 0.6, otherwise llm_rag |

| Classifier | Accuracy (8 classes) | Macro F1 | Accuracy, unseen (151) | Alert vs normal accuracy | Hard negatives correct (108) |
|---|---|---|---|---|---|
| level_rule | n/a | n/a | n/a | 0.876 | 0.287 |
| knn_vote | 0.981 | 0.981 | 0.921 | 0.981 | 1.000 |
| llm_only | 0.425 | 0.446 | 0.212 | 0.650 | 0.000 |
| llm_rag | 0.639 | 0.755 | 0.570 | 0.727 | 0.028 |
| **triage** | **0.982** | 0.974 | **0.927** | **0.997** | 0.981 |

Triage F1 per class: normal 0.997, memory_cpu_fault 0.893 (precision 0.852), kernel_crash
0.970, io_connection_lost 1.000, filesystem_mount_failure 1.000, job_launch_failure 1.000,
link_hardware_failure 0.958 (recall 0.919), power_control_failure 0.971.

**How to read this**
- On the 472 **seen** lines, knn_vote and triage both score 1.000. That is close to a lookup,
  so the meaningful numbers are the **unseen** subset and the **hard negatives**. Unseen
  alert subsets are small (3–14 lines per category), so per-class unseen numbers are
  indicative only.
- The 3B model on its own labels every FATAL-level normal line as an alert. Showing it the
  retrieved incidents helps (0.425 to 0.639) but it still trails the vote by a wide margin.
  Retrieval does the classifying; the LLM is used for the explanation.
- Triage fell back to the LLM on 33 of 623 lines. There the LLM was right 22 times and the
  vote would have been right 21 times, so the fallback changes little. The thresholds were
  fixed before this run and were not tuned on it.
- The level rule looks reasonable on accuracy (0.876) but marks 71% of hard negatives as alerts.

**An earlier, wrong version.** The first knn_vote weighted all 5 hits equally by similarity.
312 of the 357 indexed incidents are normal, so an exact match (score 1.0) was often outvoted
by three normal incidents scoring about 0.75. That run scored 0.753 accuracy and lost 136
alert lines (`eval/results/20261026-143929-naive-vote.md`). Restricting the vote to hits near
the best match fixed it (`eval/results/20261026-144326.md`, retrieval-only run: 0.981).

## 3. Evidence threshold for the root-cause step

Weak evidence is shown to the root-cause step only if it scores at least 0.85 **and** has the
same label as the triage result. The threshold was set on the 357 indexed incidents, not on
the test set, by comparing each incident with its 5 nearest neighbours:

| | Median similarity | Share at or above 0.85 |
|---|---|---|
| Same-label neighbours (1,561 pairs) | 0.828 | 39.6% |
| Different-label neighbours (224 pairs) | 0.786 | 25.4% |

Similarity alone separates the two groups poorly with this embedding model, which is why
the label condition was added.

## 4. End-to-end runs

Five fixed demo lines (real lines from the test set), sent over HTTP to the running service.
Files: `eval/results/samples-stage3-http.json` (local server) and
`eval/results/samples-stage4-docker.json` (Docker container).

| Line | Result | Route | Local server | Docker |
|---|---|---|---|---|
| `KERNEL FATAL instruction address` (normal) | correct | no_action | 0.68 s warm (7.86 s cold) | 0.58 s warm |
| `data TLB error interrupt` | correct | full | 11.66 s (chat model loading) | 11.79 s (same) |
| `Lustre mount FAILED` | correct | full | 6.30 s | 5.61 s |
| `ciod ... Link has been severed` | correct | full | 7.37 s | 6.96 s |
| `Power deactivated` | correct | full | 7.36 s | 5.71 s |

On the full route, the root-cause and remediation steps take about 2–3.7 s each. The first
request after Ollama has unloaded a model waits 8–9 s for it to reload.
`OLLAMA_KEEP_ALIVE` (default 30m) keeps models loaded between requests.

**Diagnosis text, read by hand on these 5 lines:**
- Every evidence item passed to the root-cause step matched the triage label.
- The 3B model still invents specifics, for example expanding "RAS" incorrectly and naming
  configuration file paths that do not appear in the data.
- Root-cause and remediation text has **not** been scored systematically yet.

## 5. Assistant (spot checks, 2026-11-03)

These are hand checks of individual answers, not a scored evaluation. The setting was the
Lustre mount failure line (template 201) diagnosed, then questions sent to `POST /chat`,
with `llama3.2:3b` warm.

| Question | Answer checked against the database | Result |
|---|---|---|
| How often has this happened, and on how many nodes? | 289 lines, 254 nodes, 2005.08.31 to 2005.09.16 | Correct. In the first version the model also computed the share itself and gave 0.0165%; the correct value is 1.65%. The server now computes it, and the answer was then correct |
| Have we seen other filesystem or Lustre problems? | Templates 166 (8 lines) and 342 (6 lines) in the same category | First version: **wrong**. It mixed up fields between rows, giving template 166 as "normal, seen 1x". After rows were reformatted (template first, labelled fields) and same-category incidents were added from the database: correct |
| Is the suggested fix backed by the evidence? | The fix named config file paths that are not in the data | Partly wrong: it concluded "partially generated", but referred to the invented paths as evidence |
| Which categories are high severity, and how many incidents each? (no diagnosis attached) | memory/CPU 9, kernel crash 8, filesystem 3 | Correct |

Timing, measured on the same runs:
- First text arrived after 1.4–6.0 s; full answers took 4.7–14.6 s.
- Before the diagnosis and chat calls shared one context size (`num_ctx` 8192), switching
  between them made Ollama reload the model: 22.2 s before the first text.

**Related-incident search on the question.** Generic questions such as "how often has this
happened?" matched templates at 0.52–0.54 similarity, all unrelated. Topical questions
scored 0.67–0.77 on the right category ("Lustre mount failures", "ciod socket errors"). A
cut-off of 0.65 is used. This was measured on 5 questions only.

In a browser, driven over the DevTools protocol, a diagnosis followed by two questions
completed in 24.5 s, both replies streaming without errors (`docs/img/assistant.png`).

## 6. Earlier baseline (synthetic data)

The first version of the project was tested on 10 fabricated OpenStack-style lines that were
also in its index: accuracy 0.90, weighted F1 0.86, macro F1 0.71 (re-run on 2026-10-26 from the
`midterm` tag). It is listed only for comparison; those numbers say little because the test
lines were in the index.

## 7. Reproducing

```
python -I scripts/sample_bgl.py data/bgl/BGL.log      # optional: regenerate sample and test set
python -m ingestion.ingest sample_data/bgl_index.log
python -m embeddings.embed
python -m eval.evaluate eval/bgl_eval.jsonl --no-llm   # retrieval-only classifiers, about 3 min
python -m eval.evaluate eval/bgl_eval.jsonl            # all classifiers, about 25 min on CPU
python -m eval.evaluate --report eval/results/<run>.json   # rebuild a report from saved predictions
```

Long runs append each finished example to `eval/results/<run>.partial.jsonl`. After a crash,
continue with `--resume eval/results/<run>.partial.jsonl`.

Reference numbers in brackets point to `docs/REFERENCES.md`.
