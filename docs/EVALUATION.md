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

All classifiers see exactly the same retrieval results. Predictions from the LLM runs are
saved, so the rules derived from them (triage variants, baselines) are recomputed without
calling the model again (`python -m eval.evaluate --report <results>.json`).

| Classifier | What it does |
|---|---|
| level_rule | FATAL or FAILURE means alert (binary only) |
| majority | Always the most common category in the index lines (`normal`) |
| tfidf_lr | TF-IDF (word 1–2 grams and character 3–5 grams) and logistic regression, trained on the index lines [5] (`eval/baselines.py`) |
| knn_vote | Similarity-weighted vote of the 5 retrieved incidents' labels; only hits within 0.05 of the best match vote |
| llm_only | The LLM picks a category from the definitions alone |
| llm_rag | The LLM also sees the 5 retrieved incidents and their labels |
| vote_llm_fallback | The earlier service rule: knn_vote when the top score is at least 0.9 and the vote share at least 0.6, otherwise llm_rag |
| **triage** | The current service rule: knn_vote when confidence (best similarity × vote share) is at least 0.712, otherwise `unknown`, with llm_rag's pick shown as an unverified suggestion |

### 2a. Random split (623 held-out lines)

Results: `eval/results/20261103-baselines-random-split.json` and `.md` (the LLM predictions
are from the 2026-10-26 run, `20261026-153603`, plus the baselines).

| Classifier | Accuracy | Macro F1 | Accuracy, unseen (151) | Alert vs normal | Hard negatives correct (108) |
|---|---|---|---|---|---|
| level_rule | n/a | n/a | n/a | 0.876 | 0.287 |
| majority | 0.482 | 0.081 | 0.662 | n/a | 1.000 |
| tfidf_lr | **0.982** | **0.982** | **0.927** | 0.982 | 1.000 |
| knn_vote | 0.981 | 0.981 | 0.921 | 0.981 | 1.000 |
| llm_only | 0.425 | 0.446 | 0.212 | 0.650 | 0.000 |
| llm_rag | 0.639 | 0.755 | 0.570 | 0.727 | 0.028 |
| vote_llm_fallback | **0.982** | 0.974 | **0.927** | 0.997 | 0.981 |
| triage | 0.970 (precision 0.993 on the 97.6% answered) | 0.866 | n/a | n/a | n/a |

### 2b. Time-ordered split (index from older logs, test on newer)

`scripts/sample_bgl_time.py` cuts the log by line position, which is time order:
- **Index:** the first 60%, 2005-06-03 to 2005-07-28. 5,161 sampled lines became 225 templates.
- **Validation:** the next 10%, to 2005-09-02, 410 lines. Used only to choose the abstain threshold.
- **Test:** the last 30%, to 2006-01-04, 632 lines. Used once.

Built into its own database, collection and parser state (`incident_copilot_bgl_time`,
`bgl_time`, `sample_data/bgl_time_drain3_state.bin`).

Two facts about this split, known from a scan before running anything:
- **Filesystem mount failures never occur in the index period.** Their first line is around
  60% into the log, so no retrieval or text classifier trained on the index can name them.
- **Most test lines are new patterns:** 500 of 632 have a message pattern that never occurred
  in the index period, including 254 of the 300 normal lines.

Results: `eval/results/20261103-time-test.json` and `.md` (test, all classifiers) and
`20261103-time-val.json` (validation, retrieval and baselines only).

| Classifier | Accuracy | Macro F1 | Accuracy, unseen (500) | Alert vs normal | Hard negatives correct (150) |
|---|---|---|---|---|---|
| level_rule | n/a | n/a | n/a | 0.840 | 0.327 |
| majority | 0.475 | 0.080 | 0.508 | n/a | 1.000 |
| tfidf_lr | 0.884 | 0.830 | 0.858 | 0.884 | 1.000 |
| knn_vote | **0.888** | 0.828 | **0.858** | **0.896** | 0.993 |
| llm_only | 0.432 | 0.465 | 0.424 | 0.690 | 0.000 |
| llm_rag | 0.492 | 0.687 | 0.470 | 0.571 | 0.027 |
| vote_llm_fallback | 0.699 | 0.819 | 0.620 | 0.715 | 0.413 |
| triage | 0.840 (precision **0.971** on the 86.6% answered) | 0.744 | n/a | n/a | n/a |

All three retrieval and text classifiers labelled the 50 filesystem lines `normal`.

### 2c. Abstaining (`python -m eval.abstain <val> <test>`)

Confidence = best similarity × vote share. The target precision of 0.95 was fixed before
looking at test. The lowest threshold reaching it **on validation** is 0.712:

| Set | Precision on answered lines | Coverage | Precision if always answering |
|---|---|---|---|
| Validation | 0.953 | 0.944 (387/410) | 0.912 |
| **Test** | **0.971** | **0.866** (547/632) | 0.888 |
| Random split (same threshold) | 0.993 | 0.976 (608/623) | 0.981 |

On the time-split test, per category:

| Category | Lines | Answered right | Answered wrong | Abstained |
|---|---|---|---|---|
| filesystem_mount_failure (never indexed) | 50 | 0 | 12 | 38 |
| normal | 300 | 269 | 1 | 30 |
| io_connection_lost | 50 | 45 | 0 | 5 |
| power_control_failure | 50 | 38 | 2 | 10 |
| the other four | 182 | 179 | 1 | 2 |

The full precision-vs-coverage table and curve are in
`eval/results/20261103-time-test-abstain.md` / `.svg`. When the system abstains, llm_rag's
pick, shown to the user as an unverified suggestion, was right on 60 of 85 time-split lines
(including all 38 abstained filesystem lines) and on 9 of 15 random-split lines.

### How to read section 2

- **A simple text classifier ties retrieval on BGL.** TF-IDF with logistic regression
  matches knn_vote on both splits (0.982 vs 0.981, then 0.884 vs 0.888) and trains in
  10 seconds. BGL's alert text varies so little that this is expected. Retrieval earns its
  place elsewhere: it supplies the evidence the explanation is grounded in, and it can
  absorb new incidents without retraining (not measured yet).
- **The random split flatters every method.** On older-index-newer-test, accuracy drops by
  about 0.1, and one category is impossible to name.
- **The earlier LLM fallback hurts under drift.** On the time split, the vote was unsure for
  328 of 632 lines and the 3B model then labelled most new normal lines as alerts (normal
  recall 0.40). It did name all 50 never-seen filesystem lines correctly, which no other
  method could. Hence the change: **abstain, and show the LLM's pick as a suggestion.** The
  service now does this by default (`TRIAGE_LOW_CONFIDENCE=abstain`); `=llm` restores the
  old rule.
- **On the random split, seen lines are close to a lookup:** 472 of 623 score 1.000 for
  knn_vote. The unseen subsets and hard negatives are the meaningful numbers there.
- The level rule looks fine on accuracy but marks about 70% of the alarming-looking normal
  lines as alerts on both splits.

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

## 5b. Does the generated text invent things? (2026-11-03)

There is no ground truth for root causes, so this measures one thing that *can* be checked:
whether the root cause and fix name **specifics that appear nowhere in what the model was
given** (the log line, the category and its definition, the evidence templates).

**Method.** `python -m eval.groundedness build` picks 42 held-out lines (6 per alert category,
seed 7) from the random-split test set, runs them through the real pipeline (`llama3.2:3b`,
local), and saves text and evidence. `python -m eval.groundedness check` splits the root
cause and fix into sentences and labels each one:
- **unsupported:** names a specific not found in the context. Specifics are file paths, file
  names, numbers of 2+ digits, IPs, hex values, code-like identifiers, and host or device names.
- **grounded:** names only specifics that are in the context.
- **generic:** names no specific, e.g. "check the network connection". Not wrong, but not
  evidence-based either.

**Hand check of the checker.** I read 10 sampled sentences (4 unsupported, 3 grounded,
3 generic) against their log and evidence and agreed with 9 verdicts:
- All 4 "unsupported" were real inventions: "PDU" for a socket error, `/etc/config`,
  `/etc/cpuinfo`, "wait 15–30 minutes".
- The miss: a "Power deactivated" line was told to check the "mailbox/idoproxy socket".
  That's blended in from the category definition, which the checker counts as context.

So the checker **undercounts**. It can't judge whether a general claim is true. An LLM judge
with a larger model is planned once a hosted provider key is available.

| Prompt | Sentences | Grounded | Generic | **Unsupported** | Diagnoses with ≥1 unsupported |
|---|---|---|---|---|---|
| Current (`eval/rubric/v1-baseline-prompt.jsonl`) | 148 | 24.3% | 70.3% | **5.4%** | 8/42 (19.0%) |
| Stricter rules, tried and reverted (`eval/rubric/v2-grounding-rule.jsonl`) | 178 | 29.8% | 60.7% | 9.6% | 9/42 (21.4%) |

The stricter prompt added two things: the log line marked as quoted data, and the rule
"only name files, paths, hosts, devices, services, commands or numbers that appear in the
log line or the evidence; otherwise describe what to check in general terms". It **did not
help**:
- It produced longer answers (90 vs 80 words on average) with more invented specifics.
  New ones included "NIC", "PSU" and more Lustre config paths such as
  `/etc/lustre/conf/lustre.conf`.
- One run each at temperature 0.2 on 42 lines, so part of the difference may be noise.
- There is no evidence it helps, so the original prompts are kept.

Typical inventions under both prompts: configuration file paths (`/etc/lustre/fgl.conf`,
`/p/gb1/lnfs/lnfsd.conf`, `/etc/config/ras.conf`) and hardware names not in the log.

## 6. LLM providers and embeddings (2026-11-03)

**Tokens per call**, as reported by Ollama (`llama3.2:3b`), through the running API:

| Request | LLM calls | Prompt tokens | Reply tokens |
|---|---|---|---|
| Normal line (`no_action` route) | 0 | 0 | 0 |
| Full diagnosis, 4 demo lines | 2 | 476–699 | 103–121 |
| Assistant turn, 3 turns on one diagnosis | 1 | 1,408–1,547 | 52–71 |

Hosted providers count tokens with their own tokenizers, so their numbers will differ
somewhat. Their free-tier limits will be read from each provider's console when the keys
are added, and the provider comparison is pending until then.

**Fallback behaviour** is covered by tests with faked HTTP responses (`tests/test_llm.py`):
- 429 with `Retry-After`, then success
- `Retry-After` capped at 10 s
- 5xx three times, then the next provider
- timeouts
- 401 not retried
- everything failing down to Ollama
- streaming fallback before the first chunk, but not after it

**In-process embeddings** (`scripts/compare_embeddings.py nomic-ai/nomic-embed-text-v1.5 100`):

| Check | Result |
|---|---|
| Cosine between fastembed and Ollama vectors, 357 templates | mean 1.0000, minimum 1.0000 |
| Same top-1 incident in the existing index, 100 test lines | 100/100 |
| Same top-5 set | 100/100 |
| Retrieval latency with fastembed, model loaded | 0.24 s (first query 3.4 s, model load) |

So Ollama's `nomic-embed-text` is nomic v1.5, and the in-process copy can be used against
the existing index. The one-time model download was 523 MB and took about 5 minutes.

## 7. Earlier baseline (synthetic data)

The first version of the project was tested on 10 fabricated OpenStack-style lines that were
also in its index: accuracy 0.90, weighted F1 0.86, macro F1 0.71 (re-run on 2026-10-26 from the
`midterm` tag). It is listed only for comparison; those numbers say little because the test
lines were in the index.

## 8. Reproducing

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
