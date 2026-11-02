# Design notes

Why the system is built the way it is, what was tried and dropped, and its known limits.
Results referred to here are in `docs/EVALUATION.md`. Numbers in brackets refer to
`docs/REFERENCES.md`.

## Pipeline

```
raw line -> retrieval -> triage -> router -> root_cause -> remediation
                                        \-> no_action      (severity none/low)
```

- **Retrieval.**
  - Removes the dataset's label token.
  - Matches the line against the parsed templates (Drain [3, 4], read-only at query time).
  - Embeds the template and takes the 5 nearest stored incidents from Qdrant.
  - Reads their counts and labels from MongoDB.
- **Triage.** A nearest-neighbour vote [10] over the retrieved incidents. The LLM is called
  only when the vote is not confident.
- **Router.** A plain function: the category's severity in `bgl/categories.json` decides the
  path. No model is involved, so the route can be unit-tested.
- **Root cause.** An LLM prompt grounded in retrieved evidence, in the style of
  retrieval-augmented generation [6]. The best match is marked as primary evidence. Other
  matches are included only if they score at least 0.85 and share the triage label.
- **Remediation.** A second LLM prompt that receives the root cause and the category definition.

Retrieval runs before triage because the vote needs the retrieved labels.

## Decisions

| Decision | Reason |
|---|---|
| Per-line classification on BGL | BGL labels every line [1, 2], which matches a "paste one line, get a diagnosis" workflow |
| Retrieval classifies, the LLM explains | Measured: the vote scores 0.981, the 3B LLM alone 0.425 |
| Abstain ("unknown") when the vote is weak, instead of asking the LLM | On a time-ordered split, the LLM fallback dropped accuracy from 0.888 to 0.699 by labelling unfamiliar normal lines as alerts. Abstaining below a threshold chosen on validation gave 0.971 precision on 86.6% of lines. The LLM's pick is still shown, labelled unverified: it was right on 60 of 85 abstained lines, including every never-seen filesystem failure |
| Keep retrieval although TF-IDF + logistic regression ties it | Same accuracy, but retrieval gives the explanation step its evidence and needs no retraining to add incidents |
| Deterministic routing | A route chosen by a model can't be tested; a severity table can |
| Drain3 read-only at query time | `add_log_message` changes the parser state on every query, so templates drift away from what was embedded. Queries use `match()`; a test checks the saved state stays unchanged |
| One embedding model everywhere | Vectors from different models are not comparable, even at the same dimension |
| Ollama as the default LLM | Local, no rate limits; a live demo can't depend on a free hosted tier |
| Qdrant for vectors | Runs locally and has a free cloud tier. No detailed comparison with other vector stores was done |
| Categories in one JSON file | Prompts, evaluation, routing and the UI read the same list |

## Tried and rejected

**OpenStack logs (Loghub).**
- `openstack_abnormal.log` has 18,434 lines and no ERROR lines.
- Its 4 anomalous instances are unusual *sequences* of normal-looking INFO events (for
  example a VM started, paused and resumed during spawn).
- That is a sequence-level problem, the kind addressed by DeepLog [7], not per-line triage.
  It is kept as future work.

**HDFS logs (Loghub)** were considered as a second dataset. Their labels are per storage
block [8], a group of lines, not per line, so they don't fit this system for the same
reason as OpenStack.

**Gemini as the main LLM.** The free tier's rate limits and occasional 503 "overloaded"
errors made it unreliable for live demos. It remains an optional provider.

**Switching embedding models without re-indexing.** After an embedding model change, an
exact match scored about 0.04 instead of 1.0, until the collection was cleared and
re-embedded. Since then, one model is used for both the index and the queries.

**Equal-weight neighbour vote.** The first vote weighted all 5 hits by similarity. Because
312 of the 357 indexed incidents are normal, an exact match was regularly outvoted by three
normal incidents scoring around 0.75: accuracy 0.753, with 136 alert lines lost. Now only
hits within 0.05 of the best match vote, giving 0.981 on the same test set.

**Drain `sim_th` = 0.4.** The commonly used value merged short, unrelated messages on the
words "KERNEL FATAL" alone. Measured cluster purity led to 0.7 (`docs/EVALUATION.md`, section 1).

**Similarity alone as the evidence filter.** Among indexed incidents, 25% of
different-label neighbour pairs still score at least 0.85, so a label match is also required.

**Contradictory labels.** The message "program interrupt" appears 3,662 times labelled normal
and 5 times labelled KERNPROG. No classifier can separate these, so KERNPROG is mapped to normal.

## Limitations

1. **Only classification is measured.** The dataset has no ground truth for root cause or
   remediation. The text was read by hand on 5 cases only.
2. **The 3B model invents specifics.** Examples: a wrong expansion of "RAS", and file paths
   that don't appear in the data. The explanations are plausible in general, not reliable
   in detail.
3. **Seen-pattern accuracy is high because of the data.** BGL has only 88 distinct alert
   messages. The unseen subset and the hard negatives are the fair test, and the unseen alert
   subsets are small.
4. **Accuracy drops on later logs.** On the time-ordered split, retrieval falls from 0.981
   to 0.888. A category absent from the older logs (filesystem failures) can't be named:
   at best it is answered "unknown" (38 of 50).
5. **A plain text classifier is as accurate.** TF-IDF with logistic regression [5] ties the
   vote on both splits. The retrieval design is justified by grounding and updatability,
   not by classification accuracy.
6. **One dataset.** Categories and the parser settings are tuned for BGL.
7. **Per line only.** Sequence anomalies are out of scope.
8. **CPU latency.** About 6–7 s for a full diagnosis when warm; 8–9 s extra when a model has
   to be reloaded.
9. **No offline mode.** MongoDB Atlas and Qdrant Cloud must be reachable.
10. **The assistant is only spot-checked.** Database facts were answered correctly in the
    checks, but it has treated generated text as evidence (`docs/EVALUATION.md`, section 5).

## LLM providers

| Decision | Reason |
|---|---|
| A chain: Groq, then OpenRouter, then local Ollama | Hosted models are larger and faster, but free tiers have small rate limits and go down. Ollama is the last resort that needs no key or network, so a demo can't fail on a quota |
| Retry 429/5xx/timeouts twice, then move on; don't retry 401/404/400 | Waiting can fix overload, but not a bad key or a retired model name |
| Honour `Retry-After`, capped at 10 s | Respects the provider's rate limit without stalling a request for minutes |
| Plain `httpx` against the OpenAI-compatible endpoints | Groq and OpenRouter share one request format, and no extra SDK is needed |
| Redact before hosted calls, not for Ollama | Log text names nodes, IPs and users. Local inference never leaves the machine |
| No client-side rate limiter yet | Measured: a full diagnosis is at most about 810 tokens and an assistant turn about 1,600. Whether that fits a free tier depends on its current limits, which will be read from the console. 429s are already retried and fall back |
| Embeddings never fall back between models | Vectors from different models are not comparable; a silent fallback would corrupt retrieval |
| fastembed as an in-process option, only after a check | Checked against Ollama: identical vectors (cosine 1.0000) and identical retrieval on 100 lines |
| Streaming calls record usage into a list passed in, not a context variable | Starlette runs each step of a streaming response in a fresh copy of the request context, so a context variable set in the first step was lost. Found in a live test when the provider was missing from the `done` event |

## Assistant

A follow-up conversation about a diagnosis, on the same local model (`agents/chat.py`,
`POST /chat`).

| Decision | Reason |
|---|---|
| The server does the database lookups; the model only reads | A 3B model is unreliable at deciding when to call tools and with what arguments. Fixed lookups each turn (category counts, facts about the matched template, same-category incidents, a vector search on the question) are predictable and testable |
| Numbers are computed in code | The model turned 289 of 17,552 into "0.0165%" (it is 1.65%). The context now states the percentage |
| One incident per line, template first, every field labelled | With a compact format the model swapped fields between rows, reporting a filesystem incident as "normal, seen 1x" |
| Question-based search only keeps hits at 0.65 or more | Prose questions match log templates poorly. Generic questions scored about 0.53 against unrelated incidents |
| Same `num_ctx` (8192) for diagnosis and chat | Ollama reloads the model when `num_ctx` changes. Switching between the two cost about 22 s |
| Streaming, as newline-delimited JSON | First text arrives in about 1.5–6 s instead of waiting 5–15 s for the whole answer |
| A new diagnosis starts a new conversation | Mixing two incidents in one context makes the small model confuse them |
| Log text sits in the context block, with an instruction not to obey it | Basic prompt-injection hygiene. A test checks that log text never becomes a user message. This doesn't prove the model can't be talked into something |

## Questions likely to come up

**If a nearest-neighbour vote gets 98%, why use an LLM?**
For classification, it mostly isn't used: it runs on 33 of 623 test lines. The LLM writes the
explanation and the suggested fix, which can't be looked up.

**Isn't 100% on seen patterns memorisation?**
Yes. That is why patterns are held out entirely (unseen subset: 0.927) and why 108 hard
negatives are reported separately.

**Why not flag FATAL lines?**
That rule gets 0.876 on alert-vs-normal but marks 71% of the normal FATAL/ERROR lines as alerts.

**How do you know the label didn't leak into the input?**
`bgl/lines.py` strips it before anything else sees the line. A test checks this, and none of
the 623 test lines start with their label.

**How do you know test lines weren't in the index?**
The sampler asserts that no line appears in both and that held-out patterns are absent from
the index.

**Why several agents instead of one prompt?**
- Each step uses a different method (vector search, a vote, two prompts) and is timed separately.
- The router lets normal lines skip both LLM calls: 0.6 s instead of 6–7 s.
- The fix is written after, and based on, the root cause.

**A TF-IDF model gets the same accuracy. Why build retrieval at all?**
On BGL, it doesn't win on accuracy, and the evaluation says so (0.982 vs 0.981 random,
0.884 vs 0.888 time-ordered). Retrieval gives two things a classifier doesn't:
- the past incidents that the explanation is grounded in, shown to the user
- new knowledge without retraining: adding an incident is an insert

The second still has to be measured (the feedback loop).

**What happens with a failure type the system has never seen?**
On the time split, filesystem failures never occur in the index period. With
abstaining, 38 of 50 were answered "unknown" and 12 were wrongly called normal. The
LLM's unverified suggestion named all 38 abstained ones correctly.

**What would you do next?**
- A second per-line-labelled dataset (Thunderbird [1]). A 10-million-line prefix turned out
  to be 99.96% one alert type, so the whole log has to be streamed.
- A larger judge model for the groundedness check, and a scored assistant evaluation.
- The feedback loop: measure whether adding corrected incidents fixes later lines.
- Sequence-level detection for cases like OpenStack.
