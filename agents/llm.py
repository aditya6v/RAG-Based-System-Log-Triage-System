"""
All LLM calls go through here: chat_text, chat_stream and embed_text.
Nothing else in the codebase talks to an LLM provider.

Chat: a provider chain
----------------------
config.LLM_PROVIDER_CHAIN lists providers in order, e.g. "groq,openrouter,ollama".
Hosted providers without an API key are skipped. For each call:

  * 429, 5xx, timeouts and dropped connections are retried on the same
    provider (twice, short backoff, honouring Retry-After up to 10 s), then
    the next provider is tried.
  * Other errors (401 bad key, 404 unknown model, 400) are not retried: the
    next provider is tried straight away.
  * If every provider fails, RuntimeError lists what each one said.

Groq and OpenRouter use their OpenAI-compatible HTTP APIs through httpx (no
extra SDK). Ollama uses its Python client; Gemini the google-genai SDK.

Before text goes to a hosted provider it is redacted (agents/redact.py):
IPs, node and host names, user names in paths, tokens. Ollama runs locally
and gets the text unchanged. REDACT_HOSTED=0 disables redaction.

Every answered call is recorded: provider, model, latency, attempts and
token counts as reported by the provider. Call start_recording() at the start
of a request to collect them (the API returns them as `llm_calls`).

Embeddings: one model, no fallback
----------------------------------
embed_text uses config.EMBED_PROVIDER only. Vectors from different models
don't share a space, so falling back to another embedding model would
silently corrupt retrieval. Switching needs a full re-embed of the index.

Breaks-and-fixes:
  - "no LLM provider answered": read the per-provider reasons in the message.
    Typical: Ollama not running (connection refused on 11434), a hosted key
    missing or wrong (401), a model name retired by the provider (404).
  - Slow first response from Ollama: the model is being loaded. Later calls
    are faster while OLLAMA_KEEP_ALIVE keeps it in memory.
  - Gemini ImportError on `from google import genai`: install `google-genai`,
    not the older `google-generativeai`.
"""
import contextvars
import json
import logging
import time

import httpx

import config
from agents.redact import redact_messages

log = logging.getLogger("llm")

# Ollama context window. The assistant's prompt (category definitions,
# diagnosis, related incidents, conversation) overflows Ollama's old default
# and would be truncated silently. Measured prompts: diagnosis calls under
# 700 tokens, assistant turns about 1,550 plus history, so 4096 leaves room.
# At 8192, Ollama reported llama3.2:3b at 3.25 GB loaded and the 16 GB laptop
# ran out of free memory. The same value is used for every call because
# Ollama reloads the model whenever num_ctx changes (~20 s).
CHAT_NUM_CTX = 4096

RETRIES_PER_PROVIDER = 2      # attempts per provider = 1 + this
BACKOFF_S = (1.0, 3.0)
MAX_RETRY_AFTER_S = 10.0
HOSTED = {"groq", "openrouter", "gemini"}
OPENAI_COMPATIBLE = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


class ProviderError(Exception):
    def __init__(self, provider: str, message: str, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.provider, self.retryable, self.retry_after = provider, retryable, retry_after


# --- recording ------------------------------------------------------------------

_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_calls", default=None)


def start_recording() -> list[dict]:
    """Collect a record of every answered LLM call made from this context."""
    calls: list[dict] = []
    _calls.set(calls)
    return calls


def _record(entry: dict, into: list | None = None) -> None:
    calls = into if into is not None else _calls.get()
    if calls is not None:
        calls.append(entry)
    log.info("llm_call %s", json.dumps(entry))


# --- providers --------------------------------------------------------------------

def provider_chain() -> list[str]:
    names = [n.strip().lower() for n in config.LLM_PROVIDER_CHAIN.split(",") if n.strip()]
    keys = {"groq": config.GROQ_API_KEY, "openrouter": config.OPENROUTER_API_KEY, "gemini": config.GEMINI_API_KEY}
    usable = []
    for n in names:
        if n in keys and not keys[n]:
            continue  # hosted provider without a key
        if n not in keys and n != "ollama":
            log.warning("unknown LLM provider %r in LLM_PROVIDER_CHAIN, skipped", n)
            continue
        usable.append(n)
    return usable


def _model(name: str) -> str:
    return {"groq": config.GROQ_MODEL, "openrouter": config.OPENROUTER_MODEL,
            "ollama": config.OLLAMA_CHAT_MODEL, "gemini": config.GEMINI_CHAT_MODEL}[name]


_http: httpx.Client | None = None


def _http_client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=config.LLM_TIMEOUT_S)
    return _http


def _openai_request(name: str, messages: list[dict], temperature: float, stream: bool) -> dict:
    key = config.GROQ_API_KEY if name == "groq" else config.OPENROUTER_API_KEY
    return {
        "url": OPENAI_COMPATIBLE[name],
        "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "json": {"model": _model(name), "messages": messages, "temperature": temperature, "stream": stream},
    }


def _http_error(name: str, r: httpx.Response) -> ProviderError:
    retry_after = None
    try:
        retry_after = float(r.headers.get("retry-after", ""))
    except ValueError:
        pass
    retryable = r.status_code == 429 or r.status_code >= 500
    return ProviderError(name, f"HTTP {r.status_code}: {r.text[:200]}", retryable, retry_after)


def _openai_complete(name: str, messages: list[dict], temperature: float) -> dict:
    req = _openai_request(name, messages, temperature, stream=False)
    try:
        r = _http_client().post(req["url"], headers=req["headers"], json=req["json"])
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ProviderError(name, f"{type(e).__name__}: {e}", retryable=True) from e
    if r.status_code != 200:
        raise _http_error(name, r)
    data = r.json()
    usage = data.get("usage") or {}
    return {"text": data["choices"][0]["message"]["content"] or "",
            "model": data.get("model", _model(name)),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def _openai_stream(name: str, messages: list[dict], temperature: float):
    """Yields ("text", chunk) pieces, then ("usage", dict) once."""
    req = _openai_request(name, messages, temperature, stream=True)
    usage, model = {}, _model(name)
    try:
        with _http_client().stream("POST", req["url"], headers=req["headers"], json=req["json"]) as r:
            if r.status_code != 200:
                r.read()
                raise _http_error(name, r)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                model = chunk.get("model", model)
                usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield "text", text
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise ProviderError(name, f"{type(e).__name__}: {e}", retryable=True) from e
    yield "usage", {"model": model, "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens")}


_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return _ollama_client


def _ollama_error(e: Exception) -> ProviderError:
    status = getattr(e, "status_code", None)
    # Connection refused means Ollama isn't running: retrying won't help.
    return ProviderError("ollama", f"{type(e).__name__}: {e}", retryable=bool(status and status >= 500))


def _ollama_options(temperature: float) -> dict:
    return {"temperature": temperature, "num_ctx": CHAT_NUM_CTX}


def _ollama_complete(messages: list[dict], temperature: float) -> dict:
    try:
        r = _get_ollama_client().chat(model=config.OLLAMA_CHAT_MODEL, messages=messages,
                                      options=_ollama_options(temperature), keep_alive=config.OLLAMA_KEEP_ALIVE)
    except Exception as e:
        raise _ollama_error(e) from e
    return {"text": r["message"]["content"], "model": config.OLLAMA_CHAT_MODEL,
            "prompt_tokens": r.get("prompt_eval_count"), "completion_tokens": r.get("eval_count")}


def _ollama_stream(messages: list[dict], temperature: float):
    usage = {}
    try:
        for part in _get_ollama_client().chat(model=config.OLLAMA_CHAT_MODEL, messages=messages, stream=True,
                                              options=_ollama_options(temperature),
                                              keep_alive=config.OLLAMA_KEEP_ALIVE):
            if part["message"]["content"]:
                yield "text", part["message"]["content"]
            if part.get("done"):
                usage = {"prompt_tokens": part.get("prompt_eval_count"), "completion_tokens": part.get("eval_count")}
    except Exception as e:
        raise _ollama_error(e) from e
    yield "usage", {"model": config.OLLAMA_CHAT_MODEL, **usage}


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _gemini_complete(messages: list[dict], temperature: float) -> dict:
    from google.genai import errors, types
    if len(messages) == 1:
        prompt = messages[0]["content"]
    else:  # no chat roles in this call: flatten the conversation
        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:"
    try:
        r = _get_gemini_client().models.generate_content(
            model=config.GEMINI_CHAT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature))
    except errors.ServerError as e:
        raise ProviderError("gemini", str(e), retryable=True) from e
    except errors.ClientError as e:
        raise ProviderError("gemini", str(e), retryable=getattr(e, "code", None) == 429) from e
    usage = getattr(r, "usage_metadata", None)
    return {"text": r.text or "", "model": config.GEMINI_CHAT_MODEL,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None)}


def _gemini_stream(messages: list[dict], temperature: float):
    out = _gemini_complete(messages, temperature)
    yield "text", out["text"]
    yield "usage", {k: out[k] for k in ("model", "prompt_tokens", "completion_tokens")}


def _complete_with(name: str, messages: list[dict], temperature: float) -> dict:
    if name in OPENAI_COMPATIBLE:
        return _openai_complete(name, messages, temperature)
    return {"ollama": _ollama_complete, "gemini": _gemini_complete}[name](messages, temperature)


def _stream_with(name: str, messages: list[dict], temperature: float):
    if name in OPENAI_COMPATIBLE:
        return _openai_stream(name, messages, temperature)
    return {"ollama": _ollama_stream, "gemini": _gemini_stream}[name](messages, temperature)


# --- the chain ----------------------------------------------------------------------

def _prepare(name: str, messages: list[dict]) -> list[dict]:
    return redact_messages(messages) if name in HOSTED and config.REDACT_HOSTED else messages


def _wait(e: ProviderError, attempt: int) -> None:
    delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
    if e.retry_after is not None:
        delay = min(e.retry_after, MAX_RETRY_AFTER_S)
    time.sleep(delay)


def _chain_or_raise() -> list[str]:
    chain = provider_chain()
    if not chain:
        raise RuntimeError("no LLM provider configured: set LLM_PROVIDER_CHAIN and the API keys it needs")
    return chain


def _complete(messages: list[dict], temperature: float) -> str:
    failures: list[str] = []
    for name in _chain_or_raise():
        send = _prepare(name, messages)
        for attempt in range(RETRIES_PER_PROVIDER + 1):
            t0 = time.perf_counter()
            try:
                out = _complete_with(name, send, temperature)
            except ProviderError as e:
                failures.append(f"{name} (attempt {attempt + 1}): {e}")
                log.warning("llm_failure provider=%s attempt=%d retryable=%s %s", name, attempt + 1, e.retryable, e)
                if e.retryable and attempt < RETRIES_PER_PROVIDER:
                    _wait(e, attempt)
                    continue
                break
            _record({"provider": name, "model": out["model"], "ms": round((time.perf_counter() - t0) * 1000),
                     "attempts": attempt + 1, "failed_before": len(failures),
                     "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                     "redacted": send is not messages})
            return out["text"]
    raise RuntimeError("no LLM provider answered: " + " | ".join(failures))


def chat_text(prompt: str) -> str:
    """One prompt in, the reply text out, through the provider chain."""
    return _complete([{"role": "user", "content": prompt}], temperature=0.2)


def chat_stream(messages: list[dict], record_to: list | None = None):
    """Multi-turn chat; yields the reply in text chunks as they arrive.
    `messages` is a list of {"role": "system"|"user"|"assistant", "content"}.
    The chain can only move to the next provider before the first chunk has
    been sent; a failure after that raises.

    Pass `record_to` to collect the call record. start_recording() doesn't
    reach here when the generator is driven by a streaming HTTP response:
    Starlette runs each step in a fresh copy of the request's context."""
    failures: list[str] = []
    for name in _chain_or_raise():
        send = _prepare(name, messages)
        for attempt in range(RETRIES_PER_PROVIDER + 1):
            t0, started = time.perf_counter(), False
            try:
                for kind, value in _stream_with(name, send, temperature=0.3):
                    if kind == "text":
                        started = True
                        yield value
                    else:
                        _record({"provider": name, "model": value.get("model"),
                                 "ms": round((time.perf_counter() - t0) * 1000),
                                 "attempts": attempt + 1, "failed_before": len(failures),
                                 "prompt_tokens": value.get("prompt_tokens"),
                                 "completion_tokens": value.get("completion_tokens"),
                                 "redacted": send is not messages}, record_to)
                return
            except ProviderError as e:
                if started:
                    raise RuntimeError(f"{name} failed part-way through the answer: {e}") from e
                failures.append(f"{name} (attempt {attempt + 1}): {e}")
                log.warning("llm_failure provider=%s attempt=%d retryable=%s %s", name, attempt + 1, e.retryable, e)
                if e.retryable and attempt < RETRIES_PER_PROVIDER:
                    _wait(e, attempt)
                    continue
                break
    raise RuntimeError("no LLM provider answered: " + " | ".join(failures))


# --- embeddings ---------------------------------------------------------------------

_fastembed = None


def _fastembed_model():
    """In-process ONNX copy of nomic-embed-text v1.5, for running without
    Ollama. Checked against Ollama's vectors before use: cosine 1.0000 on
    all 357 BGL templates and identical top-5 retrieval for 100 test lines
    (scripts/compare_embeddings.py). Needs `pip install fastembed`."""
    global _fastembed
    if _fastembed is None:
        from fastembed import TextEmbedding
        _fastembed = TextEmbedding(model_name=config.FASTEMBED_MODEL, cache_dir=config.FASTEMBED_CACHE_DIR)
    return _fastembed


def embed_text(text: str) -> list[float]:
    if config.EMBED_PROVIDER == "fastembed":
        return [float(x) for x in next(iter(_fastembed_model().embed([text])))]

    if config.EMBED_PROVIDER == "ollama":
        # nomic-embed-text natively outputs 768-dim vectors, matching
        # config.EMBEDDING_DIM. Swapping OLLAMA_EMBED_MODEL for a model with a
        # different size needs EMBEDDING_DIM updated and the collection rebuilt.
        response = _get_ollama_client().embed(
            model=config.OLLAMA_EMBED_MODEL,
            input=text,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        return response["embeddings"][0]

    from google.genai import types

    # output_dimensionality is pinned to config.EMBEDDING_DIM: newer Gemini
    # embedding models output 3072-dim vectors unless asked otherwise, and the
    # Qdrant collection expects EMBEDDING_DIM.
    response = _get_gemini_client().models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIM),
    )
    return response.embeddings[0].values
