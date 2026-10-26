"""
All LLM calls go through here -- two functions, chat_text and embed_text.
Nothing else in the codebase should import google.genai or ollama
directly, so if you ever swap providers, this is the only file that
changes.

Provider is picked by config.LLM_PROVIDER ("ollama", the default, fully
local; or "gemini" -- see config.py for how to switch).

Note: we use the plain `google-genai` SDK / `ollama` client directly
rather than going through LangChain's chat-model wrapper. LangGraph
doesn't require LangChain model classes -- a node is just a Python
function -- and one less abstraction layer is one less place for
version-mismatch bugs to hide during a crunch.

Breaks-and-fixes (Gemini):
  - ImportError on `from google import genai` -> you installed the
    deprecated `google-generativeai` package instead of `google-genai`.
    They're different packages with similar names; requirements.txt
    specifies the right one.
  - API key errors -> GEMINI_API_KEY in .env, from aistudio.google.com.
  - Empty / malformed response from chat_text when you asked for JSON ->
    models occasionally wrap JSON in markdown fences or add stray text.
    See the parsing fallback in agents/graph.py rather than trusting
    raw json.loads() on the response.
  - `google.genai.errors.ClientError: 404 NOT_FOUND` mentioning a model
    name -> that model's been retired/renamed on Google's side (happened
    twice during initial setup). The error message sometimes names the
    replacement directly. This is NOT retried below -- a bad model name
    won't fix itself by waiting, so it fails immediately with a clear
    traceback rather than burning time retrying something that can't
    succeed.
  - `google.genai.errors.ServerError: 503 UNAVAILABLE` ("high demand") ->
    this IS retried automatically below, with backoff. A newly-released
    model (like whatever you get redirected to after a deprecation
    notice) can get hit hard by everyone switching over at once. If it's
    still 503ing after all retries, the error message prints how many
    attempts were made -- wait a bit longer and try again, or check
    aistudio.google.com's status page.

Breaks-and-fixes (Ollama, LLM_PROVIDER=ollama):
  - Connection refused to localhost:11434 -> the Ollama app/service isn't
    running. It installs as a background service, so usually just
    launching it once is enough; it stays up after that.
  - `model "..." not found` -> you haven't pulled it yet. `ollama pull
    llama3.2:3b` / `ollama pull nomic-embed-text` (or whatever you set
    OLLAMA_CHAT_MODEL / OLLAMA_EMBED_MODEL to). No account/login needed
    for any of this -- it's all local.
  - Slow first response -> the model is being loaded into memory on
    first use; subsequent calls are faster as long as Ollama keeps it
    loaded.
  - No retry/backoff loop here like the Gemini path -- Ollama is local,
    so there's no "high demand" 503 case to wait out. A failure here
    almost always means the service isn't running or the model isn't
    pulled, and retrying won't fix either.
"""
import time

import config

_MAX_RETRIES = 4
_BASE_DELAY_SECONDS = 5  # 5s, 10s, 20s, 40s

_gemini_client = None
_ollama_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return _ollama_client


def _call_with_retry(fn, *args, **kwargs):
    """Retries only on ServerError (Google's side overloaded/unavailable) --
    ClientError (bad model name, bad API key, bad request) fails immediately
    since no amount of waiting fixes those. Only used on the Gemini path --
    see the module docstring for why Ollama doesn't get this treatment."""
    from google.genai import errors

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except errors.ServerError as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                print(f"  Gemini overloaded (attempt {attempt + 1}/{_MAX_RETRIES}), retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(
        f"Gemini still unavailable after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def chat_text(prompt: str) -> str:
    if config.LLM_PROVIDER == "ollama":
        response = _get_ollama_client().chat(
            model=config.OLLAMA_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        return response["message"]["content"]

    response = _call_with_retry(
        _get_gemini_client().models.generate_content,
        model=config.GEMINI_CHAT_MODEL,
        contents=prompt,
    )
    return response.text


def embed_text(text: str) -> list[float]:
    if config.LLM_PROVIDER == "ollama":
        # nomic-embed-text natively outputs 768-dim vectors, matching
        # config.EMBEDDING_DIM with no resizing needed -- unlike the
        # Gemini path, Ollama's embed API has no output_dimensionality
        # knob, so if you swap OLLAMA_EMBED_MODEL for a model with a
        # different native size you'll need to update EMBEDDING_DIM and
        # recreate the Qdrant collection to match (same rule as Gemini).
        response = _get_ollama_client().embed(
            model=config.OLLAMA_EMBED_MODEL,
            input=text,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        return response["embeddings"][0]

    from google.genai import types

    # output_dimensionality is pinned explicitly to config.EMBEDDING_DIM so
    # this keeps working no matter which Gemini embedding model is
    # configured -- the newer gemini-embedding-* models natively output much
    # larger vectors (e.g. 3072-dim) unless you ask for a smaller size, and
    # our Qdrant collection was created expecting EMBEDDING_DIM. Without
    # this, switching embedding models silently breaks every upsert with a
    # dimension-mismatch error.
    response = _call_with_retry(
        _get_gemini_client().models.embed_content,
        model=config.GEMINI_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIM),
    )
    return response.embeddings[0].values