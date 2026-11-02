"""
Central place every other module pulls settings from. Nothing here does
real work -- if something downstream is misbehaving because of a missing
key, you want one obvious place to check, not env var lookups scattered
across six files.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val or val.startswith("<"):
        raise RuntimeError(
            f"{name} is not set (or still has the placeholder value). "
            f"Copy .env.example to .env and fill it in."
        )
    return val


# Fail fast and loud at import time instead of three function calls deep
# during a demo. If you're just exploring code without a .env yet, import
# individual names lazily instead of `import config` at module load time.
MONGO_URI = os.environ.get("MONGO_URI", "")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "incident_copilot")

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "incidents")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# Chat providers, tried in order. Hosted ones without an API key are skipped,
# so with no keys this is Ollama only. If unset, falls back to LLM_PROVIDER
# (the older single-provider setting) for existing .env files.
LLM_PROVIDER_CHAIN = os.environ.get("LLM_PROVIDER_CHAIN") or os.environ.get("LLM_PROVIDER") or "groq,openrouter,ollama"
# Embeddings never fall back between models: index and queries must use the
# same one. Default: Ollama (or Gemini, if that was the single provider).
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "gemini" if LLM_PROVIDER == "gemini" else "ollama").lower()
# EMBED_PROVIDER=fastembed runs the same nomic model in-process (no Ollama).
# Verified to give the same vectors as Ollama's nomic-embed-text; see
# scripts/compare_embeddings.py before changing the model name.
FASTEMBED_MODEL = os.environ.get("FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
FASTEMBED_CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR", "data/fastembed_cache")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Model names on free tiers change; check the provider's model list.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "30"))
# Mask IPs, node and host names, user names and tokens before text goes to a
# hosted provider (agents/redact.py). Ollama is local and never redacted.
REDACT_HOSTED = os.environ.get("REDACT_HOSTED", "1") != "0"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# How long Ollama keeps a model in memory after the last call. Its default
# is 5 minutes, after which the next request waits 8+ s for a reload.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# text-embedding-004 outputs 768-dim vectors. If you ever switch embedding
# models, this MUST match the new model's output size or Qdrant will reject
# every upsert with a dimension-mismatch error.
EMBEDDING_DIM = 768


def require_all():
    required = ["MONGO_URI"]
    if EMBED_PROVIDER == "gemini":
        required.append("GEMINI_API_KEY")
    for name in required:
        _require(name)