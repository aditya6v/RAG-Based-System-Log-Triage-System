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
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# text-embedding-004 outputs 768-dim vectors. If you ever switch embedding
# models, this MUST match the new model's output size or Qdrant will reject
# every upsert with a dimension-mismatch error.
EMBEDDING_DIM = 768


def require_all():
    required = ["MONGO_URI"]
    if LLM_PROVIDER == "gemini":
        required.append("GEMINI_API_KEY")
    for name in required:
        _require(name)