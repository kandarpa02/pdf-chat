import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# OpenAI-compatible upstream (Groq by default). Existing API_KEY/BASE_URL names
# remain supported so current .env files do not break.
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("API_KEY")
)

BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
)
CHAT_MODEL = os.getenv("CHAT_MODEL")
PUBLIC_MODEL_NAME = os.getenv("PUBLIC_MODEL_NAME", "pdf-chat")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE") or None
EMBEDDING_TRUST_REMOTE_CODE = _env_bool("EMBEDDING_TRUST_REMOTE_CODE", False)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX")
EMBEDDING_DOCUMENT_PREFIX = os.getenv("EMBEDDING_DOCUMENT_PREFIX")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_HTTPS = _env_bool("QDRANT_HTTPS", False)
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "10"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = (
    float(os.getenv("SCORE_THRESHOLD"))
    if os.getenv("SCORE_THRESHOLD")
    else None
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# auto: prefer context injected by Open WebUI; otherwise use custom Qdrant when
# it contains documents; finally fall back to a normal upstream chat request.
# openwebui: never query this app's Qdrant collection for /v1/chat/completions.
# qdrant: always use this app's Qdrant collection and fail clearly if empty.
RAG_MODE = os.getenv("RAG_MODE", "auto").strip().lower()
if RAG_MODE not in {"auto", "openwebui", "qdrant"}:
    raise ValueError("RAG_MODE must be one of: auto, openwebui, qdrant")
