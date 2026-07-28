import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
TOP_K = int(os.getenv("TOP_K", "5"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = (
    int(os.getenv("POSTGRES_PORT"))
    if os.getenv("POSTGRES_PORT")
    else None
)
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")