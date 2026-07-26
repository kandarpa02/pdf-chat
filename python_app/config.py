from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    QDRANT_HOST: str
    QDRANT_PORT: int

    class Config:
        env_file = ".env"


settings = Settings()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
TOP_K = int(os.getenv("TOP_K", 5))