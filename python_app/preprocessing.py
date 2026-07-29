from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DOCUMENT_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDING_TRUST_REMOTE_CODE,
)
from .data_parser import ParsedPage


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Create non-empty character chunks while preserving paragraph boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    clean_text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean_text:
        return []

    paragraphs = [part.strip() for part in clean_text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    def append_sliding_window(value: str) -> None:
        step = chunk_size - overlap
        start = 0
        while start < len(value):
            piece = value[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_size >= len(value):
                break
            start += step

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            append_sliding_window(paragraph)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
            if len(current) > chunk_size:
                append_sliding_window(current)
                current = ""
        else:
            current = paragraph

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_pages(pages: Iterable[ParsedPage]) -> list[dict]:
    """Chunk each page and retain page numbers for citations/debugging."""
    result: list[dict] = []
    for page in pages:
        for chunk_index, text in enumerate(chunk_text(page.text)):
            result.append(
                {
                    "text": text,
                    "page": page.page_number,
                    "page_chunk": chunk_index,
                }
            )
    return result


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    kwargs = {"trust_remote_code": EMBEDDING_TRUST_REMOTE_CODE}
    if EMBEDDING_DEVICE:
        kwargs["device"] = EMBEDDING_DEVICE
    return SentenceTransformer(EMBEDDING_MODEL, **kwargs)


def _prefix_for(kind: str) -> str:
    configured = (
        EMBEDDING_QUERY_PREFIX if kind == "query" else EMBEDDING_DOCUMENT_PREFIX
    )
    if configured is not None:
        return configured

    # Nomic explicitly requires task prefixes for retrieval.
    if "nomic-embed-text" in EMBEDDING_MODEL.lower():
        return "search_query: " if kind == "query" else "search_document: "

    return ""


def _encode(texts: list[str], kind: str) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefix = _prefix_for(kind)
    prepared = [f"{prefix}{text}" for text in texts]
    return get_embedding_model().encode(
        prepared,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_documents(texts: list[str]) -> np.ndarray:
    return _encode(texts, "document")


def embed_query(text: str) -> np.ndarray:
    return _encode([text], "query")[0]


def embedding_text(chunks: list[str]) -> np.ndarray:
    """Backward-compatible alias. Prefer embed_documents/embed_query."""
    return embed_documents(chunks)
