import os
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

def chunk_text(text, chunk_size=1000, overlap=200):
    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())

            current = current[-overlap:] + para + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def embedding_text(chunks):
    return embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )