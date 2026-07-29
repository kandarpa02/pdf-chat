from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import COLLECTION_NAME, SCORE_THRESHOLD, TOP_K
from .data_parser import parse_pdf_pages
from .preprocessing import chunk_pages, embed_documents, embed_query


class NoDocumentsError(RuntimeError):
    pass


def content_to_text(content: Any) -> str:
    """Extract text from OpenAI string or multimodal content arrays."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def latest_user_question(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = content_to_text(message.get("content"))
            if text.strip():
                return text.strip()
    raise ValueError("No non-empty user message was supplied.")


def contains_injected_context(messages: list[dict[str, Any]]) -> bool:
    """Detect the source wrappers used by Open WebUI's built-in RAG pipeline."""
    for message in messages:
        text = content_to_text(message.get("content")).lower()
        if "<source" in text and "</source>" in text:
            return True
    return False


class RAGPipeline:
    def __init__(self, inference, database):
        self.inference = inference
        self.db = database

    def upload_pdf(self, pdf_path: str, filename: str | None = None) -> dict[str, Any]:
        pages = parse_pdf_pages(pdf_path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No extractable text chunks were produced from the PDF.")

        embeddings = embed_documents([chunk["text"] for chunk in chunks])
        if len(embeddings) == 0:
            raise ValueError("The embedding model returned no vectors.")

        self.db.ensure_collection(
            name=COLLECTION_NAME,
            vector_size=int(embeddings.shape[1]),
        )

        document_id = str(uuid4())
        display_name = filename or Path(pdf_path).name
        payloads = [
            {
                "document_id": document_id,
                "filename": display_name,
                "chunk_id": str(uuid4()),
                "page": chunk["page"],
                "page_chunk": chunk["page_chunk"],
                "text": chunk["text"],
            }
            for chunk in chunks
        ]

        self.db.insert_many(
            collection_name=COLLECTION_NAME,
            embeddings=[embedding.tolist() for embedding in embeddings],
            payloads=payloads,
        )

        return {
            "document_id": document_id,
            "filename": display_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]),
        }

    def has_documents(self) -> bool:
        return self.db.has_points(COLLECTION_NAME)

    def retrieve(
        self,
        question: str,
        *,
        document_id: str | None = None,
    ):
        if not self.has_documents():
            return []

        query_embedding = embed_query(question)
        return self.db.search(
            collection_name=COLLECTION_NAME,
            embedding=query_embedding.tolist(),
            limit=TOP_K,
            document_id=document_id,
            score_threshold=SCORE_THRESHOLD,
        )

    def augment_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        document_id: str | None = None,
        require_documents: bool = False,
    ) -> list[dict[str, Any]]:
        question = latest_user_question(messages)
        results = self.retrieve(question, document_id=document_id)

        if not results:
            if require_documents:
                raise NoDocumentsError(
                    "No indexed PDF content is available. Upload a PDF through POST /upload first."
                )
            return messages

        context_parts: list[str] = []
        for point in results:
            payload = point.payload or {}
            text = payload.get("text")
            if not text:
                continue
            filename = payload.get("filename", "uploaded PDF")
            page = payload.get("page", "?")
            score = getattr(point, "score", None)
            score_label = f", score={score:.3f}" if isinstance(score, float) else ""
            context_parts.append(
                f'<source filename="{filename}" page="{page}"{score_label}>\n{text}\n</source>'
            )

        if not context_parts:
            if require_documents:
                raise NoDocumentsError("No relevant text was found in the indexed PDF.")
            return messages

        rag_instruction = {
            "role": "system",
            "content": (
                "Use the retrieved PDF sources below to answer the user's question. "
                "Do not invent facts that are absent from the sources. When useful, "
                "mention the source filename and page number.\n\n"
                + "\n\n".join(context_parts)
            ),
        }

        # Keep the original conversation intact and place fresh retrieval context
        # immediately before the newest user turn.
        augmented = [dict(message) for message in messages]
        insert_at = len(augmented)
        for index in range(len(augmented) - 1, -1, -1):
            if augmented[index].get("role") == "user":
                insert_at = index
                break
        augmented.insert(insert_at, rag_instruction)
        return augmented

    def ask(self, question: str, document_id: str | None = None) -> str:
        messages = self.augment_messages(
            [{"role": "user", "content": question}],
            document_id=document_id,
            require_documents=True,
        )
        response = self.inference.chat(messages)
        return response.choices[0].message.content or ""
