import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Keep these unit tests light: no model download is needed to test chunking and
# RAG control flow.
stub_sentence_transformers = types.ModuleType("sentence_transformers")
stub_sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", stub_sentence_transformers)

from python_app.preprocessing import chunk_text
from python_app.rag_pipeline import (
    NoDocumentsError,
    RAGPipeline,
    contains_injected_context,
    latest_user_question,
)


class FakeDatabase:
    def __init__(self, points=None):
        self.points = points or []
        self.inserted = []
        self.collection = None

    def has_points(self, collection_name):
        return bool(self.points)

    def search(self, **kwargs):
        return self.points

    def ensure_collection(self, name, vector_size):
        self.collection = (name, vector_size)

    def insert_many(self, collection_name, embeddings, payloads):
        self.inserted = list(zip(embeddings, payloads))


class FakeInference:
    def chat(self, messages, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )


class CoreTests(unittest.TestCase):
    def test_chunk_text_handles_long_paragraph_without_empty_chunks(self):
        chunks = chunk_text("A" * 2500, chunk_size=1000, overlap=200)
        self.assertEqual([len(chunk) for chunk in chunks], [1000, 1000, 900])
        self.assertTrue(all(chunks))

    def test_latest_user_question_supports_content_arrays(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is this PDF about?"}],
            }
        ]
        self.assertEqual(latest_user_question(messages), "What is this PDF about?")

    def test_openwebui_source_context_is_detected(self):
        messages = [
            {
                "role": "user",
                "content": '<source id="1">PDF text</source>\nQuestion',
            }
        ]
        self.assertTrue(contains_injected_context(messages))

    def test_missing_collection_returns_original_messages_when_optional(self):
        messages = [{"role": "user", "content": "hello"}]
        pipeline = RAGPipeline(FakeInference(), FakeDatabase())
        self.assertEqual(
            pipeline.augment_messages(messages, require_documents=False),
            messages,
        )

    def test_missing_collection_is_clear_when_required(self):
        pipeline = RAGPipeline(FakeInference(), FakeDatabase())
        with self.assertRaises(NoDocumentsError):
            pipeline.augment_messages(
                [{"role": "user", "content": "hello"}],
                require_documents=True,
            )

    def test_retrieval_context_preserves_page_metadata(self):
        point = SimpleNamespace(
            payload={"text": "The answer is 42.", "filename": "guide.pdf", "page": 7},
            score=0.91,
        )
        pipeline = RAGPipeline(FakeInference(), FakeDatabase([point]))
        with patch("python_app.rag_pipeline.embed_query", return_value=np.array([0.1, 0.2])):
            messages = pipeline.augment_messages(
                [{"role": "user", "content": "What is the answer?"}],
                require_documents=True,
            )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('filename="guide.pdf"', messages[0]["content"])
        self.assertIn('page="7"', messages[0]["content"])

    def test_upload_batches_vectors_and_returns_document_id(self):
        database = FakeDatabase()
        pipeline = RAGPipeline(FakeInference(), database)
        fake_pages = [SimpleNamespace(page_number=1, text="hello")]
        fake_chunks = [{"text": "hello", "page": 1, "page_chunk": 0}]
        fake_vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        with (
            patch("python_app.rag_pipeline.parse_pdf_pages", return_value=fake_pages),
            patch("python_app.rag_pipeline.chunk_pages", return_value=fake_chunks),
            patch("python_app.rag_pipeline.embed_documents", return_value=fake_vectors),
        ):
            result = pipeline.upload_pdf("anything.pdf", "anything.pdf")

        self.assertEqual(database.collection[1], 3)
        self.assertEqual(len(database.inserted), 1)
        self.assertEqual(result["chunks"], 1)
        self.assertTrue(result["document_id"])


if __name__ == "__main__":
    unittest.main()
