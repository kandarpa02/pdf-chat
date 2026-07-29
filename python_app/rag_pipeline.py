from uuid import uuid4

from .config import COLLECTION_NAME, TOP_K, CHAT_MODEL
from .data_parser import parse_pdf
from .preprocessing import chunk_text, embedding_text


class RAGPipeline:

    def __init__(self, inference, database):
        self.inference = inference
        self.db = database

    def upload_pdf(self, pdf_path):
        text = parse_pdf(pdf_path)

        print("PDF extracted chars:", len(text))

        chunks = chunk_text(text)

        print("Total chunks:", len(chunks))

        embeddings = embedding_text(chunks)

        print(
            "Embedding dimension:",
            len(embeddings[0])
        )


        self.db.create_collection(
            name=COLLECTION_NAME,
            vector_size=len(embeddings[0]),
        )


        for chunk, embedding in zip(chunks, embeddings):

            self.db.insert(
                collection_name=COLLECTION_NAME,
                embedding=embedding.tolist(),
                payload={
                    "chunk_id": str(uuid4()),
                    "text": chunk,
                },
            )


        print(
            "Current collections:",
            self.db.client.get_collections()
        )

        print("Upload complete")

    def ask(self, question):

        query_embedding = embedding_text([question])[0]

        results = self.db.search(
            collection_name=COLLECTION_NAME,
            embedding=query_embedding.tolist(),
            limit=TOP_K,
        )

        if not results:
            return "No relevant information found in the uploaded PDF."


        context = "\n\n".join(
            point.payload["text"]
            for point in results
        )


        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only using the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]


        response = self.inference.chat(
            CHAT_MODEL,
            messages,
        )

        return response.choices[0].message.content