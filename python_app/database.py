import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantError(RuntimeError):
    pass


class CollectionDimensionError(QdrantError):
    pass


class QdrantDatabase:
    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        https: bool = False,
        timeout: float = 10.0,
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            timeout=timeout,
        )
        self.host = host
        self.port = port

    def ping(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:
            raise QdrantError(
                f"Could not connect to Qdrant at {self.host}:{self.port}: {exc}"
            ) from exc

    def collection_exists(self, name: str) -> bool:
        try:
            return self.client.collection_exists(name)
        except Exception as exc:
            raise QdrantError(f"Could not check Qdrant collection '{name}': {exc}") from exc

    def ensure_collection(self, name: str, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        try:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                return

            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            existing_size = getattr(vectors, "size", None)
            if existing_size is not None and existing_size != vector_size:
                raise CollectionDimensionError(
                    f"Collection '{name}' uses {existing_size}-dimensional vectors, "
                    f"but the current embedding model produces {vector_size}. "
                    "Use a new COLLECTION_NAME or delete/recreate the old collection."
                )
        except CollectionDimensionError:
            raise
        except Exception as exc:
            raise QdrantError(f"Could not prepare Qdrant collection '{name}': {exc}") from exc

    # Backward-compatible name used by the original project.
    def create_collection(self, name: str, vector_size: int) -> None:
        self.ensure_collection(name, vector_size)

    def insert_many(
        self,
        collection_name: str,
        embeddings: Iterable[list[float]],
        payloads: Iterable[dict],
    ) -> None:
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
            for embedding, payload in zip(embeddings, payloads)
        ]
        if not points:
            return

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise QdrantError(f"Could not insert vectors into Qdrant: {exc}") from exc

    def insert(self, collection_name: str, embedding: list[float], payload: dict) -> None:
        self.insert_many(collection_name, [embedding], [payload])

    def has_points(self, collection_name: str) -> bool:
        if not self.collection_exists(collection_name):
            return False
        try:
            return self.client.count(
                collection_name=collection_name,
                exact=False,
            ).count > 0
        except Exception as exc:
            raise QdrantError(f"Could not count points in '{collection_name}': {exc}") from exc

    def search(
        self,
        collection_name: str,
        embedding: list[float],
        limit: int = 5,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ):
        if not self.collection_exists(collection_name):
            return []

        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        try:
            result = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            return result.points
        except Exception as exc:
            raise QdrantError(f"Qdrant search failed: {exc}") from exc

    def delete_document(self, collection_name: str, document_id: str) -> None:
        if not self.collection_exists(collection_name):
            return
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                wait=True,
            )
        except Exception as exc:
            raise QdrantError(f"Could not delete document '{document_id}': {exc}") from exc
