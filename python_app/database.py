import uuid
import psycopg2
from psycopg2.extras import Json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .config import settings

class PostgreSQLDatabase:

    def __init__(self):
        self.conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )

    def execute(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            self.conn.commit()

    def fetchone(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetchall(self, query, params=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def close(self):
        self.conn.close()


class QdrantDatabase:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upload_embedding(
        self,
        collection_name: str,
        point_id,
        embedding,
        payload: dict,
    ):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

class RAGDatabase:

    def __init__(self, postgres, qdrant):
        self.pg = postgres
        self.qdrant = qdrant

    def save_document(
        self,
        collection_name,
        embedding,
        text,
        metadata,
    ):
        document_id = str(uuid.uuid4())

        # PostgreSQL
        self.pg.execute(
            """
            INSERT INTO documents(id, text, metadata)
            VALUES (%s, %s, %s)
            """,
            (
                document_id,
                text,
                Json(metadata),
            ),
        )

        self.qdrant.upload_embedding(
            collection_name=collection_name,
            point_id=document_id,
            embedding=embedding,
            payload=metadata,
        )

        return document_id