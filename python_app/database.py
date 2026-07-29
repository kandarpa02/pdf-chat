import uuid
import psycopg2
from psycopg2.extras import Json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .config import POSTGRES_HOST, POSTGRES_DB, POSTGRES_PORT, POSTGRES_PASSWORD, POSTGRES_USER

class PostgreSQLDatabase:

    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
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
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)


class QdrantDatabase:

    def __init__(self, host, port):
        self.client = QdrantClient(
            host=host,
            port=port,
        )

        print(
            "Connected to Qdrant:",
            host,
            port
        )


    def create_collection(
        self,
        name,
        vector_size,
    ):

        if not self.client.collection_exists(name):

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

            print(
                f"Created collection: {name}"
            )

        else:

            print(
                f"Collection already exists: {name}"
            )


    def insert(
        self,
        collection_name,
        embedding,
        payload,
    ):

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )


    def search(
        self,
        collection_name,
        embedding,
        limit=5,
    ):

        if not self.client.collection_exists(collection_name):
            raise Exception(
                f"Collection '{collection_name}' does not exist"
            )


        result = self.client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        )

        return result.points