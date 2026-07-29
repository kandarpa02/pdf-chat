import os
import time
import tempfile

from fastapi import FastAPI, UploadFile, File, Form

from .database import QdrantDatabase
from .inference import InferenceAPI
from .rag_pipeline import RAGPipeline
from .config import (
    CHAT_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
)


app = FastAPI()


inference = InferenceAPI()


qdrant = QdrantDatabase(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


rag = RAGPipeline(
    inference=inference,
    database=qdrant,
)



@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(
        file.filename
    )[1]


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        temp.write(
            await file.read()
        )

        temp_path = temp.name


    try:

        rag.upload_pdf(temp_path)

        return {
            "message": "Upload successful"
        }


    finally:

        os.remove(temp_path)



@app.post("/chat")
async def chat(
    question: str = Form(...)
):

    answer = rag.ask(question)

    return {
        "answer": answer
    }



@app.post("/v1/chat/completions")
async def chat_completion(request: dict):

    question = request["messages"][-1]["content"]

    answer = rag.ask(question)


    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": CHAT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }



@app.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": "pdf-chat",
                "object": "model",
                "owned_by": "kandarpa sarkar",
            }
        ],
    }