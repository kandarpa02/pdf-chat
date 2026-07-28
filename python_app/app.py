import os
import time

from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import os

from .database import QdrantDatabase
from .inference import InferenceAPI
from .rag_pipeline import RAGPipeline
from .config import CHAT_MODEL

app = FastAPI()

inference = InferenceAPI()
qdrant = QdrantDatabase()

rag = RAGPipeline(
    inference=inference,
    database=qdrant,
)

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:

        temp.write(await file.read())
        temp_path = temp.name

    try:
        rag.upload_pdf(temp_path)

        return {
            "message": "Upload successful."
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

from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):

    question = request.messages[-1].content

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
                "owned_by": "kandarpa sarkar"
            }
        ]
    }