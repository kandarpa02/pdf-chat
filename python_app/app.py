from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import os

from .database import QdrantDatabase
from .inference import InferenceAPI
from .rag_pipeline import RAGPipeline

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