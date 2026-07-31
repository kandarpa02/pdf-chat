import json
import logging
import os
import tempfile
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import iterate_in_threadpool

from .config import (
    MAX_FILE_SIZE_MB,
    PUBLIC_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_HTTPS,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    RAG_MODE,
)
from .data_parser import PDFError
from .database import CollectionDimensionError, QdrantDatabase, QdrantError
from .inference import InferenceAPI, InferenceConfigurationError
from .rag_pipeline import (
    NoDocumentsError,
    RAGPipeline,
    contains_injected_context,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-chat")

app = FastAPI(
    title="PDF Chat API",
    version="2.0.0",
    description="OpenAI-compatible chat endpoint plus an optional custom Qdrant PDF pipeline.",
)

inference = InferenceAPI()
qdrant = QdrantDatabase(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    https=QDRANT_HTTPS,
    timeout=QDRANT_TIMEOUT,
)
rag = RAGPipeline(inference=inference, database=qdrant)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    max_completion_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    document_id: str | None = None


def _error_status(exc: Exception) -> int:
    if isinstance(exc, (PDFError, ValueError)):
        return 400
    if isinstance(exc, NoDocumentsError):
        return 409
    if isinstance(exc, CollectionDimensionError):
        return 409
    if isinstance(exc, InferenceConfigurationError):
        return 500
    if isinstance(exc, QdrantError):
        return 503
    return 500


def _raise_http(exc: Exception) -> None:
    logger.exception("Request failed: %s", exc)
    raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc


def _has_file_metadata(request: ChatCompletionRequest) -> bool:
    extras = request.model_extra or {}
    if extras.get("files"):
        return True
    return any(bool(message.get("files")) for message in request.messages)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove Open WebUI-only message metadata before forwarding upstream."""
    allowed = {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
        "refusal",
        "audio",
    }
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        clean = {key: value for key, value in message.items() if key in allowed}
        if clean.get("role") == "tool" and clean.get("content") is None:
            clean["content"] = ""
        sanitized.append(clean)
    return sanitized


def _prepare_openai_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    """Choose Open WebUI-injected context or this app's custom Qdrant RAG."""
    raw_messages = [dict(message) for message in request.messages]
    openwebui_supplied_context = contains_injected_context(raw_messages)
    messages = _sanitize_messages(raw_messages)

    if RAG_MODE == "openwebui":
        return messages

    if RAG_MODE == "qdrant":
        return rag.augment_messages(
            messages,
            document_id=request.document_id,
            require_documents=True,
        )

    # auto mode: attachments are owned by Open WebUI. Its retrieval pipeline
    # injects source text into messages before this provider is called. Never
    # replace that context with unrelated vectors from our own collection.
    if openwebui_supplied_context or _has_file_metadata(request):
        return messages

    try:
        return rag.augment_messages(
            messages,
            document_id=request.document_id,
            require_documents=False,
        )
    except QdrantError as exc:
        logger.warning("Skipping optional Qdrant retrieval in auto mode: %s", exc)
        return messages


def _chunk_to_sse(chunk: Any) -> str:
    if hasattr(chunk, "model_dump"):
        data = chunk.model_dump(exclude_none=True)
    elif isinstance(chunk, dict):
        data = chunk
    else:
        data = {"error": "Unsupported upstream stream chunk"}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_response(upstream_stream: Iterator[Any]):
    try:
        async for chunk in iterate_in_threadpool(iter(upstream_stream)):
            yield _chunk_to_sse(chunk)
        yield "data: [DONE]\n\n"
    finally:
        close = getattr(upstream_stream, "close", None)
        if callable(close):
            close()


@app.on_event("startup")
def startup_check() -> None:
    try:
        qdrant.ping()
        logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
    except QdrantError as exc:
        # Open WebUI passthrough mode can still work without Qdrant, so do not
        # make the whole API unavailable. Custom /upload and Qdrant RAG calls
        # will return a clear 503 until Qdrant is reachable.
        logger.warning("Qdrant is unavailable at startup: %s", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    qdrant_ok = True
    qdrant_error = None
    try:
        qdrant.ping()
    except QdrantError as exc:
        qdrant_ok = False
        qdrant_error = str(exc)

    return {
        "status": "ok" if qdrant_ok or RAG_MODE == "openwebui" else "degraded",
        "rag_mode": RAG_MODE,
        "qdrant": {
            "ok": qdrant_ok,
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "error": qdrant_error,
        },
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    suffix = os.path.splitext(filename)[1] or ".pdf"
    temp_path: str | None = None

    try:
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF is larger than the {MAX_FILE_SIZE_MB} MB limit.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = temp.name

        result = await run_in_threadpool(rag.upload_pdf, temp_path, filename)
        return {"message": "Upload successful", **result}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http(exc)
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/chat")
async def chat(
    question: str = Form(...),
    document_id: str | None = Form(default=None),
):
    try:
        answer = await run_in_threadpool(rag.ask, question, document_id)
        return {"answer": answer}
    except Exception as exc:
        _raise_http(exc)


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        messages = await run_in_threadpool(_prepare_openai_messages, request)
        response = await run_in_threadpool(
            inference.chat,
            messages,
            stream=request.stream,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            max_completion_tokens=request.max_completion_tokens,
            stop=request.stop,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
        )

        if request.stream:
            return StreamingResponse(
                _stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        return response
    except Exception as exc:
        _raise_http(exc)


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": PUBLIC_MODEL_NAME,
                "object": "model",
            }
        ],
    }
