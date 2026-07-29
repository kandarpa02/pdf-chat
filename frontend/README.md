# Paperchat Streamlit UI

A polished three-pane Streamlit frontend for your FastAPI PDF RAG backend:

- Far-left sidebar: new chat, chat history, backend status, settings, account hook
- Centre-left pane: attached PDF list and stable PDF preview
- Right pane: fixed-height chat window with OpenAI-compatible SSE streaming
- Two attachment flows: **Add documents** button and the chat-input paperclip

## Backend endpoints used

| Endpoint | Purpose |
|---|---|
| `GET /health` | Sidebar backend status |
| `GET /v1/models` | Model selection in Settings |
| `POST /upload` | Upload and index each PDF |
| `POST /v1/chat/completions` | Stream chat responses |

The UI intentionally does not use `POST /chat` because `/v1/chat/completions` already supports messages, `document_id`, and streaming.

## Run it

Start your FastAPI backend first:

```powershell
uvicorn python_app.app:app --host 0.0.0.0 --port 8000 --reload
```

Then, from this UI folder:

```powershell
python -m pip install -r requirements.txt
$env:PDF_CHAT_API_URL="http://localhost:8000"
streamlit run app.py
```

Open `http://localhost:8501`.

## Important backend limitation

Your current request model contains one `document_id`, not a list. Therefore this UI can attach and preview several PDFs, but only the **selected PDF** is sent to the backend for a question.

For true multi-document questions, change the backend contract to something like:

```python
document_ids: list[str] = Field(default_factory=list)
```

and update your Qdrant filter/retrieval code to search within those IDs.

## What is demo-level vs SaaS-level

The visual chat history and PDF bytes live in `st.session_state`, so they last for the current Streamlit session. A production SaaS still needs:

1. Authentication and a stable `user_id` or `workspace_id`
2. Chat CRUD endpoints and persistent message storage
3. Document list/delete endpoints
4. Object storage or signed PDF URLs instead of keeping whole PDFs in frontend memory
5. Mandatory tenant filters in every Qdrant search and delete operation
6. Rate limits, usage metering, subscriptions, and audit-safe logging

The Account dialog is deliberately a UI hook rather than fake authentication.
