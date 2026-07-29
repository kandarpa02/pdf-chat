from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from api_client import APIError, PDFChatAPI


APP_NAME = "Paperchat"
DEFAULT_API_URL = os.getenv("PDF_CHAT_API_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("PUBLIC_MODEL_NAME", "pdf-chat")
MAX_UPLOAD_MB = int(os.getenv("PDF_CHAT_MAX_UPLOAD_MB", "50"))
CHAT_PANEL_HEIGHT = 670
PDF_PANEL_HEIGHT = 720

st.set_page_config(
    page_title=f"{APP_NAME} · PDF chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --panel-border: color-mix(in srgb, var(--text-color) 13%, transparent);
                --muted-text: color-mix(in srgb, var(--text-color) 58%, transparent);
                --soft-bg: color-mix(in srgb, var(--background-color) 86%, var(--secondary-background-color));
            }

            #MainMenu, footer { visibility: hidden; }
            [data-testid="stToolbar"] { display: none; }
            [data-testid="stDecoration"] { display: none; }
            header[data-testid="stHeader"] { background: transparent; }

            .block-container {
                max-width: 100%;
                padding-top: 1.05rem;
                padding-bottom: 0.8rem;
                padding-left: 1.15rem;
                padding-right: 1.15rem;
            }

            [data-testid="stSidebar"] {
                min-width: 272px;
                max-width: 272px;
                border-right: 1px solid var(--panel-border);
            }

            [data-testid="stSidebar"] > div:first-child {
                padding-top: 1rem;
            }

            .brand-row {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                margin: 0 0 0.8rem 0.1rem;
            }

            .brand-mark {
                display: grid;
                place-items: center;
                width: 34px;
                height: 34px;
                border-radius: 11px;
                background: linear-gradient(145deg, #7c3aed, #4f46e5);
                color: white;
                font-size: 18px;
                box-shadow: 0 8px 24px rgba(79, 70, 229, 0.23);
            }

            .brand-name {
                font-size: 1.03rem;
                font-weight: 750;
                line-height: 1.1;
                letter-spacing: -0.02em;
            }

            .brand-subtitle {
                color: var(--muted-text);
                font-size: 0.74rem;
                margin-top: 0.12rem;
            }

            .section-label {
                color: var(--muted-text);
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin: 0.75rem 0 0.35rem 0.15rem;
            }

            .pane-title {
                font-size: 1.03rem;
                font-weight: 720;
                letter-spacing: -0.015em;
                margin: 0.08rem 0 0 0;
            }

            .pane-caption {
                color: var(--muted-text);
                font-size: 0.78rem;
                margin-top: 0.12rem;
            }

            .document-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                max-width: 100%;
                border: 1px solid var(--panel-border);
                border-radius: 999px;
                padding: 0.27rem 0.58rem;
                color: var(--muted-text);
                background: var(--soft-bg);
                font-size: 0.75rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .empty-state {
                min-height: 610px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                padding: 2rem;
                border: 1px dashed var(--panel-border);
                border-radius: 18px;
                background: var(--soft-bg);
            }

            .empty-icon {
                width: 52px;
                height: 52px;
                display: grid;
                place-items: center;
                border-radius: 16px;
                background: color-mix(in srgb, #6d5dfc 16%, transparent);
                font-size: 24px;
                margin-bottom: 0.9rem;
            }

            .empty-title { font-weight: 720; margin-bottom: 0.25rem; }
            .empty-copy { color: var(--muted-text); font-size: 0.84rem; max-width: 280px; }

            div[data-testid="stButton"] > button,
            div[data-testid="stPopover"] > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 11px;
                min-height: 2.42rem;
                transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
            }

            div[data-testid="stButton"] > button:hover,
            div[data-testid="stPopover"] > button:hover {
                transform: translateY(-1px);
            }

            .st-key-history_list {
                border: 0 !important;
                background: transparent !important;
            }

            .st-key-chat_messages {
                border-radius: 16px;
                border: 1px solid var(--panel-border) !important;
                background: color-mix(in srgb, var(--background-color) 97%, var(--secondary-background-color));
                padding: 0.65rem 0.7rem;
            }

            div[data-testid="stChatMessage"] {
                border-radius: 14px;
                padding-top: 0.55rem;
                padding-bottom: 0.55rem;
            }

            div[data-testid="stChatInput"] {
                border-radius: 14px;
            }

            div[data-testid="stFileUploaderDropzone"] {
                border-radius: 14px;
                border-style: dashed;
            }

            [data-testid="stStatusWidget"] { visibility: hidden; }

            @media (max-width: 1000px) {
                [data-testid="stSidebar"] {
                    min-width: 244px;
                    max-width: 244px;
                }
                .block-container { padding-left: 0.65rem; padding-right: 0.65rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_chat() -> dict[str, Any]:
    chat_id = uuid.uuid4().hex
    return {
        "id": chat_id,
        "title": "New chat",
        "messages": [],
        "documents": [],
        "active_document_id": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def initialise_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = {
            "api_url": DEFAULT_API_URL,
            "model": DEFAULT_MODEL,
            "temperature": 0.2,
            "max_tokens": 1400,
        }

    if "chats" not in st.session_state:
        first = make_chat()
        st.session_state.chats = {first["id"]: first}
        st.session_state.active_chat_id = first["id"]

    if st.session_state.get("active_chat_id") not in st.session_state.chats:
        replacement = make_chat()
        st.session_state.chats[replacement["id"]] = replacement
        st.session_state.active_chat_id = replacement["id"]


def active_chat() -> dict[str, Any]:
    return st.session_state.chats[st.session_state.active_chat_id]


def touch(chat: dict[str, Any]) -> None:
    chat["updated_at"] = now_iso()


def create_new_chat() -> None:
    chat = make_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.active_chat_id = chat["id"]


def select_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def api_client() -> PDFChatAPI:
    return PDFChatAPI(st.session_state.config["api_url"])


@st.cache_data(ttl=10, show_spinner=False)
def cached_health(api_url: str) -> dict[str, Any]:
    return PDFChatAPI(api_url).health()


@st.cache_data(ttl=60, show_spinner=False)
def cached_models(api_url: str) -> list[str]:
    return PDFChatAPI(api_url).models()


def file_fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def upload_documents(files: list[Any], api: PDFChatAPI) -> list[str]:
    chat = active_chat()
    existing_hashes = {doc["sha256"] for doc in chat["documents"]}
    added_names: list[str] = []

    for uploaded_file in files:
        filename = uploaded_file.name or "upload.pdf"
        content = uploaded_file.getvalue()
        fingerprint = file_fingerprint(content)

        if fingerprint in existing_hashes:
            matching = next(doc for doc in chat["documents"] if doc["sha256"] == fingerprint)
            chat["active_document_id"] = matching["id"]
            continue

        result = api.upload_pdf(filename, content)
        document_id = str(result.get("document_id") or result.get("id"))
        document = {
            "id": document_id,
            "name": filename,
            "bytes": content,
            "size": len(content),
            "sha256": fingerprint,
            "upload_result": result,
        }
        chat["documents"].append(document)
        chat["active_document_id"] = document_id
        existing_hashes.add(fingerprint)
        added_names.append(filename)

    if added_names:
        touch(chat)
    return added_names


def set_active_document() -> None:
    chat = active_chat()
    key = f"document_selector_{chat['id']}"
    selected = st.session_state.get(key)
    if selected:
        chat["active_document_id"] = selected
        touch(chat)


def active_document(chat: dict[str, Any]) -> dict[str, Any] | None:
    active_id = chat.get("active_document_id")
    return next((doc for doc in chat["documents"] if doc["id"] == active_id), None)


def safe_history_messages(chat: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in chat["messages"]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def title_from_prompt(prompt: str) -> str:
    compact = " ".join(prompt.split())
    return compact if len(compact) <= 38 else compact[:37].rstrip() + "…"


@st.dialog("Add documents", width="medium", icon=":material/note_add:")
def add_documents_dialog() -> None:
    st.caption(
        "Upload one or more PDFs. They will be shown in this chat, but the selected PDF "
        "is the one sent as `document_id` to your current backend."
    )
    files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_MB,
        key=f"dialog_upload_{st.session_state.active_chat_id}",
    )

    if st.button(
        "Upload and add",
        type="primary",
        icon=":material/upload:",
        width="stretch",
        disabled=not files,
    ):
        try:
            with st.status("Indexing documents…", expanded=True) as status:
                names = upload_documents(list(files), api_client())
                if names:
                    for name in names:
                        st.write(f"Added **{name}**")
                    status.update(label="Documents ready", state="complete", expanded=False)
                else:
                    status.update(label="Already attached", state="complete", expanded=False)
            st.rerun()
        except APIError as exc:
            st.error(str(exc))


@st.dialog("Settings", width="small", icon=":material/settings:")
def settings_dialog() -> None:
    config = st.session_state.config
    with st.form("settings_form"):
        api_url = st.text_input("Backend URL", value=config["api_url"])

        try:
            model_options = cached_models(api_url.strip())
        except APIError:
            model_options = []

        current_model = config["model"]
        if current_model and current_model not in model_options:
            model_options.insert(0, current_model)
        if not model_options:
            model_options = [DEFAULT_MODEL]

        model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(current_model) if current_model in model_options else 0,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(config["temperature"]),
            step=0.05,
        )
        max_tokens = st.number_input(
            "Maximum response tokens",
            min_value=128,
            max_value=8192,
            value=int(config["max_tokens"]),
            step=128,
        )
        submitted = st.form_submit_button(
            "Save changes",
            type="primary",
            icon=":material/check:",
            width="stretch",
        )

    if submitted:
        st.session_state.config = {
            "api_url": api_url.strip().rstrip("/"),
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        cached_health.clear()
        cached_models.clear()
        st.rerun()


@st.dialog("Account", width="small", icon=":material/account_circle:")
def account_dialog() -> None:
    st.markdown("### Guest workspace")
    st.caption("This is the UI hook for your future authentication and billing layer.")
    st.info(
        "Your current backend has no login, user, subscription, or tenant endpoints yet, "
        "so this demo intentionally does not fake them.",
        icon=":material/info:",
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-row">
                <div class="brand-mark">P</div>
                <div>
                    <div class="brand-name">{APP_NAME}</div>
                    <div class="brand-subtitle">chat with your documents</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "New chat",
            type="primary",
            icon=":material/add:",
            width="stretch",
            on_click=create_new_chat,
        )

        st.markdown('<div class="section-label">Recent chats</div>', unsafe_allow_html=True)
        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda item: item["updated_at"],
            reverse=True,
        )

        with st.container(height=475, border=False, key="history_list"):
            for chat in ordered:
                is_active = chat["id"] == st.session_state.active_chat_id
                st.button(
                    chat["title"],
                    key=f"chat_{chat['id']}",
                    type="primary" if is_active else "tertiary",
                    icon=":material/chat_bubble:" if is_active else ":material/chat_bubble_outline:",
                    width="stretch",
                    on_click=select_chat,
                    args=(chat["id"],),
                )

        try:
            health = cached_health(st.session_state.config["api_url"])
            is_ok = health.get("status") in {"ok", "degraded"}
            status_text = "Backend online" if is_ok else "Backend unavailable"
            status_icon = "🟢" if is_ok else "🔴"
        except APIError:
            status_text = "Backend offline"
            status_icon = "🔴"

        st.caption(f"{status_icon} {status_text}")
        settings_col, account_col = st.columns(2, gap="small")
        with settings_col:
            if st.button(
                "Settings",
                icon=":material/settings:",
                width="stretch",
                key="settings_button",
            ):
                settings_dialog()
        with account_col:
            if st.button(
                "Account",
                icon=":material/person:",
                width="stretch",
                key="account_button",
            ):
                account_dialog()


def render_document_panel() -> None:
    chat = active_chat()
    docs = chat["documents"]

    top_left, top_right = st.columns([1, 0.62], vertical_alignment="center", gap="small")
    with top_left:
        st.markdown('<div class="pane-title">Documents</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pane-caption">Preview and choose the active PDF</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button(
            "Add documents",
            type="secondary",
            icon=":material/note_add:",
            width="stretch",
            key=f"add_document_{chat['id']}",
        ):
            add_documents_dialog()

    if not docs:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📄</div>
                <div class="empty-title">No document attached</div>
                <div class="empty-copy">
                    Add a PDF here or use the paperclip in the chat box. Your preview will stay in this pane.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    doc_ids = [doc["id"] for doc in docs]
    selected_id = chat.get("active_document_id")
    if selected_id not in doc_ids:
        selected_id = doc_ids[-1]
        chat["active_document_id"] = selected_id

    st.selectbox(
        "Document",
        options=doc_ids,
        index=doc_ids.index(selected_id),
        format_func=lambda value: next(doc["name"] for doc in docs if doc["id"] == value),
        key=f"document_selector_{chat['id']}",
        on_change=set_active_document,
        label_visibility="collapsed",
    )

    document = active_document(chat)
    if document is None:
        st.warning("Select a document to preview it.")
        return

    size_mb = document["size"] / (1024 * 1024)
    st.markdown(
        f'<div class="document-chip">📎 {document["name"]} · {size_mb:.1f} MB</div>',
        unsafe_allow_html=True,
    )
    st.pdf(
        document["bytes"],
        height=PDF_PANEL_HEIGHT,
        key=f"pdf_{document['id']}",
    )


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    avatar = ":material/person:" if role == "user" else ":material/auto_awesome:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))


def render_chat_panel() -> None:
    chat = active_chat()
    document = active_document(chat)

    header_left, header_right = st.columns([1, 0.8], vertical_alignment="center", gap="small")
    with header_left:
        st.markdown(f'<div class="pane-title">{chat["title"]}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pane-caption">Answers stream from your FastAPI RAG backend</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        label = document["name"] if document else "No active PDF"
        st.markdown(f'<div class="document-chip">📄 {label}</div>', unsafe_allow_html=True)

    messages_box = st.container(
        height=CHAT_PANEL_HEIGHT,
        border=True,
        key="chat_messages",
        autoscroll=True,
    )
    with messages_box:
        if not chat["messages"]:
            st.markdown("### Ask anything about your PDF")
            st.caption(
                "Attach a document, then ask for summaries, evidence, comparisons, page references, or explanations."
            )
        else:
            for message in chat["messages"]:
                render_message(message)

    submission = st.chat_input(
        "Ask about the selected document…",
        accept_file="multiple",
        file_type=["pdf"],
        max_upload_size=MAX_UPLOAD_MB,
        submit_mode="disable",
        key=f"chat_input_{chat['id']}",
    )

    if not submission:
        return

    prompt = (submission.text or "").strip()
    attached_files = list(submission.files or [])

    if attached_files:
        try:
            with st.spinner("Indexing attached document…"):
                upload_documents(attached_files, api_client())
            document = active_document(chat)
        except APIError as exc:
            with messages_box:
                st.error(str(exc))
            return

    if not prompt:
        with messages_box:
            st.success("Document added. Ask a question whenever you’re ready.")
        return

    if document is None:
        with messages_box:
            st.warning("Attach a PDF first, then send your question.")
        return

    user_message = {"role": "user", "content": prompt}
    chat["messages"].append(user_message)
    if chat["title"] == "New chat":
        chat["title"] = title_from_prompt(prompt)
    touch(chat)

    with messages_box:
        render_message(user_message)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            try:
                response_text = st.write_stream(
                    api_client().stream_chat(
                        messages=safe_history_messages(chat),
                        document_id=document["id"],
                        model=st.session_state.config["model"],
                        temperature=st.session_state.config["temperature"],
                        max_tokens=st.session_state.config["max_tokens"],
                    ),
                    cursor="▌",
                )
            except APIError as exc:
                st.error(str(exc))
                return

    if isinstance(response_text, str) and response_text.strip():
        chat["messages"].append({"role": "assistant", "content": response_text})
        touch(chat)


inject_css()
initialise_state()
render_sidebar()

preview_col, chat_col = st.columns([0.42, 0.58], gap="medium", vertical_alignment="top")
with preview_col:
    render_document_panel()
with chat_col:
    render_chat_panel()
