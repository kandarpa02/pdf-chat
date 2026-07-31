import json
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 PDF Chat")

# -------------------------
# Health
# -------------------------
if st.button("Health Check"):
    r = requests.get(f"{API}/health")
    st.json(r.json())

# -------------------------
# Models
# -------------------------
if st.button("List Models"):
    r = requests.get(f"{API}/v1/models")
    st.json(r.json())

st.divider()

# -------------------------
# Upload PDF
# -------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded and st.button("Upload"):
    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            "application/pdf",
        )
    }

    r = requests.post(
        f"{API}/upload",
        files=files,
    )

    if r.ok:
        payload = r.json()
        st.success("Uploaded!")
        st.json(payload)

        st.session_state.document_id = payload["document_id"]
    else:
        st.error(r.text)

st.divider()

document_id = st.session_state.get("document_id")

st.write("Current document:", document_id)

prompt = st.chat_input("Ask something...")

if prompt:

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    payload = {
        "model": "pdf-chat",
        "messages": messages,
        "stream": True,
        "document_id": document_id,
    }

    response = requests.post(
        f"{API}/v1/chat/completions",
        json=payload,
        stream=True,
        headers={
            "Accept": "text/event-stream",
        },
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()
        answer = ""

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                break

            event = json.loads(data)

            choices = event.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})
            chunk = delta.get("content", "")

            if chunk:
                answer += chunk
                placeholder.markdown(answer + "▌")

        placeholder.markdown(answer)