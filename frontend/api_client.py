from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests


class APIError(RuntimeError):
    """A clean, user-facing error raised for backend/API failures."""


class PDFChatAPI:
    def __init__(self, base_url: str, *, connect_timeout: int = 10, read_timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error") or payload.get("message")
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
            if detail:
                return str(detail)

        text = response.text.strip()
        return text[:500] if text else f"Backend returned HTTP {response.status_code}."

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        raise APIError(self._error_message(response))

    def health(self) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=(5, 15))
            self._raise_for_status(response)
            return response.json()
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the backend at {self.base_url}: {exc}") from exc

    def models(self) -> list[str]:
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=(5, 15))
            self._raise_for_status(response)
            payload = response.json()
            return [str(item["id"]) for item in payload.get("data", []) if item.get("id")]
        except requests.RequestException as exc:
            raise APIError(f"Could not load models: {exc}") from exc

    def upload_pdf(self, filename: str, content: bytes) -> dict[str, Any]:
        if not content:
            raise APIError(f"{filename} is empty.")

        try:
            response = self.session.post(
                f"{self.base_url}/upload",
                files={"file": (filename, content, "application/pdf")},
                timeout=self.timeout,
            )
            self._raise_for_status(response)
            payload = response.json()
        except requests.RequestException as exc:
            raise APIError(f"Upload failed for {filename}: {exc}") from exc
        except ValueError as exc:
            raise APIError("The upload endpoint returned invalid JSON.") from exc

        document_id = payload.get("document_id") or payload.get("id")
        if not document_id:
            raise APIError(
                "Upload succeeded, but the backend response did not contain "
                "`document_id` (or `id`)."
            )
        return payload

    @staticmethod
    def _content_from_event(event: dict[str, Any]) -> str | None:
        if event.get("error"):
            error = event["error"]
            if isinstance(error, dict):
                error = error.get("message") or json.dumps(error, ensure_ascii=False)
            raise APIError(str(error))

        choices = event.get("choices") or []
        if not choices:
            return None

        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        content = delta.get("content")

        # Some OpenAI-compatible providers return structured content parts.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
            return "".join(parts) or None

        if isinstance(content, str):
            return content

        # Defensive fallback for providers that send a final message object.
        message = choice.get("message") or {}
        final_content = message.get("content")
        return final_content if isinstance(final_content, str) else None

    def stream_chat(
        self,
        *,
        messages: list[dict[str, str]],
        document_id: str | None,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "document_id": document_id,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers={"Accept": "text/event-stream"},
            ) as response:
                self._raise_for_status(response)
                # FastAPI sends text/event-stream without always declaring a charset.
                # Force UTF-8 so Bengali and other non-ASCII model output is not mangled.
                response.encoding = "utf-8"
                yielded = False

                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk = self._content_from_event(event)
                    if chunk:
                        yielded = True
                        yield chunk

                if not yielded:
                    raise APIError("The model stream completed without returning any text.")
        except requests.RequestException as exc:
            raise APIError(f"Chat request failed: {exc}") from exc
