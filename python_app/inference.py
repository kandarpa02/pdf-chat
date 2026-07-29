from typing import Any

from openai import OpenAI

from .config import API_KEY, BASE_URL, CHAT_MODEL


class InferenceConfigurationError(RuntimeError):
    pass


class InferenceAPI:
    def __init__(
        self,
        api_key: str | None = API_KEY,
        base_url: str = BASE_URL,
        model: str | None = CHAT_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if not self.api_key:
            raise InferenceConfigurationError(
                "No upstream API key is configured. Set OPENAI_API_KEY, GROQ_API_KEY, or API_KEY in .env."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: str | list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        if not self.model:
            raise InferenceConfigurationError(
                "CHAT_MODEL is missing. Add the upstream model ID to your .env file."
            )

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        optional = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        return self.client.chat.completions.create(**params)
