from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_BASE_URL

class InferenceAPI:

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def chat(self, model, messages:list[dict]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
        )