from openai import OpenAI
from .config import API_KEY, BASE_URL
class InferenceAPI:

    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    def chat(self, model, messages:list[dict]):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
        )