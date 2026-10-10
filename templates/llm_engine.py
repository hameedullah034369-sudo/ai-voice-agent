import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[LLM Warning]: GEMINI_API_KEY missing in .env file!")
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"[LLM Error]: {e}")
            return "I am having trouble processing that right now."
