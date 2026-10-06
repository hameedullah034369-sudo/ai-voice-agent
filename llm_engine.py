import os
from google import genai

class LLMEngine:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def generate_response_stream(self, prompt: str):
        """
        Generates streaming response using Gemini 2.5 Flash.
        """
        try:
            response = self.client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"[LLM Error]: {e}")
            yield "I am sorry, I encountered an issue processing your request."
