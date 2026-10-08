import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[LLM Warning]: GEMINI_API_KEY missing in .env file!")
        self.client = genai.Client(api_key=api_key)

    async def generate_response_stream(self, system_prompt: str, user_prompt: str):
        """
        Generates a fast, streaming response using Gemini guided by a dynamic system prompt.
        """
        try:
            # Gemini mein System Prompt config parameter ke zariye paas hota hai
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.6,
                max_output_tokens=150
            )

            response = self.client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=config
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0)

        except Exception as e:
            print(f"[LLM Error]: {e}")
            yield "I am sorry, I encountered an issue processing your request."