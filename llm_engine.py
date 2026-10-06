from typing import AsyncGenerator
from groq import AsyncGroq
from config import settings

class LLMEngine:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generates a fast, streaming response using Llama 3 on Groq LPUs.
        """
        try:
            stream = await self.client.chat.completions.create(
               model="llama-3.1-8b-instant",  
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional, helpful voice assistant. "
                            "Keep your answers brief, clear, and natural for speech synthesis (1-3 sentences)."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=150,
                temperature=0.7
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"[LLM Error]: {e}")
            yield "I am sorry, I encountered an issue processing your request."
