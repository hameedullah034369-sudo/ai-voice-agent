import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    DEFAULT_VOICE: str = "en-US-AvaNeural"  # Free high-quality neural voice from Edge-TTS

settings = Settings()
