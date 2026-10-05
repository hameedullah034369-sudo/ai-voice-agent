import io
from groq import AsyncGroq
from config import settings

class STTEngine:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "input.wav") -> str:
        """
        Transcribes audio bytes into text using Groq's ultra-fast Whisper API.
        """
        try:
            # Create an in-memory file object from raw audio bytes
            audio_file = (filename, audio_bytes, "audio/wav")
            
            transcription = await self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )
            return transcription.text.strip()
        except Exception as e:
            print(f"[STT Error]: {e}")
            return ""
