import edge_tts
from config import settings

class TTSEngine:
    def __init__(self):
        self.voice = settings.DEFAULT_VOICE

    async def text_to_speech_bytes(self, text: str) -> bytes:
        """
        Converts text input into MP3 audio bytes using Microsoft Edge Neural TTS.
        """
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_data = bytearray()
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
                    
            return bytes(audio_data)
        except Exception as e:
            print(f"[TTS Error]: {e}")
            return b""
