from gtts import gTTS
import io

class TTSEngine:
    async def text_to_speech_bytes(self, text: str) -> bytes:
        """Converts text into audio bytes using gTTS."""
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
