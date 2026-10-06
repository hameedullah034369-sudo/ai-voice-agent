import io
from gtts import gTTS

def text_to_speech(text: str) -> bytes:
    # Google TTS se text ko MP3 bytes mein convert karta hai
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()
