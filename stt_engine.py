import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

load_dotenv()

class STTEngine:
    def __init__(self):
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            print("[STT Warning]: DEEPGRAM_API_KEY missing in .env")
        self.dg_client = DeepgramClient(api_key)

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        try:
            payload = {"buffer": audio_bytes}
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
            )
            response = await self.dg_client.listen.asyncprerecorded.v("1").transcribe_file(
                payload, options
            )
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript
        except Exception as e:
            print(f"[STT Error]: {e}")
            return ""
