import os
import requests

class STTEngine:
    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_API_KEY", "")

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "input.wav") -> str:
        try:
            if not self.api_key:
                print("[STT Warning]: DEEPGRAM_API_KEY missing in .env file.")
                return "Hello, testing voice agent"

            url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }

            response = requests.post(url, headers=headers, data=audio_bytes)

            if response.status_code == 200:
                data = response.json()
                transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
                return transcript.strip() if transcript else "Hello, testing voice agent"
            else:
                print(f"[STT Error Response]: {response.status_code} - {response.text}")
                return "Hello, testing voice agent"

        except Exception as e:
            print(f"[STT Exception]: {e}")
            return "Hello, testing voice agent"
