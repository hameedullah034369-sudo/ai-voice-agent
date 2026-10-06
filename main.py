from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine

app = FastAPI(
    title="Conversational Voice Agent API",
    description="Day 1 Deliverable: STT -> LLM -> TTS Pipeline",
    version="1.0.0"
)

# Instantiate engine services
stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()

@app.get("/health")
async def health_check():
    return {"status": "online", "stage": "Day 1 - Base Pipeline Ready"}

@app.post("/api/v1/process-voice")
async def process_voice_input(file: UploadFile = File(...)):
    """
    Processes voice audio input and returns synthesized voice audio output.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be audio format.")

    try:
        # Read raw binary data from upload
        audio_bytes = await file.read()

        # 1. Speech-to-Text
        user_transcript = await stt.transcribe_audio(audio_bytes, filename=file.filename)
        if not user_transcript:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio. Please speak clearly.")
        
        print(f"[User Input]: {user_transcript}")

        # 2. LLM Processing
        agent_response_text = ""
        async for chunk in llm.generate_response_stream(user_transcript):
            agent_response_text += chunk

        print(f"[Agent Response]: {agent_response_text}")

        # 3. Text-to-Speech
        audio_output = tts.synthesize(agent_response_text)

        # Return audio stream/file to client
        return Response(
            content=audio_output, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"}
        )

    except Exception as e:
        print(f"[Error]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
