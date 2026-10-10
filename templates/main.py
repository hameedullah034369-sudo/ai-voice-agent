import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from intent_engine import IntentEngine
from state_manager import ConversationFlowManager
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(
    title="Conversational Voice Agent API",
    description="Day 2 Deliverable: Intent Classification & State Machine Flow",
    version="1.1.0"
)

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
intent_engine = IntentEngine()
flow_manager = ConversationFlowManager()

# Updated TemplateResponse syntax for Python 3.14 / Starlette compatibility
@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "stage": "Day 2 - Intent Detection & State Machine Active",
        "current_state": flow_manager.current_state
    }

@app.post("/api/v1/process-voice")
async def process_voice_input(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be audio format.")

    try:
        audio_bytes = await file.read()

        user_transcript = await stt.transcribe_audio(audio_bytes, filename=file.filename)
        if not user_transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")

        detected_intent = await intent_engine.classify_intent(user_transcript)
        system_instruction = flow_manager.process_state_transition(detected_intent)

        prompt = f"Instruction: {system_instruction}\nUser Speech: {user_transcript}"
        agent_response_text = ""
        async for chunk in llm.generate_response_stream(prompt):
            agent_response_text += chunk

        audio_response_bytes = await tts.text_to_speech_bytes(agent_response_text)

        return Response(
            content=audio_response_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=response.mp3",
                "X-Detected-Intent": str(detected_intent.intent),
                "X-Conversation-State": str(flow_manager.current_state)
            }
        )

    except Exception as e:
        print(f"[Pipeline Failure]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
