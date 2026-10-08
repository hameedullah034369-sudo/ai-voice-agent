from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates

from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from intent_engine import IntentEngine
from state_manager import ConversationFlowManager
from prompts import build_system_prompt

app = FastAPI(
    title="Conversational Voice Agent API",
    description="Day 3 Deliverable: System Prompt & Personality Engine Integration",
    version="1.2.0"
)

templates = Jinja2Templates(directory="templates")

stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
intent_engine = IntentEngine()
flow_manager = ConversationFlowManager()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "stage": "Day 3 - System Prompt & Personality Active",
        "current_state": flow_manager.current_state
    }

@app.post("/api/v1/process-voice")
async def process_voice_input(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be audio format.")

    try:
        audio_bytes = await file.read()

        # 1. Speech-to-Text
        user_transcript = await stt.transcribe_audio(audio_bytes, filename=file.filename)
        if not user_transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")
        
        print(f"\n[User Spoke]: {user_transcript}")

        # 2. Intent Detection
        detected_intent = await intent_engine.classify_intent(user_transcript)
        print(f"[Detected Intent]: {detected_intent.intent} (Confidence: {detected_intent.confidence})")

        # 3. State Router Instruction
        system_instruction = flow_manager.process_state_transition(detected_intent)
        print(f"[New State]: {flow_manager.current_state}")

        # 4. Day 3: Build Dynamic System Prompt
        system_prompt = build_system_prompt(
            system_instruction=system_instruction,
            detected_intent=str(detected_intent.intent)
        )

        # 5. LLM Response Generation (Gemini with System Prompt)
        agent_response_text = ""
        async for chunk in llm.generate_response_stream(system_prompt=system_prompt, user_prompt=user_transcript):
            agent_response_text += chunk

        print(f"[Agent Response]: {agent_response_text}")

        # 6. Text-to-Speech
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