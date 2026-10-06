from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from intent_engine import IntentEngine
from state_manager import ConversationFlowManager

app = FastAPI(
    title="Conversational Voice Agent API",
    description="Day 2 Deliverable: Intent Classification & State Machine Flow",
    version="1.1.0"
)

stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
intent_engine = IntentEngine()
flow_manager = ConversationFlowManager()

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

        # 1. Speech-to-Text
        user_transcript = await stt.transcribe_audio(audio_bytes, filename=file.filename)
        if not user_transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")
        
        print(f"\n[User Spoke]: {user_transcript}")

        # 2. Intent Detection
        detected_intent = await intent_engine.classify_intent(user_transcript)
        print(f"[Detected Intent]: {detected_intent.intent} (Confidence: {detected_intent.confidence})")
        print(f"[Entities]: {detected_intent.extracted_entities}")

        # 3. State Router Instruction
        system_instruction = flow_manager.process_state_transition(detected_intent)
        print(f"[New State]: {flow_manager.current_state}")

        # 4. LLM Generation guided by state instruction
        prompt = f"Instruction: {system_instruction}\nUser Speech: {user_transcript}"
        agent_response_text = ""
        async for chunk in llm.generate_response_stream(prompt):
            agent_response_text += chunk

        print(f"[Agent Response]: {agent_response_text}")

        # 5. Text-to-Speech
        audio_response_bytes = await tts.text_to_speech_bytes(agent_response_text)

        return Response(
            content=audio_response_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=response.mp3",
                "X-Detected-Intent": detected_intent.intent,
                "X-Conversation-State": flow_manager.current_state
            }
        )

    except Exception as e:
        print(f"[Pipeline Failure]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
