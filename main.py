from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from stt_engine import STTEngine
from llm_engine import LLMEngine
from tts_engine import TTSEngine
from intent_engine import IntentEngine
from state_manager import ConversationFlowManager
from prompts import build_system_prompt
from rag_retriever import RAGRetriever

app = FastAPI(
    title="Conversational Voice Agent API",
    description="Day 4 Deliverable: Knowledge Base & RAG Integration",
    version="1.3.0"
)

templates = Jinja2Templates(directory="templates")

stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
intent_engine = IntentEngine()
flow_manager = ConversationFlowManager()
rag = RAGRetriever()  # Initialize RAG knowledge base

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "stage": "Day 4 - Knowledge Base & RAG Active",
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
        print(f"[Detected Intent]: {detected_intent.intent}")

        # 3. State Router Instruction
        system_instruction = flow_manager.process_state_transition(detected_intent)

        # 4. Day 4: Retrieve Knowledge Base Context (RAG)
        rag_context = rag.retrieve_context(user_transcript)
        print(f"[RAG Retrieved Context]: {rag_context}")

        # 5. Build System Prompt with Personality, State, and RAG Data
        combined_instruction = f"{system_instruction}\nVerified Knowledge Base Facts:\n{rag_context}"
        system_prompt = build_system_prompt(
            system_instruction=combined_instruction,
            detected_intent=str(detected_intent.intent)
        )

        # 6. LLM Response Generation
        agent_response_text = ""
        async for chunk in llm.generate_response_stream(system_prompt=system_prompt, user_prompt=user_transcript):
            agent_response_text += chunk

        print(f"[Agent Response]: {agent_response_text}")

        # 7. Text-to-Speech
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