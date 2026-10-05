# AI Voice Agent — Internship Project

An end-to-end real-time Conversational Voice Agent built using **FastAPI**, **Groq LPU (Llama 3)**, **Groq Whisper**, and **Edge Neural TTS**.

## Features
- **Speech-to-Text (STT):** High-speed audio transcription via Groq Whisper (`whisper-large-v3-turbo`).
- **LLM Engine:** Real-time conversational intelligence using `llama-3.3-70b-versatile` on Groq.
- **Text-to-Speech (TTS):** Free neural audio generation via Microsoft Edge TTS.

## How to Run Locally

1. **Clone Repository & Set Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
