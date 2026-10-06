import json
import os
from enum import Enum
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class IntentCategory(str, Enum):
    GREETING = "GREETING"
    ASK_INFORMATION = "ASK_INFORMATION"
    BOOK_APPOINTMENT = "BOOK_APPOINTMENT"
    HUMAN_TRANSFER = "HUMAN_TRANSFER"
    UNKNOWN = "UNKNOWN"
    GOODBYE = "GOODBYE"

class UserIntent(BaseModel):
    intent: IntentCategory = Field(description="The primary classified intent of the user.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    extracted_entities: dict = Field(default_factory=dict, description="Any key details extracted, e.g., name, date, topic.")

class IntentEngine:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def classify_intent(self, user_text: str) -> UserIntent:
        """
        Classifies user intent using Gemini 2.5 Flash in Structured JSON output mode.
        """
        system_prompt = (
            "You are an intent classification system for a voice agent. "
            "Analyze the user's spoken input and classify it into EXACTLY ONE of these categories: "
            f"{[e.value for e in IntentCategory]}.\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "intent": "CATEGORY_NAME",\n'
            '  "confidence": 0.95,\n'
            '  "extracted_entities": {"key": "value"}\n'
            "}"
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{system_prompt}\nUser said: '{user_text}'",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            parsed = json.loads(response.text)
            return UserIntent(**parsed)
        except Exception as e:
            print(f"[Intent Engine Error]: {e}")
            return UserIntent(intent=IntentCategory.UNKNOWN, confidence=0.0, extracted_entities={})
