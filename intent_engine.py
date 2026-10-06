import json
from enum import Enum
from pydantic import BaseModel, Field
from groq import AsyncGroq
from config import settings

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
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def classify_intent(self, user_text: str) -> UserIntent:
        """
        Classifies user intent into predefined categories using Groq Llama 3 JSON mode.
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
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User said: '{user_text}'"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            return UserIntent(**parsed)
        except Exception as e:
            print(f"[Intent Engine Error]: {e}")
            return UserIntent(intent=IntentCategory.UNKNOWN, confidence=0.0, extracted_entities={})
