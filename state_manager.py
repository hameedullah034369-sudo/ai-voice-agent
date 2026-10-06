from enum import Enum
from intent_engine import UserIntent, IntentCategory

class ConversationState(str, Enum):
    WELCOME = "WELCOME"
    ACTIVE_DIALOGUE = "ACTIVE_DIALOGUE"
    BOOKING_FLOW = "BOOKING_FLOW"
    TRANSFERRING = "TRANSFERRING"
    COMPLETED = "COMPLETED"

class ConversationFlowManager:
    def __init__(self):
        self.current_state = ConversationState.WELCOME

    def process_state_transition(self, intent_data: UserIntent) -> str:
        """
        Updates internal state and returns instruction context for the LLM response generator.
        """
        intent = intent_data.intent

        if intent == IntentCategory.GREETING:
            self.current_state = ConversationState.ACTIVE_DIALOGUE
            return "Greet the user warmly, introduce yourself briefly, and ask how you can assist them today."

        elif intent == IntentCategory.BOOK_APPOINTMENT:
            self.current_state = ConversationState.BOOKING_FLOW
            return "Acknowledge the booking request and ask for preferred date and time."

        elif intent == IntentCategory.HUMAN_TRANSFER:
            self.current_state = ConversationState.TRANSFERRING
            return "Inform the user politely that you are transferring them to a live support team member."

        elif intent == IntentCategory.GOODBYE:
            self.current_state = ConversationState.COMPLETED
            return "Politely thank the user for calling and wish them a wonderful day."

        elif intent == IntentCategory.UNKNOWN:
            return "Politely inform the user that you didn't quite catch that, and ask them to clarify or rephrase."

        else: # ASK_INFORMATION
            self.current_state = ConversationState.ACTIVE_DIALOGUE
            return "Provide a concise and helpful answer based on the user's question."
