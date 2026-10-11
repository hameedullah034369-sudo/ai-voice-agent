import json
import os

class RAGRetriever:
    def __init__(self, kb_path="knowledge_base.json"):
        self.kb_path = kb_path
        self.kb_data = self._load_kb()

    def _load_kb(self) -> dict:
        """Loads the knowledge base JSON file safely."""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, "r") as f:
                return json.load(f)
        return {}

    def retrieve_context(self, user_query: str) -> str:
        """
        Keyword-based retrieval to find the most relevant knowledge base entry 
        matching the user's spoken query.
        """
        query_lower = user_query.lower()
        matched_contexts = []

        for key, value in self.kb_data.items():
            if key in query_lower or any(word in query_lower for word in value.lower().split()[:5]):
                matched_contexts.append(f"{key.upper()}: {value}")

        if matched_contexts:
            return " \n".join(matched_contexts)
        
        return f"Company Overview: {self.kb_data.get('company_name')} \n{self.kb_data.get('services')}"