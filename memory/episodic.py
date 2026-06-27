import json
import os
from datetime import datetime
from .base import BaseMemory, MemoryEntry

def _summarize(text: str, model: str) -> str:
    """Use Groq if available, fallback to Ollama."""
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": text}],
            max_tokens=300
        )
        return response.choices[0].message.content
    else:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": text}]
        )
        return response['message']['content']


class EpisodicMemory(BaseMemory):
    def __init__(self, storage_path: str = "./episodes.json", model: str = "llama3.1:8b"):
        self.storage_path = storage_path
        self.model = model
        self._episodes: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(self._episodes, f, indent=2, default=str)

    def store(self, content: str, metadata: dict = None, importance: float = 1.0) -> None:
        episode = {
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._episodes.append(episode)
        self._save()

    def store_conversation(self, messages: list[dict], customer_id: str) -> str:
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ])
        prompt = f"""Summarize this customer support conversation in 3-5 sentences.
Focus on: what the customer issue was, how it was resolved, and any important preferences noted.

CONVERSATION:
{conversation_text}

Summary:"""
        summary = _summarize(prompt, self.model)
        self.store(
            content=summary,
            metadata={"customer_id": customer_id, "type": "conversation_summary"},
            importance=0.8
        )
        return summary

    def retrieve(self, query: str = None, top_k: int = 3) -> list[MemoryEntry]:
        if not self._episodes:
            return []
        relevant = self._episodes[-top_k:]
        return [
            MemoryEntry(
                content=ep["content"],
                metadata=ep.get("metadata", {}),
                importance=ep.get("importance", 1.0),
                timestamp=datetime.fromisoformat(ep["timestamp"])
            )
            for ep in reversed(relevant)
        ]

    def retrieve_for_customer(self, customer_id: str, top_k: int = 3) -> list[MemoryEntry]:
        customer_episodes = [
            ep for ep in self._episodes
            if ep.get("metadata", {}).get("customer_id") == customer_id
        ]
        recent = customer_episodes[-top_k:]
        return [
            MemoryEntry(
                content=ep["content"],
                metadata=ep.get("metadata", {}),
                importance=ep.get("importance", 1.0),
                timestamp=datetime.fromisoformat(ep["timestamp"])
            )
            for ep in reversed(recent)
        ]

    def clear(self) -> None:
        self._episodes = []
        self._save()

    def summary(self) -> dict:
        return {
            "type": "episodic",
            "episode_count": len(self._episodes),
            "storage_path": self.storage_path,
        }
