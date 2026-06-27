import os
from .base import BaseMemory, MemoryEntry

def _reflect(text: str, model: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": text}],
            max_tokens=400
        )
        return response.choices[0].message.content
    else:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": text}]
        )
        return response['message']['content']


class ReflectionMemory(BaseMemory):
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._reflections: list[MemoryEntry] = []

    def generate_reflection(self, episodes: list[str], context: str = "") -> str:
        if not episodes:
            return "No episodes available for reflection."
        episodes_text = "\n\n".join([f"Episode {i+1}: {ep}" for i, ep in enumerate(episodes)])
        prompt = f"""Analyze these past customer support interactions and extract patterns.

PAST EPISODES:
{episodes_text}

{f"CONTEXT: {context}" if context else ""}

Generate 2-3 actionable insights about patterns in issues, what resolutions worked, and how to serve this customer better.

Insights:"""
        insight = _reflect(prompt, self.model)
        self.store(
            content=insight,
            metadata={"type": "reflection", "episode_count": len(episodes)},
            importance=0.9
        )
        return insight

    def store(self, content: str, metadata: dict = None, importance: float = 1.0) -> None:
        entry = MemoryEntry(content=content, metadata=metadata or {}, importance=importance)
        self._reflections.append(entry)

    def retrieve(self, query: str = None, top_k: int = 3) -> list[MemoryEntry]:
        return self._reflections[-top_k:]

    def clear(self) -> None:
        self._reflections = []

    def summary(self) -> dict:
        return {
            "type": "reflection",
            "reflection_count": len(self._reflections),
        }
