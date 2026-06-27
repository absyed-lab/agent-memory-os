from collections import deque
from .base import BaseMemory, MemoryEntry

class ShortTermMemory(BaseMemory):
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: deque[MemoryEntry] = deque(maxlen=max_messages)

    def store(self, content: str, metadata: dict = None, importance: float = 1.0) -> None:
        entry = MemoryEntry(content=content, metadata=metadata or {}, importance=importance)
        self._messages.append(entry)

    def retrieve(self, query: str = None, top_k: int = 20) -> list[MemoryEntry]:
        messages = list(self._messages)
        return messages[-top_k:] if top_k else messages

    def get_context_string(self) -> str:
        lines = []
        for entry in self._messages:
            role = entry.metadata.get("role", "unknown")
            lines.append(f"{role.upper()}: {entry.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()

    def summary(self) -> dict:
        total_chars = sum(len(e.content) for e in self._messages)
        return {
            "type": "short_term",
            "message_count": len(self._messages),
            "max_messages": self.max_messages,
            "estimated_tokens": total_chars // 4,
            "utilization": f"{len(self._messages)}/{self.max_messages}"
        }
