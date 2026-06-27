from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class MemoryEntry:
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    importance: float = 1.0

class BaseMemory(ABC):
    @abstractmethod
    def store(self, content: str, metadata: dict = None, importance: float = 1.0) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        pass
