import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from .base import BaseMemory, MemoryEntry

class LongTermMemory(BaseMemory):
    def __init__(self, collection_name: str = "long_term_memory", persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, content: str, metadata: dict = None, importance: float = 1.0) -> None:
        meta = metadata or {}
        meta["importance"] = importance
        meta["timestamp"] = datetime.utcnow().isoformat()
        doc_id = f"mem_{datetime.utcnow().timestamp()}_{hash(content) % 100000}"
        self.collection.add(documents=[content], metadatas=[meta], ids=[doc_id])

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )
        entries = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            entries.append(MemoryEntry(
                content=doc,
                metadata=meta,
                importance=meta.get("importance", 1.0),
                timestamp=datetime.fromisoformat(meta.get("timestamp", datetime.utcnow().isoformat()))
            ))
        return entries

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.ef
        )

    def summary(self) -> dict:
        return {
            "type": "long_term",
            "document_count": self.collection.count(),
            "collection_name": self.collection.name,
        }
