import os
import json
import time
import datetime
from dotenv import load_dotenv
load_dotenv()

def get_groq_key() -> str:
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY = get_groq_key()
USE_GROQ = bool(GROQ_API_KEY)

if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)

from memory import ShortTermMemory, LongTermMemory, EpisodicMemory, ReflectionMemory

DEFAULT_MODEL_GROQ = "llama-3.1-8b-instant"
DEFAULT_MODEL_OLLAMA = "llama3.1:8b"

SYSTEM_PROMPT = """You are a customer support agent for a software company.
You have access to the customer history and previous interactions.
Always be helpful and concise. Reference past context when relevant.
Never say based on my memory, just use the information naturally."""

METRICS_PATH = "./data/metrics.json"

def load_metrics() -> list:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return []

def save_metric(entry: dict) -> None:
    metrics = load_metrics()
    metrics.append(entry)
    os.makedirs("./data", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

def chat_with_llm(messages: list, system: str) -> tuple:
    start = time.time()
    if USE_GROQ:
        full_messages = [{"role": "system", "content": system}] + messages
        response = groq_client.chat.completions.create(
            model=DEFAULT_MODEL_GROQ,
            messages=full_messages,
            max_tokens=1000
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
    else:
        import ollama
        response = ollama.chat(
            model=DEFAULT_MODEL_OLLAMA,
            messages=[{"role": "system", "content": system}] + messages
        )
        text = response['message']['content']
        tokens = len(text) // 4
    latency = round(time.time() - start, 2)
    return text, latency, tokens


class CustomerSupportAgent:
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.model = DEFAULT_MODEL_GROQ if USE_GROQ else DEFAULT_MODEL_OLLAMA
        self.backend = "groq" if USE_GROQ else "ollama"
        self.short_term = ShortTermMemory(max_messages=20)
        self.long_term = LongTermMemory(collection_name=f"customer_{customer_id}")
        self.episodic = EpisodicMemory(
            storage_path=f"./data/episodes_{customer_id}.json",
            model=self.model
        )
        self.reflection = ReflectionMemory(model=self.model)
        self.turn_count = 0
        self.REFLECTION_INTERVAL = 5

    def _build_memory_context(self, user_message: str) -> str:
        context_parts = []
        lt_memories = self.long_term.retrieve(query=user_message, top_k=3)
        if lt_memories:
            facts = "\n".join([f"- {m.content}" for m in lt_memories])
            context_parts.append(f"KNOWN FACTS ABOUT THIS CUSTOMER:\n{facts}")
        past_episodes = self.episodic.retrieve_for_customer(self.customer_id, top_k=2)
        if past_episodes:
            episodes = "\n".join([f"- {e.content}" for e in past_episodes])
            context_parts.append(f"PAST CONVERSATIONS:\n{episodes}")
        reflections = self.reflection.retrieve(top_k=1)
        if reflections:
            context_parts.append(f"AGENT INSIGHT:\n{reflections[0].content}")
        return "\n\n".join(context_parts)

    def get_greeting_context(self) -> str:
        past_episodes = self.episodic.retrieve_for_customer(self.customer_id, top_k=2)
        if not past_episodes:
            return ""
        episodes = "\n".join([f"- {e.content}" for e in past_episodes])
        return f"PAST CONVERSATIONS WITH THIS CUSTOMER:\n{episodes}"

    def chat(self, user_message: str) -> str:
        self.turn_count += 1
        self.short_term.store(user_message, metadata={"role": "user"})
        memory_context = self._build_memory_context(user_message)
        system = SYSTEM_PROMPT
        if memory_context:
            system += f"\n\n---\nMEMORY CONTEXT:\n{memory_context}\n---"
        recent_messages = self.short_term.retrieve()
        api_messages = []
        for entry in recent_messages[:-1]:
            role = entry.metadata.get("role", "user")
            api_messages.append({"role": role, "content": entry.content})
        api_messages.append({"role": "user", "content": user_message})

        assistant_message, latency, tokens = chat_with_llm(api_messages, system)

        save_metric({
            "customer_id": self.customer_id,
            "turn": self.turn_count,
            "latency_seconds": latency,
            "tokens_used": tokens,
            "memory_facts": self.long_term.summary()["document_count"],
            "episodes": self.episodic.summary()["episode_count"],
            "backend": self.backend,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        self.short_term.store(assistant_message, metadata={"role": "assistant"})
        if self.turn_count % self.REFLECTION_INTERVAL == 0:
            self._run_reflection()
        return assistant_message

    def remember_fact(self, fact: str, importance: float = 0.8) -> None:
        self.long_term.store(
            content=fact,
            metadata={"customer_id": self.customer_id},
            importance=importance
        )

    def end_conversation(self) -> str:
        messages = [
            {"role": e.metadata.get("role", "user"), "content": e.content}
            for e in self.short_term.retrieve()
        ]
        summary = self.episodic.store_conversation(messages, self.customer_id)
        self.short_term.clear()
        return summary

    def _run_reflection(self) -> None:
        past_episodes = self.episodic.retrieve_for_customer(self.customer_id, top_k=5)
        if len(past_episodes) >= 2:
            episode_texts = [e.content for e in past_episodes]
            self.reflection.generate_reflection(
                episodes=episode_texts,
                context=f"Customer ID: {self.customer_id}"
            )

    def memory_status(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "model": self.model,
            "backend": self.backend,
            "short_term": self.short_term.summary(),
            "long_term": self.long_term.summary(),
            "episodic": self.episodic.summary(),
            "reflection": self.reflection.summary(),
            "turn_count": self.turn_count
        }