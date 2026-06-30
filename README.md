# Agent Memory OS

AI agents that remember everything — across sessions, across time.

Live Demo: https://agent-memory-os-i5xjsxgvbxhtcgdzgavpsr.streamlit.app

---

## The Problem

Every AI chatbot forgets everything the moment a conversation ends.
Customer explains their issue. Bot helps. Session ends. Next session: Bot has no idea who they are.

## The Solution

Agent Memory OS gives AI agents four layers of persistent memory.

Short-term: Current conversation context, sliding window of last 20 messages.
Long-term: Persistent facts about the customer stored in a vector database.
Episodic: Summaries of past conversations, recalled automatically next session.
Reflection: Patterns and insights generated from past episodes every 5 turns.

## What It Looks Like

First session:
Customer: My login is broken
Agent: I see you are on the Pro plan. Let me help reset your credentials.
Session ends.

Next session, days later:
Agent: Welcome back. I see we resolved a login issue last time. Is everything still working?

No ticket number. No explaining the issue again. Pure memory.

## Tech Stack

LLM: Groq API (cloud) or Ollama (local, fully offline)
Vector DB: ChromaDB
UI: Streamlit
Language: Python

## Run Locally

git clone https://github.com/absyed-lab/agent-memory-os
cd agent-memory-os
pip install -r requirements.txt
Add GROQ_API_KEY=your_key to .env file
streamlit run ui/app.py

## Run Offline with Ollama

ollama pull llama3.1:8b
Leave GROQ_API_KEY empty in .env
streamlit run ui/app.py

## Project Structure

agent-memory-os/
├── memory/
│   ├── base.py          Abstract memory interface
│   ├── short_term.py    Sliding context window
│   ├── long_term.py     ChromaDB vector store
│   ├── episodic.py      Conversation summaries
│   └── reflection.py    Pattern insights
├── agent/
│   └── support_agent.py Orchestrates all memory layers
├── ui/
│   └── app.py           Streamlit demo
└── main.py              Terminal interface

## Built By

Abdul Basit
https://github.com/absyed-lab