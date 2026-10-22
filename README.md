# Agent Memory OS

A customer-support agent that keeps memory across sessions. When a customer comes back, the agent can recall what happened last time instead of starting from zero.

**Live demo:** https://agent-memory-os-i5xjsxgvbxhtcgdzgavpsr.streamlit.app (free Streamlit hosting: if the app is asleep, click the wake-up button and wait about a minute).

## Why

Most chat agents lose everything when a session ends, so returning customers have to explain their problem again. This project is a small, readable experiment in giving an agent separate memory layers and combining them into one prompt.

## Memory layers

| Layer | What it keeps | How |
|---|---|---|
| Short-term | the current conversation | sliding window of the last 20 messages |
| Long-term | facts about the customer (plan, preferences) | ChromaDB vector store, retrieved by similarity |
| Episodic | summaries of past conversations | an LLM summarises each session when it ends |
| Reflection | patterns across conversations | an LLM writes insights every 5 turns (kept in memory only) |

`agent/support_agent.py` pulls context from all four layers into the system prompt for each reply. Every reply's latency and token count is logged to `data/metrics.json` and shown on the Metrics page of the Streamlit app.

## Run it

```bash
git clone https://github.com/absyed-lab/agent-memory-os
cd agent-memory-os
pip install -r requirements.txt
```

**With Groq (cloud):** create a `.env` file containing `GROQ_API_KEY=<your key>`. The agent uses `llama-3.1-8b-instant`.

**Fully offline with Ollama:** leave `GROQ_API_KEY` unset and run `ollama pull llama3.1:8b`.

Then start either interface:

```bash
streamlit run ui/app.py   # web UI with memory panels and metrics page
python main.py            # terminal chat
```

On first run ChromaDB downloads a small embedding model, so you need internet access once.

## Project structure

```
memory/
  base.py          shared memory interface
  short_term.py    sliding context window
  long_term.py     ChromaDB vector store
  episodic.py      conversation summaries
  reflection.py    insights from past episodes
agent/
  support_agent.py combines the memory layers and calls the LLM
ui/
  app.py           Streamlit chat with memory panels
  pages/metrics.py latency and token charts
main.py            terminal interface
```

## Results

No systematic evaluation yet. The metrics page shows latency and tokens per reply, not answer quality.

## Limitations

- Demo data only: the customer facts (such as "Pro plan") are seeded by the app, not taken from a real support system.
- With Ollama, token counts are estimated as characters / 4, not counted by the model.
- Memories are not pruned or merged automatically, so stored memory grows with every session.
- Reflections are not saved to disk, so they are lost when the app restarts.
- No test suite yet.

## Author

Syed Abdul Basit, M.Sc. AI Engineering (thesis track), Okan University, Istanbul.
GitHub: [absyed-lab](https://github.com/absyed-lab) · LinkedIn: [syedabdulbasit](https://www.linkedin.com/in/syedabdulbasit)

License: MIT
