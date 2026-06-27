import streamlit as st
import json
import re
import os
import sys

# Fix path - go up one level from ui/ to project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from agent.support_agent import CustomerSupportAgent

st.set_page_config(
    page_title="Agent Memory OS",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .memory-box {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        border-left: 4px solid;
    }
    .short-term { border-color: #00d4ff; }
    .long-term  { border-color: #7c3aed; }
    .episodic   { border-color: #059669; }
    .reflection { border-color: #d97706; }
    .memory-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .memory-content {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }
    .metric-row {
        display: flex;
        gap: 8px;
        margin-top: 6px;
    }
    .metric-chip {
        background: #2d3748;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: #cbd5e0;
    }
    .chat-user {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-left: 20%;
        color: #e2e8f0;
        font-size: 14px;
    }
    .chat-agent {
        background: #1e2130;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-right: 20%;
        color: #e2e8f0;
        font-size: 14px;
        border-left: 3px solid #7c3aed;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-active { background: #065f46; color: #6ee7b7; }
    .badge-empty  { background: #1f2937; color: #6b7280; }
    div[data-testid="stSidebar"] { background-color: #0d1117; }
    .stTextInput input {
        background-color: #1e2130;
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def sanitize_id(customer_id: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', customer_id)
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "c_" + sanitized
    if len(sanitized) < 3:
        sanitized = sanitized + "_id"
    return sanitized[:512]


def get_agent(customer_id: str) -> CustomerSupportAgent:
    key = f"agent_{customer_id}"
    if key not in st.session_state:
        os.makedirs("./data", exist_ok=True)
        agent = CustomerSupportAgent(customer_id=customer_id)
        agent.remember_fact(f"Customer {customer_id} is on the Pro plan", importance=0.9)
        agent.remember_fact(f"Customer {customer_id} prefers concise responses", importance=0.7)
        st.session_state[key] = agent
    return st.session_state[key]


def render_memory_panel(agent: CustomerSupportAgent):
    status = agent.memory_status()

    st.markdown("## 🧠 Live Memory")
    st.markdown("---")

    st_data = status["short_term"]
    badge = "badge-active" if st_data["message_count"] > 0 else "badge-empty"
    st.markdown(f"""
    <div class="memory-box short-term">
        <div class="memory-title" style="color:#00d4ff">
            ⚡ Short-Term Memory
            <span class="status-badge {badge}" style="float:right">
                {st_data['utilization']}
            </span>
        </div>
        <div class="memory-content">
            Sliding context window — last {st_data['max_messages']} messages
        </div>
        <div class="metric-row">
            <span class="metric-chip">~{st_data['estimated_tokens']} tokens</span>
            <span class="metric-chip">{st_data['message_count']} messages</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lt_data = status["long_term"]
    badge = "badge-active" if lt_data["document_count"] > 0 else "badge-empty"
    lt_memories = agent.long_term.retrieve(query="customer preferences", top_k=3)
    lt_preview = "<br>".join([f"• {m.content[:80]}..." if len(m.content) > 80 else f"• {m.content}" for m in lt_memories]) if lt_memories else "No facts stored yet"
    st.markdown(f"""
    <div class="memory-box long-term">
        <div class="memory-title" style="color:#7c3aed">
            💾 Long-Term Memory
            <span class="status-badge {badge}" style="float:right">
                {lt_data['document_count']} facts
            </span>
        </div>
        <div class="memory-content">{lt_preview}</div>
    </div>
    """, unsafe_allow_html=True)

    ep_data = status["episodic"]
    badge = "badge-active" if ep_data["episode_count"] > 0 else "badge-empty"
    episodes = agent.episodic.retrieve_for_customer(agent.customer_id, top_k=1)
    ep_preview = episodes[0].content[:120] + "..." if episodes else "No past sessions yet"
    st.markdown(f"""
    <div class="memory-box episodic">
        <div class="memory-title" style="color:#059669">
            📖 Episodic Memory
            <span class="status-badge {badge}" style="float:right">
                {ep_data['episode_count']} episodes
            </span>
        </div>
        <div class="memory-content">{ep_preview}</div>
    </div>
    """, unsafe_allow_html=True)

    ref_data = status["reflection"]
    badge = "badge-active" if ref_data["reflection_count"] > 0 else "badge-empty"
    reflections = agent.reflection.retrieve(top_k=1)
    ref_preview = reflections[0].content[:120] + "..." if reflections else f"Triggers after turn {agent.REFLECTION_INTERVAL}"
    st.markdown(f"""
    <div class="memory-box reflection">
        <div class="memory-title" style="color:#d97706">
            🔮 Reflection Memory
            <span class="status-badge {badge}" style="float:right">
                {ref_data['reflection_count']} insights
            </span>
        </div>
        <div class="memory-content">{ref_preview}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Turn:** {status['turn_count']} &nbsp;|&nbsp; **Model:** `{status['model']}`")


# ── MAIN UI ──────────────────────────────────────────────────────────────────

st.markdown("# 🧠 Agent Memory OS")
st.markdown("*Customer Support Agent with Persistent Memory*")
st.markdown("---")

with st.sidebar:
    st.markdown("### 👤 Customer Session")
    raw_id = st.text_input("Customer ID", value="demo_user", key="customer_input")
    customer_id = sanitize_id(raw_id)

    if st.button("🔄 New Session", use_container_width=True):
        key = f"agent_{customer_id}"
        if key in st.session_state:
            del st.session_state[key]
        if "messages" in st.session_state:
            del st.session_state["messages"]
        st.rerun()

    if st.button("💾 End & Save Episode", use_container_width=True):
        agent = get_agent(customer_id)
        if agent.short_term.summary()["message_count"] > 0:
            with st.spinner("Summarizing conversation..."):
                summary = agent.end_conversation()
            st.success("Episode saved!")
            st.text_area("Summary", summary, height=120)
        else:
            st.warning("No messages to save.")

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
- **⚡ Short-term** — current conversation
- **💾 Long-term** — facts about the customer
- **📖 Episodic** — past session summaries
- **🔮 Reflection** — patterns and insights
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = get_agent(customer_id)

if len(st.session_state.messages) == 0:
    greeting_context = agent.get_greeting_context()
    if greeting_context:
        with st.spinner("Loading memory..."):
            opening = agent.chat(
                f"The customer just reconnected. Greet them warmly and briefly reference their last issue. Context: {greeting_context}"
            )
        st.session_state.messages.append({"role": "agent", "content": opening})

chat_col, memory_col = st.columns([2, 1])

with memory_col:
    render_memory_panel(agent)

with chat_col:
    st.markdown("### 💬 Conversation")

    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                '<div class="memory-content" style="text-align:center;padding:40px;color:#4a5568">'
                '👋 Start a conversation below'
                '</div>',
                unsafe_allow_html=True
            )
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-agent">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Type your message...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Agent thinking..."):
            response = agent.chat(user_input)
        st.session_state.messages.append({"role": "agent", "content": response})
        st.rerun()
