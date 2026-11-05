"""Tests for episodic memory. The LLM call is stubbed, so no network is used."""

import json

import pytest

import memory.episodic as episodic
from memory.episodic import EpisodicMemory


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the LLM call and record the prompts it receives."""
    prompts = []

    def fake_summarize(text, model):
        prompts.append((text, model))
        return f"summary {len(prompts)}"

    monkeypatch.setattr(episodic, "_summarize", fake_summarize)
    return prompts


@pytest.fixture
def memory(tmp_path):
    return EpisodicMemory(storage_path=str(tmp_path / "episodes.json"), model="stub")


MESSAGES = [
    {"role": "user", "content": "My invoice is wrong"},
    {"role": "assistant", "content": "I have corrected it"},
]


def test_store_conversation_stores_the_llm_summary(memory, stub_llm):
    assert memory.store_conversation(MESSAGES, customer_id="c1") == "summary 1"

    entry = memory.retrieve()[0]
    assert entry.content == "summary 1"
    assert entry.importance == 0.8
    assert entry.metadata == {
        "customer_id": "c1",
        "type": "conversation_summary",
    }


def test_prompt_contains_the_conversation(memory, stub_llm):
    memory.store_conversation(MESSAGES, customer_id="c1")
    prompt, model = stub_llm[0]
    assert "USER: My invoice is wrong" in prompt
    assert "ASSISTANT: I have corrected it" in prompt
    assert model == "stub"


def test_retrieve_returns_newest_first(memory):
    for i in range(4):
        memory.store(f"episode {i}")
    assert [e.content for e in memory.retrieve(top_k=2)] == ["episode 3", "episode 2"]


def test_retrieve_is_empty_with_no_episodes(memory):
    assert memory.retrieve() == []


def test_retrieve_for_customer_filters_by_id(memory, stub_llm):
    memory.store_conversation(MESSAGES, customer_id="c1")
    memory.store_conversation(MESSAGES, customer_id="c2")
    memory.store_conversation(MESSAGES, customer_id="c1")

    found = memory.retrieve_for_customer("c1")
    assert [e.content for e in found] == ["summary 3", "summary 1"]
    assert memory.retrieve_for_customer("c3") == []


def test_episodes_survive_a_restart(memory, stub_llm):
    memory.store_conversation(MESSAGES, customer_id="c1")

    reloaded = EpisodicMemory(storage_path=memory.storage_path, model="stub")
    assert reloaded.retrieve_for_customer("c1")[0].content == "summary 1"


def test_clear_empties_the_file(memory):
    memory.store("episode")
    memory.clear()
    assert memory.summary()["episode_count"] == 0
    with open(memory.storage_path) as f:
        assert json.load(f) == []
