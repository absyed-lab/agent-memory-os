"""Tests for the sliding-window short-term memory."""

from memory.short_term import ShortTermMemory


def fill(memory, count):
    for i in range(count):
        memory.store(f"message {i}", metadata={"role": "user"})


def test_keeps_only_the_newest_messages():
    memory = ShortTermMemory(max_messages=3)
    fill(memory, 5)
    assert [e.content for e in memory.retrieve()] == [
        "message 2",
        "message 3",
        "message 4",
    ]


def test_default_window_is_twenty_messages():
    memory = ShortTermMemory()
    fill(memory, 25)
    assert len(memory.retrieve()) == 20
    assert memory.retrieve()[0].content == "message 5"


def test_retrieve_top_k_returns_most_recent_in_order():
    memory = ShortTermMemory(max_messages=5)
    fill(memory, 5)
    assert [e.content for e in memory.retrieve(top_k=2)] == ["message 3", "message 4"]


def test_context_string_labels_roles():
    memory = ShortTermMemory()
    memory.store("hello", metadata={"role": "user"})
    memory.store("hi there", metadata={"role": "assistant"})
    memory.store("no role")
    assert memory.get_context_string() == (
        "USER: hello\nASSISTANT: hi there\nUNKNOWN: no role"
    )


def test_summary_reports_usage():
    memory = ShortTermMemory(max_messages=4)
    memory.store("x" * 8)
    memory.store("y" * 8)
    assert memory.summary() == {
        "type": "short_term",
        "message_count": 2,
        "max_messages": 4,
        "estimated_tokens": 4,
        "utilization": "2/4",
    }


def test_clear_empties_the_window():
    memory = ShortTermMemory()
    fill(memory, 3)
    memory.clear()
    assert memory.retrieve() == []
    assert memory.get_context_string() == ""
