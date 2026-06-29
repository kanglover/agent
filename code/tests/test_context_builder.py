# code/tests/test_context_builder.py
import pytest
from context_builder.compressor import Compressor


def test_short_content_returned_unchanged():
    c = Compressor(max_chars=100)
    summary, ref_id, char_count = c.compress("short")
    assert summary == "short"
    assert ref_id == ""
    assert char_count == 5


def test_long_content_is_truncated():
    c = Compressor(max_chars=10)
    summary, ref_id, char_count = c.compress("x" * 50)
    assert summary.startswith("x" * 10)
    assert "截断" in summary
    assert ref_id.startswith("ref:")
    assert char_count == 50


def test_ref_id_is_deterministic_for_same_content():
    content = "hello world this is long enough"
    c1 = Compressor(max_chars=5)
    c2 = Compressor(max_chars=5)
    _, ref1, _ = c1.compress(content)
    _, ref2, _ = c2.compress(content)
    assert ref1 == ref2


def test_retrieve_returns_original():
    c = Compressor(max_chars=5)
    original = "hello world this is a long string"
    _, ref_id, _ = c.compress(original)
    assert c.retrieve(ref_id) == original


def test_retrieve_missing_returns_none():
    c = Compressor()
    assert c.retrieve("ref:nonexistent") is None


def test_is_compressed_true_for_long():
    c = Compressor(max_chars=5)
    _, ref_id, _ = c.compress("hello world")
    assert c.is_compressed(ref_id) is True


def test_is_compressed_false_for_short():
    c = Compressor(max_chars=100)
    _, ref_id, _ = c.compress("short")
    assert c.is_compressed(ref_id) is False


# ── ContextBuilder ────────────────────────────────────────────

from context_builder.builder import ContextBuilder
from context_builder.layers import ContextBundle


def _make_builder() -> ContextBuilder:
    return ContextBuilder(Compressor(max_chars=50))


def test_builder_requires_system_before_build():
    b = _make_builder()
    b.set_task("task", "goal")
    with pytest.raises(ValueError, match="system"):
        b.build()


def test_builder_requires_task_before_build():
    b = _make_builder()
    b.set_system("assistant", ["be helpful"])
    with pytest.raises(ValueError, match="task"):
        b.build()


def test_builder_returns_context_bundle():
    bundle = (
        _make_builder()
        .set_system("assistant", ["be helpful", "be concise"], ["search_notes"])
        .set_task("搜索笔记", "找到 agent 相关内容")
        .build()
    )
    assert isinstance(bundle, ContextBundle)
    assert bundle.system.role == "assistant"
    assert len(bundle.system.instructions) == 2
    assert bundle.task.goal == "找到 agent 相关内容"


def test_builder_adds_multiple_memory_items():
    bundle = (
        _make_builder()
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_memory("user_lang", "中文")
        .add_memory("last_topic", "agent")
        .build()
    )
    assert len(bundle.memory.items) == 2
    assert bundle.memory.items[0].key == "user_lang"
    assert bundle.memory.items[1].key == "last_topic"


def test_builder_compresses_long_evidence():
    bundle = (
        ContextBuilder(Compressor(max_chars=20))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_evidence("source.txt", "x" * 100)
        .build()
    )
    ev = bundle.evidence.items[0]
    assert ev.char_count == 100
    assert ev.ref_id.startswith("ref:")
    assert "截断" in ev.summary


def test_builder_keeps_short_evidence_uncompressed():
    bundle = (
        ContextBuilder(Compressor(max_chars=200))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_evidence("notes.txt", "short content")
        .build()
    )
    ev = bundle.evidence.items[0]
    assert ev.ref_id == ""
    assert ev.summary == "short content"


def test_builder_compresses_long_trace_result():
    bundle = (
        ContextBuilder(Compressor(max_chars=20))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_trace_step(1, "search_notes", {"query": "agent"}, "x" * 100)
        .build()
    )
    step = bundle.trace.steps[0]
    assert step.result_ref.startswith("ref:")
    assert step.result_char_count == 100


def test_builder_keeps_short_trace_result_uncompressed():
    bundle = (
        ContextBuilder(Compressor(max_chars=200))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_trace_step(1, "search_notes", {"query": "agent"}, "short result")
        .build()
    )
    step = bundle.trace.steps[0]
    assert step.result_ref == ""
    assert step.result_summary == "short result"


def test_builder_is_chainable():
    b = _make_builder()
    result = b.set_system("a", ["b"])
    assert result is b
    result2 = b.set_task("t", "g")
    assert result2 is b
