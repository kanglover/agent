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


# ── Renderer + 集成测试 ───────────────────────────────────────

from context_builder.renderer import render


def _make_full_bundle():
    """标准测试 bundle，包含 5 层数据"""
    return (
        ContextBuilder(Compressor(max_chars=50))
        .set_system(
            role="你是一个 AI 助手",
            instructions=["用中文回复", "保持简洁"],
            tool_names=["search_notes", "write_summary"],
        )
        .set_task(
            description="搜索 agent 相关笔记",
            goal="生成一份摘要",
            constraints=["不超过 200 字"],
        )
        .add_memory("user_preference", "简洁")
        .add_evidence("notes/agent.md", "Agent 是能代替用户完成任务的系统。")
        .add_trace_step(1, "search_notes", {"query": "agent"}, "找到 1 条笔记")
        .build()
    )


def test_render_contains_all_five_sections():
    output = render(_make_full_bundle())
    assert "[SYSTEM]" in output
    assert "[TASK]" in output
    assert "[MEMORY]" in output
    assert "[RETRIEVED EVIDENCE]" in output
    assert "[RECENT TRACE]" in output


def test_render_section_order_is_fixed():
    output = render(_make_full_bundle())
    positions = {s: output.index(s) for s in [
        "[SYSTEM]", "[TASK]", "[MEMORY]",
        "[RETRIEVED EVIDENCE]", "[RECENT TRACE]"
    ]}
    assert positions["[SYSTEM]"] < positions["[TASK]"]
    assert positions["[TASK]"] < positions["[MEMORY]"]
    assert positions["[MEMORY]"] < positions["[RETRIEVED EVIDENCE]"]
    assert positions["[RETRIEVED EVIDENCE]"] < positions["[RECENT TRACE]"]


def test_render_system_contains_role_and_instructions():
    output = render(_make_full_bundle())
    assert "你是一个 AI 助手" in output
    assert "用中文回复" in output
    assert "保持简洁" in output


def test_render_task_contains_description_and_goal():
    output = render(_make_full_bundle())
    assert "搜索 agent 相关笔记" in output
    assert "生成一份摘要" in output


def test_render_memory_contains_items():
    output = render(_make_full_bundle())
    assert "user_preference" in output
    assert "简洁" in output


def test_render_evidence_contains_source_and_ref():
    bundle = (
        ContextBuilder(Compressor(max_chars=10))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_evidence("web.html", "这是一段超过十个字的长内容需要被压缩")
        .build()
    )
    output = render(bundle)
    assert "web.html" in output
    assert "ref:" in output


def test_render_trace_contains_tool_and_args():
    output = render(_make_full_bundle())
    assert "search_notes" in output
    assert "agent" in output


def test_render_omits_empty_memory():
    bundle = (
        ContextBuilder(Compressor())
        .set_system("a", ["b"])
        .set_task("t", "g")
        .build()
    )
    assert "[MEMORY]" not in render(bundle)


def test_render_omits_empty_evidence():
    bundle = (
        ContextBuilder(Compressor())
        .set_system("a", ["b"])
        .set_task("t", "g")
        .build()
    )
    assert "[RETRIEVED EVIDENCE]" not in render(bundle)


def test_render_omits_empty_trace():
    bundle = (
        ContextBuilder(Compressor())
        .set_system("a", ["b"])
        .set_task("t", "g")
        .build()
    )
    assert "[RECENT TRACE]" not in render(bundle)


# ── 验收条件 1：prompt 结构稳定 ──────────────────────────────

def test_render_is_deterministic_same_bundle():
    """同一个 bundle 对象，多次 render 输出完全相同"""
    bundle = _make_full_bundle()
    assert render(bundle) == render(bundle)


def test_render_stable_across_same_task():
    """同一任务的两次构建，输出结构完全相同"""
    def build():
        return (
            ContextBuilder(Compressor(max_chars=50))
            .set_system("assistant", ["be helpful"], ["search_notes"])
            .set_task("搜索 agent 笔记", "生成摘要")
            .add_memory("lang", "中文")
            .add_evidence("notes.md", "内容")
            .add_trace_step(1, "search_notes", {"query": "agent"}, "结果")
            .build()
        )
    assert render(build()) == render(build())


# ── 验收条件 2：不把完整内容塞进 prompt ─────────────────────

def test_render_excludes_full_long_content():
    """长内容被压缩后，完整原文不出现在 prompt 里"""
    long_content = "这是一段非常长的内容，需要被压缩，不能直接出现在 prompt 里。" * 20
    bundle = (
        ContextBuilder(Compressor(max_chars=50))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_evidence("web.html", long_content)
        .build()
    )
    output = render(bundle)
    assert long_content not in output
    assert "ref:" in output


def test_render_excludes_full_long_trace_result():
    """长工具结果被压缩后，完整原文不出现在 prompt 里"""
    long_result = "工具返回了很多内容：" + "详细日志行 " * 50
    bundle = (
        ContextBuilder(Compressor(max_chars=50))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_trace_step(1, "read_file", {"path": "big.log"}, long_result)
        .build()
    )
    output = render(bundle)
    assert long_result not in output
    assert "ref:" in output


def test_short_content_appears_directly_in_prompt():
    """短内容不压缩，直接出现在 prompt 里（无需查 ref）"""
    short = "Agent 是能代替用户完成任务的系统。"
    bundle = (
        ContextBuilder(Compressor(max_chars=500))
        .set_system("a", ["b"])
        .set_task("t", "g")
        .add_evidence("notes.md", short)
        .build()
    )
    output = render(bundle)
    assert short in output
    assert "ref:" not in output


def test_render_constraints_in_task_section():
    bundle = (
        ContextBuilder(Compressor())
        .set_system("a", ["b"])
        .set_task("描述", "目标", constraints=["约束1", "约束2"])
        .build()
    )
    output = render(bundle)
    assert "约束1" in output
    assert "约束2" in output


def test_render_tool_names_in_system_section():
    bundle = (
        ContextBuilder(Compressor())
        .set_system("a", ["b"], tool_names=["search_notes", "write_summary"])
        .set_task("t", "g")
        .build()
    )
    output = render(bundle)
    assert "search_notes" in output
    assert "write_summary" in output
