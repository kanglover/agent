import json
from pathlib import Path
import pytest
from min_agent.tracer import build_trace_entry, write_trace


def test_build_trace_entry_has_required_fields():
    entry = build_trace_entry(
        step=0,
        thought_summary="决定搜索 agent",
        tool="search_notes",
        args={"query": "agent"},
        observation="Agent 是能代替用户...",
        input_tokens=100,
        output_tokens=50,
    )
    assert entry["step"] == 0
    assert entry["thought_summary"] == "决定搜索 agent"
    assert entry["tool"] == "search_notes"
    assert entry["args"] == {"query": "agent"}
    assert entry["observation"] == "Agent 是能代替用户..."
    assert "cost_estimate" in entry
    assert "timestamp" in entry


def test_cost_estimate_is_positive():
    entry = build_trace_entry(0, "思考", "search_notes", {}, "结果", 100, 50)
    assert entry["cost_estimate"] > 0


def test_cost_estimate_formula():
    # claude-opus-4-8: 输入 $15/M，输出 $75/M
    # 1000 输入 + 1000 输出 = (1000*15 + 1000*75) / 1_000_000 = 0.09
    entry = build_trace_entry(0, "思考", None, None, "结果", 1000, 1000)
    assert abs(entry["cost_estimate"] - 0.09) < 0.001


def test_write_trace_appends_jsonl(tmp_path, monkeypatch):
    trace_file = tmp_path / "test_trace.jsonl"
    monkeypatch.setattr("min_agent.tracer.TRACE_FILE", trace_file)

    entry1 = build_trace_entry(0, "第一步", "search_notes", {"query": "x"}, "结果1", 100, 50)
    entry2 = build_trace_entry(1, "第二步", "write_summary", {"text": "y"}, "结果2", 200, 80)
    write_trace(entry1)
    write_trace(entry2)

    lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 0
    assert json.loads(lines[1])["step"] == 1


def test_write_trace_none_tool_allowed(tmp_path, monkeypatch):
    trace_file = tmp_path / "test_trace.jsonl"
    monkeypatch.setattr("min_agent.tracer.TRACE_FILE", trace_file)

    entry = build_trace_entry(0, "直接回复", None, None, "完成了", 50, 30)
    write_trace(entry)  # 不应该报错

    data = json.loads(trace_file.read_text(encoding="utf-8").strip())
    assert data["tool"] is None
