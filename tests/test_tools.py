import pytest
from min_agent.tools import search_notes, write_summary, run_tool

def test_search_notes_found():
    result = search_notes("agent")
    assert "Agent" in result

def test_search_notes_not_found():
    result = search_notes("xxxxnonexistent")
    assert "未找到" in result

def test_write_summary_returns_confirmation():
    result = write_summary("这是一段测试摘要内容")
    assert "已保存" in result
    assert "字" in result

def test_run_tool_dispatches_search():
    result = run_tool("search_notes", {"query": "workflow"})
    assert len(result) > 0

def test_run_tool_dispatches_write():
    result = run_tool("write_summary", {"text": "摘要内容"})
    assert "已保存" in result

def test_run_tool_raises_on_unknown():
    with pytest.raises(ValueError, match="未知工具"):
        run_tool("nonexistent_tool", {})
