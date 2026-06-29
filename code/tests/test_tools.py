# code/tests/test_tools.py
import json
import pytest
from unittest.mock import patch
from min_agent.tools import search_notes, write_summary, run_tool, TOOL_DEFINITIONS


# ─── search_notes: 基础功能 ───────────────────────────────────

def test_search_found_contains_content():
    result = search_notes("agent")
    assert "Agent" in result


def test_search_not_found_returns_error_json():
    result = search_notes("xxxxnonexistent")
    data = json.loads(result)
    assert data["error"] is True
    assert data["code"] == "NOT_FOUND"


def test_search_not_found_next_action_lists_topics():
    result = search_notes("xxxxnonexistent")
    data = json.loads(result)
    assert "next_action" in data
    assert "agent" in data["next_action"] or "workflow" in data["next_action"]


def test_search_empty_query_returns_error_json():
    result = search_notes("")
    data = json.loads(result)
    assert data["error"] is True
    assert data["code"] == "INVALID_QUERY"
    assert "next_action" in data


# ─── search_notes: 分页 ───────────────────────────────────────

def test_search_max_results_limits_output():
    """注入多条可命中的记录，验证 max_results=1 只返回 1 条"""
    from min_agent import tools as t
    backup = dict(t.NOTES_DB)
    t.NOTES_DB.clear()
    t.NOTES_DB.update({
        "alpha one": "内容 A",
        "alpha two": "内容 B",
        "alpha three": "内容 C",
    })
    try:
        result = search_notes("alpha", max_results=1)
        lines = [l for l in result.strip().split("\n") if l.startswith("[")]
        assert len(lines) == 1
    finally:
        t.NOTES_DB.clear()
        t.NOTES_DB.update(backup)


def test_search_truncation_hint_when_more_results():
    """结果超出 max_results 时，提示还有几条"""
    from min_agent import tools as t
    backup = dict(t.NOTES_DB)
    t.NOTES_DB.clear()
    t.NOTES_DB.update({
        "beta one": "内容 1",
        "beta two": "内容 2",
        "beta three": "内容 3",
    })
    try:
        result = search_notes("beta", max_results=1)
        assert "还有" in result or "更多" in result
    finally:
        t.NOTES_DB.clear()
        t.NOTES_DB.update(backup)


# ─── search_notes: 内容截断 ───────────────────────────────────

def test_search_long_content_is_truncated():
    """单条笔记内容超过 MAX_CHARS_PER_RESULT 时自动截断"""
    from min_agent import tools as t
    backup = dict(t.NOTES_DB)
    t.NOTES_DB["longtest"] = "x" * 300
    try:
        result = search_notes("longtest")
        assert "…（内容已截断）" in result
    finally:
        t.NOTES_DB.clear()
        t.NOTES_DB.update(backup)


# ─── search_notes: detailed 格式 ──────────────────────────────

def test_search_detailed_format_returns_json():
    result = search_notes("agent", format="detailed")
    data = json.loads(result)
    assert "total_found" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_detailed_has_returned_count():
    result = search_notes("agent", format="detailed")
    data = json.loads(result)
    assert "returned" in data


# ─── write_summary: 基础功能 ──────────────────────────────────

def test_write_summary_returns_json_success():
    result = write_summary("这是一段测试摘要内容")
    data = json.loads(result)
    assert data["success"] is True
    assert data["chars_saved"] > 0
    assert "preview" in data


def test_write_summary_empty_text_returns_error():
    result = write_summary("")
    data = json.loads(result)
    assert data["error"] is True
    assert data["code"] == "INVALID_INPUT"
    assert "next_action" in data


# ─── write_summary: 截断 ──────────────────────────────────────

def test_write_summary_truncates_long_text():
    result = write_summary("x" * 600, max_length=500)
    data = json.loads(result)
    assert data["success"] is True
    assert data["chars_saved"] == 500
    assert "warning" in data


# ─── requires_confirmation ────────────────────────────────────

def test_write_summary_response_has_requires_confirmation():
    result = write_summary("测试摘要内容")
    data = json.loads(result)
    assert "requires_confirmation" in data


def test_write_summary_marked_in_definition():
    write_def = next(t for t in TOOL_DEFINITIONS if t["name"] == "write_summary")
    has_mark = (
        "requires_confirmation" in write_def.get("description", "").lower()
        or write_def.get("x_requires_confirmation", False) is True
    )
    assert has_mark


def test_search_notes_not_requires_confirmation():
    search_def = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_notes")
    assert search_def.get("x_requires_confirmation", False) is False


# ─── _run_with_retry ──────────────────────────────────────────

def test_run_with_retry_succeeds_after_one_failure():
    from min_agent.tools import _run_with_retry
    call_count = [0]

    def flaky(**kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise RuntimeError("临时错误")
        return "success"

    result = _run_with_retry(flaky, {}, retries=2)
    assert result == "success"
    assert call_count[0] == 2


def test_run_with_retry_exhausted_returns_error_json():
    from min_agent.tools import _run_with_retry

    def always_fail(**kwargs):
        raise RuntimeError("总是失败")

    result = _run_with_retry(always_fail, {}, retries=2)
    data = json.loads(result)
    assert data["error"] is True
    assert data["code"] == "RETRY_EXHAUSTED"
    assert "next_action" in data


# ─── run_tool: 分发 ───────────────────────────────────────────

def test_run_tool_dispatches_search():
    result = run_tool("search_notes", {"query": "workflow"})
    assert len(result) > 0


def test_run_tool_dispatches_write():
    result = run_tool("write_summary", {"text": "摘要内容"})
    data = json.loads(result)
    assert data["success"] is True


def test_run_tool_unknown_returns_error_json():
    result = run_tool("nonexistent_tool", {})
    data = json.loads(result)
    assert data["error"] is True
    assert data["code"] == "UNKNOWN_TOOL"
    assert "next_action" in data


# ─── TOOL_DEFINITIONS 格式 ────────────────────────────────────

def test_tool_definitions_have_descriptions():
    for t in TOOL_DEFINITIONS:
        assert len(t.get("description", "")) > 50, f"{t['name']} description 太短"


def test_tool_definitions_have_examples_in_description():
    for t in TOOL_DEFINITIONS:
        desc = t.get("description", "")
        assert "示例" in desc or "example" in desc.lower(), \
            f"{t['name']} description 缺少示例"


def test_search_notes_definition_has_format_param():
    search_def = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_notes")
    props = search_def["input_schema"]["properties"]
    assert "format" in props
    assert "max_results" in props


def test_write_summary_definition_has_max_length_param():
    write_def = next(t for t in TOOL_DEFINITIONS if t["name"] == "write_summary")
    props = write_def["input_schema"]["properties"]
    assert "max_length" in props
