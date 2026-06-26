import pytest
import anthropic
from unittest.mock import patch, MagicMock
from min_agent.agent import run_agent


def make_end_turn_response(text="任务完成"):
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response.content = [text_block]
    return response


def make_tool_use_response(tool_name, tool_args, tool_id="mock_id"):
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_args
    tool_block.id = tool_id
    response.content = [tool_block]
    return response


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_end_turn_immediately(mock_client, mock_trace):
    mock_client.messages.create.return_value = make_end_turn_response("完成了！")
    result = run_agent("测试任务")
    assert result["success"] is True
    assert "完成了！" in result["result"]
    assert result["steps"] == 1


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_tool_then_end(mock_client, mock_trace):
    mock_client.messages.create.side_effect = [
        make_tool_use_response("search_notes", {"query": "agent"}),
        make_end_turn_response("找到笔记并完成"),
    ]
    result = run_agent("搜索 agent 笔记")
    assert result["success"] is True
    assert result["steps"] == 2


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_exceeds_max_steps(mock_client, mock_trace):
    mock_client.messages.create.return_value = make_tool_use_response(
        "search_notes", {"query": "loop"}
    )
    result = run_agent("无限循环任务")
    assert result["success"] is False
    assert "最大步数" in result["error"]
    assert result["steps"] == 5


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_api_error_no_crash(mock_client, mock_trace):
    mock_client.messages.create.side_effect = anthropic.APIStatusError(
        "Internal Server Error",
        response=MagicMock(status_code=500, headers={}),
        body={},
    )
    result = run_agent("会出错的任务")
    assert result["success"] is False
    assert "API 错误" in result["error"]
    assert result["steps"] == 0


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_unexpected_error_no_crash(mock_client, mock_trace):
    mock_client.messages.create.side_effect = RuntimeError("something unexpected")
    result = run_agent("另一个任务")
    assert result["success"] is False
    assert "未知错误" in result["error"]


@patch("min_agent.agent.write_trace")
@patch("min_agent.agent.client")
def test_agent_trace_written_on_tool_use(mock_client, mock_trace):
    mock_client.messages.create.side_effect = [
        make_tool_use_response("search_notes", {"query": "agent"}),
        make_end_turn_response("完成"),
    ]
    run_agent("搜索任务")
    assert mock_trace.call_count >= 1
