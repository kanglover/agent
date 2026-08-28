"""LLM 接口层：定义响应数据类与脚本式 Mock LLM，用于演示 Agent 无需真实 API 的推理流程。"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class TokenUsage:
    """token 用量统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ToolCall:
    """单次工具调用描述。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 响应封装。"""
    content: str = ""
    tool_calls: Optional[list] = None   # list[ToolCall] 或 None
    finish_reason: str = "stop"         # "stop" | "tool_calls" | "length"
    usage: TokenUsage = field(default_factory=TokenUsage)


class MockLLM:
    """脚本式 Mock LLM：按预设脚本依次返回响应，用于演示无需真实 API。"""

    def __init__(self, script: Optional[list[dict]] = None):
        self._script: list[dict] = script if script is not None else default_script()
        self._step: int = 0

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, **kwargs) -> LLMResponse:
        # 1. 取脚本第 self._step 项（越界则返回空响应）
        if self._step >= len(self._script):
            self._step += 1
            return LLMResponse(
                content="",
                tool_calls=None,
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                    completion_tokens=0,
                ),
            )

        item = self._script[self._step]
        content: str = item.get("content", "")

        # 解析 tool_calls
        raw_tool_calls = item.get("tool_calls")
        tool_calls: Optional[list[ToolCall]] = None
        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
                for tc in raw_tool_calls
            ]

        finish_reason: str = item.get("finish_reason", "stop")

        # 2. 估算 token
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion_tokens = len(content) // 4 + 10 * len(tool_calls or [])

        # 3. 步数前进，返回响应
        self._step += 1
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    def reset(self):
        """重置步数。"""
        self._step = 0

    @property
    def step(self) -> int:
        return self._step


def default_script() -> list[dict]:
    """演示脚本：计算半径 5 的圆面积并搜索圆相关知识，最后总结。"""
    return [
        # 第 1 步: 调用 calculator 计算 "3.14159 * 5 ** 2"
        {
            "content": "我需要先计算半径为 5 的圆的面积，公式是 π * r²。",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "calculator",
                    "arguments": {"expression": "3.14159 * 5 ** 2"},
                }
            ],
            "finish_reason": "tool_calls",
        },
        # 第 2 步: 调用 web_search 查询 "circle area formula"
        {
            "content": "计算完成，接下来搜索圆面积公式的相关知识。",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "web_search",
                    "arguments": {"query": "circle area formula"},
                }
            ],
            "finish_reason": "tool_calls",
        },
        # 第 3 步: 纯文本总结
        {
            "content": (
                "半径为 5 的圆面积约为 78.54（3.14159 * 5² = 78.53975）。"
                "圆面积公式为 A = π * r²，其中 π ≈ 3.14159，r 为半径。"
                "搜索结果摘要：圆面积公式是几何学中最基本的公式之一，"
                "广泛应用于工程、物理和日常计算中。"
            ),
            "finish_reason": "stop",
        },
    ]
