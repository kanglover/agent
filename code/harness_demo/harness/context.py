"""上下文管理层：维护对话历史、token 估算与上下文窗口超限时的摘要裁剪，支持序列化往返。"""

import json
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class Message:
    """对话消息，兼容 system/user/assistant/tool 多种角色。"""
    role: str                              # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: Optional[list] = None       # list of ToolCall-like objects (有 id/name/arguments 属性即可)
    tool_call_id: Optional[str] = None     # role="tool" 时对应的调用 id
    name: Optional[str] = None             # role="tool" 时的工具名

    def to_dict(self) -> dict:
        """转成 API 格式 dict。"""
        d: dict = {"role": self.role, "content": self.content}

        # assistant 且有 tool_calls 时加 "tool_calls" 字段
        if self.role == "assistant" and self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]

        # tool 消息带 tool_call_id 和 name
        if self.role == "tool":
            d["tool_call_id"] = self.tool_call_id
            d["name"] = self.name

        return d

    def to_serializable(self) -> dict:
        """转为可 JSON 序列化的结构（用于 to_dict/from_dict 往返）。"""
        d: dict = {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        else:
            d["tool_calls"] = None
        return d

    @classmethod
    def from_serializable(cls, d: dict) -> "Message":
        """从可序列化结构恢复 Message。"""
        tool_calls = None
        raw = d.get("tool_calls")
        if raw:
            # 用一个轻量对象承载 id/name/arguments 属性
            tool_calls = [
                _ToolCallSnapshot(tc["id"], tc["name"], tc["arguments"])
                for tc in raw
            ]
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class _ToolCallSnapshot:
    """轻量 ToolCall 快照，仅提供 id/name/arguments 属性用于序列化往返。"""
    id: str
    name: str
    arguments: dict


class ContextManager:
    """管理对话历史、token 计数、上下文窗口超限时的裁剪/摘要。"""

    def __init__(self, system_prompt: str = "", max_tokens: int = 4096):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[Message] = []
        self.summarized_count: int = 0   # 已被摘要压缩掉的历史消息数

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str = "", tool_calls: Optional[list] = None) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_message(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(
            Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)
        )

    def get_messages(self) -> list[dict]:
        """返回 [{"role": "system", "content": system_prompt}] + 消息列表。"""
        result: list[dict] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(m.to_dict() for m in self.messages)
        return result

    def get_token_count(self) -> int:
        """粗略估算: system_prompt + 所有消息内容，字符数 // 4。"""
        total = len(self.system_prompt)
        for m in self.messages:
            total += len(m.content)
            if m.tool_calls:
                for tc in m.tool_calls:
                    total += len(tc.name) + len(str(getattr(tc, "arguments", "")))
        return total // 4

    def should_summarize(self) -> bool:
        """get_token_count() > max_tokens * 0.8 时返回 True。"""
        return self.get_token_count() > self.max_tokens * 0.8

    def summarize(self) -> None:
        """超限时保留最近 4 条消息，更早的压缩成一条 system 摘要消息。"""
        keep_count = 4
        if len(self.messages) <= keep_count:
            return

        # 被压缩的消息
        old_messages = self.messages[:-keep_count]
        compressed_count = len(old_messages)
        self.summarized_count += compressed_count

        # 保留最近 keep_count 条
        self.messages = self.messages[-keep_count:]

        # 在开头插入摘要 system 消息
        summary_msg = Message(
            role="system",
            content=f"[context summary] {self.summarized_count} earlier messages compressed...",
        )
        self.messages.insert(0, summary_msg)

    def clear(self) -> None:
        """清空所有消息和摘要计数。"""
        self.messages = []
        self.summarized_count = 0

    def to_dict(self) -> dict:
        """序列化为可 JSON 结构。"""
        return {
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "messages": [m.to_serializable() for m in self.messages],
            "summarized_count": self.summarized_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextManager":
        """从序列化结构恢复。"""
        cm = cls(
            system_prompt=data.get("system_prompt", ""),
            max_tokens=data.get("max_tokens", 4096),
        )
        cm.messages = [Message.from_serializable(m) for m in data.get("messages", [])]
        cm.summarized_count = data.get("summarized_count", 0)
        return cm
