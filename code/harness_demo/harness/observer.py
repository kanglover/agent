"""评估观测模块：全链路 trace 记录、指标统计、运行摘要与导出。"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TraceEvent:
    timestamp: float
    event_type: str    # think | act | observe | error | warning | info
    step: int = 0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "step": self.step,
            "data": dict(self.data),
        }


class Observer:
    """评估观测: 全链路 trace 记录、指标统计、运行摘要。"""

    def __init__(self):
        self._trace: list[TraceEvent] = []
        self._metrics: dict = {
            "total_steps": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": None,
            "end_time": None,
        }
        self._hooks: list = []   # list[Callable[[TraceEvent], None]]

    def start(self) -> None:
        self._metrics["start_time"] = time.time()

    def stop(self) -> None:
        self._metrics["end_time"] = time.time()

    def log(self, event_type: str, step: int = 0, **data) -> None:
        event = TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            step=step,
            data=data,
        )
        self._trace.append(event)

        self._metrics["total_steps"] = max(self._metrics["total_steps"], step)
        if event_type == "think":
            self._metrics["llm_calls"] += 1
        elif event_type == "act":
            self._metrics["tool_calls"] += 1
            if data.get("success") is False:
                self._metrics["tool_errors"] += 1

        for hook in self._hooks:
            try:
                hook(event)
            except Exception:
                pass  # hook 异常不影响主流程

    def add_hook(self, hook: Callable) -> None:
        self._hooks.append(hook)

    def get_trace(self) -> list[TraceEvent]:
        return list(self._trace)

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def update_metrics(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self._metrics and isinstance(value, (int, float)) \
                    and isinstance(self._metrics[key], (int, float)) \
                    and self._metrics[key] is not None:
                self._metrics[key] += value
            else:
                self._metrics[key] = value

    def summary(self) -> str:
        end = self._metrics["end_time"] if self._metrics["end_time"] is not None else time.time()
        start = self._metrics["start_time"] if self._metrics["start_time"] is not None else end
        duration = end - start

        event_counts: dict = {}
        for event in self._trace:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        event_lines = "\n".join(
            f"  {etype}: {count}" for etype, count in event_counts.items()
        ) or "  (无事件)"

        lines = [
            "===== 运行摘要 =====",
            f"总步数: {self._metrics['total_steps']}",
            f"LLM 调用数: {self._metrics['llm_calls']}",
            f"工具调用数: {self._metrics['tool_calls']} (错误: {self._metrics['tool_errors']})",
            f"Token 消耗: prompt={self._metrics['prompt_tokens']}, "
            f"completion={self._metrics['completion_tokens']}, "
            f"total={self._metrics['prompt_tokens'] + self._metrics['completion_tokens']}",
            f"耗时: {duration:.3f}s",
            "事件类型计数:",
            event_lines,
        ]
        return "\n".join(lines)

    def export(self) -> dict:
        return {
            "trace": [event.to_dict() for event in self._trace],
            "metrics": dict(self._metrics),
        }

    def reset(self) -> None:
        self._trace = []
        self._metrics = {
            "total_steps": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "start_time": None,
            "end_time": None,
        }
        self._hooks = []

    def format_trace(self) -> str:
        lines = []
        for event in self._trace:
            data_str = ", ".join(f"{k}={v}" for k, v in event.data.items())
            lines.append(
                f"[{event.timestamp:.3f}] step={event.step} "
                f"{event.event_type}: {data_str}"
            )
        return "\n".join(lines)
