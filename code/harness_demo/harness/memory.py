"""Agent 记忆系统模块：短期记忆(dict)、长期记忆(JSON 文件持久化)、任务状态管理与快照恢复。"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class AgentState:
    task: str = ""
    status: str = "pending"        # pending | running | completed | failed | aborted
    step_count: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


class Memory:
    """Agent 记忆系统: 短期(dict)、长期(JSON 文件持久化)、任务状态管理。"""

    def __init__(self, storage_path: Optional[str] = None):
        self._short_term: dict = {}
        self._long_term: dict = {}
        self.storage_path = storage_path
        self._state = AgentState()
        if storage_path and os.path.exists(storage_path):
            try:
                with open(storage_path, "r", encoding="utf-8") as f:
                    self._long_term = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._long_term = {}

    # ---------------- 短期记忆（会话内 dict） ----------------

    def set(self, key: str, value: Any) -> None:
        self._short_term[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._short_term.get(key, default)

    def delete(self, key: str) -> None:
        self._short_term.pop(key, None)

    def keys(self) -> list:
        return list(self._short_term.keys())

    # ---------------- 长期记忆（跨会话，JSON 文件） ----------------

    def remember(self, key: str, value: Any) -> None:
        self._long_term[key] = self._jsonable(value)
        self._persist()

    def recall(self, key: str, default: Any = None) -> Any:
        return self._long_term.get(key, default)

    def recall_all(self) -> dict:
        return dict(self._long_term)

    def _jsonable(self, value: Any) -> Any:
        """确保 value 可 JSON 序列化，否则降级为 str(value)。"""
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def _persist(self) -> None:
        if not self.storage_path:
            return
        try:
            directory = os.path.dirname(self.storage_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._long_term, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---------------- 状态管理 ----------------

    @property
    def state(self) -> AgentState:
        return self._state

    def update_state(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

    def start_task(self, task: str) -> None:
        self._state.task = task
        self._state.status = "running"
        self._state.started_at = time.time()

    def finish_task(self, result: str, status: str = "completed") -> None:
        self._state.status = status
        self._state.result = result
        self._state.finished_at = time.time()

    def fail_task(self, error: str, status: str = "failed") -> None:
        self._state.status = status
        self._state.error = error
        self._state.finished_at = time.time()

    # ---------------- 快照/恢复（用于断点恢复演示） ----------------

    def snapshot(self) -> dict:
        return {
            "short_term": dict(self._short_term),
            "state": asdict(self._state),
        }

    def restore(self, data: dict) -> None:
        self._short_term = dict(data.get("short_term", {}))
        state_data = data.get("state", {})
        if state_data:
            self._state = AgentState(**{
                k: v for k, v in state_data.items() if k in AgentState.__dataclass_fields__
            })
