"""约束管理模块：步数/token/工具错误/超时限制检查、违规记录，以及重试/降级/安全执行恢复策略。"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class Constraint:
    name: str
    description: str
    check: Callable[[], bool]   # 返回 True 表示违规(violated)
    severity: str = "error"     # error | warning


class ConstraintManager:
    """约束管理: 步数/token/工具错误/超时限制，违规记录。"""

    def __init__(self, max_steps: int = 20, max_tokens: int = 10000,
                 max_tool_errors: int = 3, timeout: float = 300.0):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_tool_errors = max_tool_errors
        self.timeout = timeout
        self._custom: list[Constraint] = []
        self._violations: list[str] = []
        self._tool_error_count = 0
        self._has_error_violation = False  # 内置或 error 级自定义违规的存在标记

    def add_constraint(self, constraint: Constraint) -> None:
        self._custom.append(constraint)

    def _record_violation(self, message: str, severity: str = "error") -> None:
        self._violations.append(message)
        if severity == "error":
            self._has_error_violation = True

    def check_step(self, current_step: int) -> bool:
        if current_step > self.max_steps:
            self._record_violation(
                f"Max steps exceeded: {current_step}/{self.max_steps}"
            )
            return False
        return True

    def check_tokens(self, token_count: int) -> bool:
        if token_count > self.max_tokens:
            self._record_violation(
                f"Max tokens exceeded: {token_count}/{self.max_tokens}"
            )
            return False
        return True

    def record_tool_error(self) -> bool:
        self._tool_error_count += 1
        if self._tool_error_count > self.max_tool_errors:
            self._record_violation(
                f"Max tool errors exceeded: {self._tool_error_count}/{self.max_tool_errors}"
            )
            return False
        return True

    def check_timeout(self, elapsed: float) -> bool:
        if elapsed > self.timeout:
            self._record_violation(
                f"Timeout exceeded: {elapsed:.1f}s/{self.timeout:.1f}s"
            )
            return False
        return True

    def check_all(self, step: int, tokens: int, elapsed: float) -> tuple:
        violations: list[str] = []
        ok = True

        if not self.check_step(step):
            ok = False
        if not self.check_tokens(tokens):
            ok = False
        if not self.check_timeout(elapsed):
            ok = False

        for constraint in self._custom:
            try:
                violated = constraint.check()
            except Exception:
                violated = False
            if violated:
                ok = False
                self._record_violation(
                    f"[{constraint.severity}] {constraint.name}: {constraint.description}",
                    severity=constraint.severity,
                )

        violations = list(self._violations)
        return (ok, violations)

    @property
    def should_abort(self) -> bool:
        return self._has_error_violation

    def get_violations(self) -> list:
        return list(self._violations)

    def reset(self) -> None:
        self._violations = []
        self._tool_error_count = 0
        self._has_error_violation = False


class Recovery:
    """恢复策略: 重试、降级、安全执行。"""

    @staticmethod
    def retry(func: Callable, max_retries: int = 3, delay: float = 0.1, *args, **kwargs) -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(delay * (2 ** (attempt - 1)))
        raise last_exc  # type: ignore[misc]

    @staticmethod
    """
    使用后备机制执行函数。

    先尝试执行主函数，如果发生异常则执行备用函数。

    Args:
        primary: 主函数，优先执行
        fallback: 备用函数，主函数异常时执行
        *args: 传递给函数的位置参数
        **kwargs: 传递给函数的关键字参数

    Returns:
        主函数或备用函数的返回值
    """
    def with_fallback(primary: Callable, fallback: Callable, *args, **kwargs) -> Any:
        try:
            return primary(*args, **kwargs)
        except Exception:
            return fallback(*args, **kwargs)

    @staticmethod
    def safe_execute(func: Callable, default: Any = None, *args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            return default
