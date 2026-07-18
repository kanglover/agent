"""
08_error_handling.py — 生产级 Claude API 错误处理全攻略

为什么错误处理很重要？
─────────────────────
AI API 不像本地函数，它依赖网络、有速率限制、会随机失败。
生产环境中，没有健壮的错误处理，用户会频繁看到报错页面。
本文件展示：如何让 Claude 集成像一个"不会轻易崩溃"的系统。
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Callable

import anthropic

# ─────────────────────────────────────────────────────────────
# 第一部分：Anthropic 错误类型总览
#
# Anthropic SDK 的异常继承树：
#
#   AnthropicError
#   └── APIError
#       ├── APIConnectionError   → 网络问题（DNS/超时/断网）
#       ├── APIStatusError       → HTTP 4xx/5xx 响应
#       │   ├── AuthenticationError   → 401：API Key 无效或缺失
#       │   ├── PermissionDeniedError → 403：无权访问该资源
#       │   ├── NotFoundError         → 404：模型/资源不存在
#       │   ├── RateLimitError        → 429：请求太频繁
│       │   ├── BadRequestError       → 400：请求格式错误（如 max_tokens 超限）
#       │   └── InternalServerError   → 5xx：Anthropic 服务端错误
#       └── APITimeoutError      → 请求超时（本地超时设置触发）
# ─────────────────────────────────────────────────────────────

def demo_catch_all_errors() -> None:
    """
    演示捕获各种 Anthropic 错误类型。
    每种错误都有不同的处理策略。
    """
    print("\n── 错误类型捕获演示 ──")

    client = anthropic.Anthropic()

    def call_api(user_message: str = "Hello") -> str | None:
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=64,
                messages=[{"role": "user", "content": user_message}],
            )
            return message.content[0].text

        except anthropic.RateLimitError as e:
            # 429：请求过于频繁，需要等待后重试
            # e.headers 中通常有 retry-after 字段告诉你等多久
            retry_after = int(e.response.headers.get("retry-after", 60))
            print(f"[RateLimitError] 速率超限，建议 {retry_after}s 后重试")
            print(f"  状态码: {e.status_code}, 消息: {e.message}")
            # 生产建议：加入重试队列，不要立即报错给用户
            return None

        except anthropic.AuthenticationError as e:
            # 401：API Key 不对
            # 这个错误不应该重试，需要检查配置
            print(f"[AuthenticationError] API Key 无效或未设置")
            print(f"  请检查 ANTHROPIC_API_KEY 环境变量")
            raise   # 重新抛出，让上层处理配置问题

        except anthropic.PermissionDeniedError as e:
            # 403：账号没有权限访问该模型/功能
            print(f"[PermissionDeniedError] 权限不足")
            print(f"  可能原因：试图访问未开放的模型或功能")
            return None

        except anthropic.NotFoundError as e:
            # 404：模型名写错了，或者该资源不存在
            print(f"[NotFoundError] 资源不存在: {e.message}")
            print(f"  请检查模型名称是否正确（如 claude-opus-4-5）")
            return None

        except anthropic.BadRequestError as e:
            # 400：请求格式有问题，例如 max_tokens 超出模型上限
            print(f"[BadRequestError] 请求格式错误: {e.message}")
            print(f"  常见原因：max_tokens 过大、prompt 格式不对")
            return None

        except anthropic.InternalServerError as e:
            # 5xx：Anthropic 服务端出问题了，可以稍后重试
            print(f"[InternalServerError] 服务端错误 {e.status_code}")
            print(f"  通常是临时问题，稍后重试即可")
            return None

        except anthropic.APIConnectionError as e:
            # 网络连接问题：DNS 解析失败、连接超时、断网等
            print(f"[APIConnectionError] 网络连接失败: {e}")
            print(f"  请检查网络连接，或者是否需要代理")
            return None

        except anthropic.APIStatusError as e:
            # 捕获所有其他 HTTP 错误（上面没覆盖到的状态码）
            print(f"[APIStatusError] HTTP {e.status_code}: {e.message}")
            return None

    # 尝试调用
    result = call_api("请用一句话介绍自己")
    if result:
        print(f"调用成功：{result[:80]}...")


# ─────────────────────────────────────────────────────────────
# 第二部分：retry_with_backoff 装饰器
#
# 指数退避（Exponential Backoff）是什么？
# 第 1 次失败：等 1s 重试
# 第 2 次失败：等 2s 重试
# 第 3 次失败：等 4s 重试
# 加上 jitter（随机抖动）是为了避免多个客户端同时重试导致"重试风暴"。
# ─────────────────────────────────────────────────────────────

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    ),
):
    """
    指数退避重试装饰器。

    参数：
        max_retries          : 最大重试次数（不含首次尝试）
        base_delay           : 初始等待时间（秒）
        max_delay            : 最大等待时间上限（秒）
        jitter               : 是否加随机抖动（推荐开启）
        retryable_exceptions : 哪些异常值得重试（不可重试的如 AuthError 不要重试）

    用法：
        @retry_with_backoff(max_retries=3)
        def call_claude():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # 已经是最后一次尝试，不再重试
                        print(f"[retry] 已达最大重试次数 {max_retries}，放弃")
                        raise

                    # 计算等待时间：2^attempt * base_delay
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    # 加入 jitter：在 [delay/2, delay*1.5] 之间随机
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    print(f"[retry] 第 {attempt + 1}/{max_retries} 次失败: "
                          f"{type(e).__name__}，{delay:.1f}s 后重试...")
                    time.sleep(delay)

                except Exception:
                    # 不可重试的异常（如 AuthenticationError），直接抛出
                    raise

            raise last_exception   # 理论上不会到这里，但 mypy 需要

        return wrapper
    return decorator


# 使用装饰器的示例函数
@retry_with_backoff(max_retries=3, base_delay=1.0)
def resilient_api_call(prompt: str) -> str:
    """这个函数在遇到速率限制或服务端错误时会自动重试"""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ─────────────────────────────────────────────────────────────
# 第三部分：CircuitBreaker（熔断器）
#
# 熔断器模式来自电路保险丝：过载时自动断开，保护系统。
# 状态机：
#   CLOSED（正常）→ 连续失败 threshold 次 → OPEN（熔断）
#   OPEN（熔断，直接拒绝请求）→ reset_timeout 秒后 → HALF_OPEN（试探）
#   HALF_OPEN（试探）→ 成功 → CLOSED，失败 → OPEN
#
# 为什么需要熔断器？
# 如果 API 持续失败，不断重试只会浪费资源、阻塞线程。
# 熔断后直接快速失败，让系统有时间恢复。
# ─────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    熔断器实现。

    参数：
        failure_threshold : 连续失败多少次后熔断（默认 5）
        reset_timeout     : 熔断后等待多少秒再试探（默认 60）
    """

    # 三种状态
    CLOSED = "closed"        # 正常工作
    OPEN = "open"            # 熔断中（拒绝所有请求）
    HALF_OPEN = "half_open"  # 试探中（允许一个请求通过）

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None

    def _should_attempt(self) -> bool:
        """判断当前是否允许尝试请求"""
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            # 检查是否已超过 reset_timeout
            if (self.last_failure_time and
                    time.time() - self.last_failure_time >= self.reset_timeout):
                print(f"[熔断器] 进入 HALF_OPEN 状态，尝试探测...")
                self.state = self.HALF_OPEN
                return True
            return False

        # HALF_OPEN：允许一次请求
        return True

    def record_success(self) -> None:
        """记录一次成功"""
        if self.state == self.HALF_OPEN:
            print(f"[熔断器] 探测成功，恢复 CLOSED 状态")
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self) -> None:
        """记录一次失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == self.HALF_OPEN or self.failure_count >= self.failure_threshold:
            print(f"[熔断器] 触发熔断！连续失败 {self.failure_count} 次，"
                  f"OPEN 状态持续 {self.reset_timeout}s")
            self.state = self.OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器执行函数。
        如果熔断器 OPEN，抛出 RuntimeError 而不调用函数。
        """
        if not self._should_attempt():
            remaining = self.reset_timeout - (time.time() - (self.last_failure_time or 0))
            raise RuntimeError(
                f"熔断器 OPEN，拒绝请求。{remaining:.0f}s 后重试。"
            )
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


# 全局熔断器实例
_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)


# ─────────────────────────────────────────────────────────────
# 第四部分：asyncio 超时保护
#
# 为什么需要超时？
# 有时 API 请求会"挂住"很久没有响应（不是报错，只是没反应）。
# asyncio.wait_for 可以设置一个最长等待时间，超过就主动取消。
# ─────────────────────────────────────────────────────────────

async def async_call_with_timeout(
    prompt: str,
    timeout_seconds: float = 30.0,
) -> str | None:
    """
    带超时保护的异步 API 调用。

    参数：
        timeout_seconds : 最长等待时间，超过就取消（默认 30s）
    """
    client = anthropic.AsyncAnthropic()   # 异步版客户端

    async def _call() -> str:
        message = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    try:
        # asyncio.wait_for：如果 _call() 在 timeout_seconds 内没完成，抛出 TimeoutError
        result = await asyncio.wait_for(_call(), timeout=timeout_seconds)
        print(f"[超时保护] 请求在 {timeout_seconds}s 内完成")
        return result

    except asyncio.TimeoutError:
        print(f"[超时保护] 请求超过 {timeout_seconds}s，已取消")
        return None
    except anthropic.APIError as e:
        print(f"[超时保护] API 错误: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 第五部分：fallback_to_smaller_model（降级策略）
#
# 降级策略：当大模型不可用时，自动切换到小模型。
# 大模型（Opus）→ 中模型（Sonnet）→ 小模型（Haiku）
# 这样即使大模型过载，用户也能得到（质量略低的）响应。
# ─────────────────────────────────────────────────────────────

# 模型降级链（从贵到便宜）
MODEL_FALLBACK_CHAIN = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]


def fallback_to_smaller_model(
    prompt: str,
    preferred_model: str = "claude-opus-4-5",
    max_tokens: int = 256,
) -> dict:
    """
    带降级策略的 API 调用。
    从 preferred_model 开始，失败则尝试下一个更小的模型。

    返回：{"model_used": str, "response": str, "fallback_count": int}
    """
    client = anthropic.Anthropic()

    # 找到 preferred_model 在降级链中的位置
    try:
        start_idx = MODEL_FALLBACK_CHAIN.index(preferred_model)
    except ValueError:
        start_idx = 0

    fallback_count = 0
    last_error = None

    for model in MODEL_FALLBACK_CHAIN[start_idx:]:
        try:
            print(f"[降级策略] 尝试模型: {model}")
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if fallback_count > 0:
                print(f"[降级策略] 已降级 {fallback_count} 次，最终使用 {model}")
            return {
                "model_used": model,
                "response": message.content[0].text,
                "fallback_count": fallback_count,
            }

        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last_error = e
            fallback_count += 1
            print(f"[降级策略] {model} 失败: {type(e).__name__}，尝试下一个模型...")
            continue

        except anthropic.AuthenticationError:
            # Auth 错误无论用哪个模型都会失败，不要降级
            raise

    # 所有模型都失败了
    raise RuntimeError(f"所有模型均失败，最后错误: {last_error}") from last_error


# ─────────────────────────────────────────────────────────────
# 第六部分：structured_log_error（结构化错误日志）
#
# 什么是结构化日志？
# 普通日志：2024-01-01 ERROR: API failed
# 结构化日志：{"timestamp": "2024-01-01", "level": "ERROR", "error_type": "RateLimitError", ...}
# 结构化日志方便机器解析，可以接入监控系统（如 Datadog、Grafana）。
# ─────────────────────────────────────────────────────────────

# 配置 Python logging，输出 JSON 格式
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # 我们自己格式化 JSON
)
_logger = logging.getLogger("claude_errors")


def structured_log_error(
    error: Exception,
    context: dict | None = None,
    extra: dict | None = None,
) -> None:
    """
    记录结构化 JSON 格式的错误日志。

    参数：
        error   : 捕获到的异常
        context : 调用时的上下文（如 model、prompt 前 50 字等）
        extra   : 其他附加信息
    """
    log_entry: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": "ERROR",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    # 针对 Anthropic 错误，提取更多信息
    if isinstance(error, anthropic.APIStatusError):
        log_entry["http_status"] = error.status_code
        log_entry["request_id"] = error.response.headers.get("request-id", "unknown")

    if isinstance(error, anthropic.RateLimitError):
        log_entry["retry_after"] = error.response.headers.get("retry-after")

    # 追加上下文
    if context:
        log_entry["context"] = context
    if extra:
        log_entry["extra"] = extra

    # 输出为 JSON（生产环境中通常直接写文件或发到日志服务）
    json_str = json.dumps(log_entry, ensure_ascii=False, indent=None)
    _logger.error(json_str)
    print(f"[结构化日志] {json_str}")


# ─────────────────────────────────────────────────────────────
# 第七部分：token_budget_guard（预算守卫）
#
# Token 预算保护：在调用 API 前，先估算 token 数，
# 如果超出预算就主动拒绝，避免意外的高额账单。
# ─────────────────────────────────────────────────────────────

class TokenBudgetExceededError(Exception):
    """自定义异常：token 预算超出"""
    pass


def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数量。
    规则：
    - 英文：约 4 个字符 = 1 token
    - 中文：约 1.5 个字符 = 1 token
    注意：这只是估算，真实 token 数以 API 返回为准。
    """
    import re
    # 统计中文字符数
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    # 其余字符
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def token_budget_guard(
    prompt: str,
    max_input_tokens: int = 4000,
    max_output_tokens: int = 1000,
    session_budget: dict | None = None,
) -> None:
    """
    Token 预算检查。超出时主动抛出 TokenBudgetExceededError。

    参数：
        prompt           : 要发送的 prompt
        max_input_tokens : 单次请求最大输入 token 数
        max_output_tokens: 单次请求最大输出 token 数（max_tokens 参数）
        session_budget   : 可选，会话级累计预算 {"used": int, "limit": int}
    """
    estimated_input = estimate_tokens(prompt)
    estimated_total = estimated_input + max_output_tokens

    print(f"[预算守卫] 预估输入 tokens: ~{estimated_input}")
    print(f"[预算守卫] 请求上限 max_tokens: {max_output_tokens}")

    # 检查单次输入限制
    if estimated_input > max_input_tokens:
        raise TokenBudgetExceededError(
            f"输入 token 估算 {estimated_input} 超过单次限制 {max_input_tokens}。"
            f"请缩短 prompt 或提高 max_input_tokens。"
        )

    # 检查会话级累计预算
    if session_budget is not None:
        new_total = session_budget.get("used", 0) + estimated_total
        limit = session_budget.get("limit", float("inf"))
        if new_total > limit:
            raise TokenBudgetExceededError(
                f"会话累计 token 将达到 {new_total}，超过会话预算 {limit}。"
                f"请开始新会话或提高预算。"
            )
        session_budget["used"] = new_total
        print(f"[预算守卫] 会话已用 tokens: {session_budget['used']}/{limit}")

    print(f"[预算守卫] 通过检查，继续请求")


# ─────────────────────────────────────────────────────────────
# 演示：把所有场景串起来
# ─────────────────────────────────────────────────────────────

def demo_retry_decorator() -> None:
    print("\n── 演示：retry_with_backoff 装饰器 ──")
    # 为了演示，我们直接调用（如果有 key 就会真正请求 API）
    try:
        result = resilient_api_call("请用一句话解释什么是重试机制")
        print(f"成功：{result[:80]}")
    except Exception as e:
        print(f"最终失败：{type(e).__name__}: {e}")


def demo_circuit_breaker() -> None:
    print("\n── 演示：CircuitBreaker 熔断器 ──")
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5.0)

    def flaky_function(fail: bool = False) -> str:
        if fail:
            raise anthropic.InternalServerError(
                message="模拟服务端错误",
                response=None,  # type: ignore
                body=None,
            )
        return "成功响应"

    # 模拟连续失败触发熔断
    for i in range(4):
        try:
            result = breaker.call(flaky_function, fail=(i < 3))
            print(f"  第 {i+1} 次: {result}")
        except RuntimeError as e:
            print(f"  第 {i+1} 次: 熔断器拒绝 — {e}")
        except Exception as e:
            print(f"  第 {i+1} 次: 函数失败 — {type(e).__name__}")

    print(f"  熔断器状态: {breaker.state}")


def demo_timeout_protection() -> None:
    print("\n── 演示：asyncio 超时保护 ──")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  （跳过：未设置 API Key）")
        return

    # 运行异步函数
    result = asyncio.run(
        async_call_with_timeout("你好", timeout_seconds=30.0)
    )
    if result:
        print(f"  响应：{result[:80]}")


def demo_model_fallback() -> None:
    print("\n── 演示：fallback_to_smaller_model 降级策略 ──")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  （跳过：未设置 API Key）")
        return

    result = fallback_to_smaller_model(
        prompt="请用一句话解释什么是模型降级",
        preferred_model="claude-haiku-4-5",   # 从最小的开始（演示用）
    )
    print(f"  使用模型: {result['model_used']}")
    print(f"  降级次数: {result['fallback_count']}")
    print(f"  响应: {result['response'][:80]}")


def demo_structured_logging() -> None:
    print("\n── 演示：structured_log_error 结构化日志 ──")
    # 模拟一个 RateLimitError
    try:
        raise ValueError("模拟一个普通错误")
    except ValueError as e:
        structured_log_error(
            error=e,
            context={"model": "claude-opus-4-5", "prompt_preview": "请帮我..."},
            extra={"user_id": "u_12345", "request_id": "req_abc"},
        )


def demo_token_budget() -> None:
    print("\n── 演示：token_budget_guard 预算守卫 ──")

    # 正常请求
    short_prompt = "请介绍一下 Python。"
    print(f"\n短 prompt（'{short_prompt}'）：")
    try:
        token_budget_guard(short_prompt, max_input_tokens=500)
        print("  → 通过预算检查")
    except TokenBudgetExceededError as e:
        print(f"  → 被拦截: {e}")

    # 超长请求
    long_prompt = "请详细解释" + "每一个步骤都要非常详细，包括原理和代码示例。" * 200
    print(f"\n超长 prompt（{len(long_prompt)} 字）：")
    try:
        token_budget_guard(long_prompt, max_input_tokens=1000)
        print("  → 通过预算检查")
    except TokenBudgetExceededError as e:
        print(f"  → 被拦截: {e}")

    # 会话预算示例
    session = {"used": 0, "limit": 2000}
    print(f"\n会话预算演示（总限制 {session['limit']} tokens）：")
    prompts = ["第一个问题", "第二个问题很长" * 50, "第三个问题"]
    for p in prompts:
        try:
            token_budget_guard(p, max_input_tokens=5000, session_budget=session)
        except TokenBudgetExceededError as e:
            print(f"  → 会话预算超出: {e}")
            break


def main() -> None:
    print("=" * 60)
    print("08_error_handling.py — 生产级错误处理全演示")
    print("=" * 60)

    demo_circuit_breaker()
    demo_structured_logging()
    demo_token_budget()

    if os.environ.get("ANTHROPIC_API_KEY"):
        demo_catch_all_errors()
        demo_retry_decorator()
        demo_timeout_protection()
        demo_model_fallback()
    else:
        print("\n（未设置 ANTHROPIC_API_KEY，跳过需要 API 的演示）")
        print("以下演示需要 API Key（可正常运行上面 3 个）：")
        print("  - demo_catch_all_errors()")
        print("  - demo_retry_decorator()")
        print("  - demo_timeout_protection()")
        print("  - demo_model_fallback()")

    print("\n" + "=" * 60)
    print("全部演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
