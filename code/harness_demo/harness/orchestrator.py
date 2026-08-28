"""执行编排模块：Agent 主循环（think → act → observe），串联上下文、工具、记忆、观测与约束六大子系统。

这是 harness 的"大脑"，负责：
1. 每一步先调用 LLM 推理（think）
2. 若推理产生了工具调用则执行工具（act）
3. 把工具结果写回上下文并检查约束/摘要（observe）
4. 循环直到 LLM 给出最终回答，或触发约束中止
"""

import time
from typing import Optional

from .llm import MockLLM, LLMResponse
from .context import ContextManager
from .tools import ToolRegistry, ToolResult
from .memory import Memory
from .observer import Observer
from .constraints import ConstraintManager, Recovery


class AbortRunError(Exception):
    """约束违规导致运行中止时抛出。"""


class Orchestrator:
    """Agent 执行编排器：驱动 think → act → observe 循环。"""

    def __init__(
        self,
        llm: MockLLM,
        context: ContextManager,
        tools: ToolRegistry,
        memory: Memory,
        observer: Observer,
        constraints: ConstraintManager,
        verbose: bool = True,
    ):
        self.llm = llm
        self.context = context
        self.tools = tools
        self.memory = memory
        self.observer = observer
        self.constraints = constraints
        self.verbose = verbose
        self._final_answer: Optional[str] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, task: str) -> str:
        """执行一个完整任务，返回最终回答文本。"""
        # ---- 初始化 ----
        self.constraints.reset()
        self.observer.reset()
        self.observer.start()
        self.memory.start_task(task)
        self.context.clear()
        self.context.add_user_message(task)
        self.observer.log("info", step=0, message=f"task started: {task[:60]}...")
        self._print(f"\n{'=' * 60}\n任务: {task}\n{'=' * 60}")

        # ---- 主循环 ----
        try:
            step = 0
            while True:
                step += 1
                self.memory.update_state(step_count=step)

                # 每步开始前做约束检查（步数/token/超时/自定义）
                ok, violations = self.constraints.check_all(
                    step=step,
                    tokens=self.context.get_token_count(),
                    elapsed=time.time() - (self.memory.state.started_at or time.time()),
                )
                if not ok:
                    for v in violations:
                        self.observer.log("warning", step=step, violation=v)
                    if self.constraints.should_abort:
                        raise AbortRunError(
                            "约束违规，中止运行: " + "; ".join(self.constraints.get_violations())
                        )

                # 单步执行: think → act → observe
                if not self.step(step):
                    break  # 任务完成（拿到最终回答）

            # ---- 正常收尾 ----
            result = self._final_answer or "(无最终回答)"
            self.memory.finish_task(result)
            self.observer.log("info", step=step, message="task completed")
            self.observer.stop()
            self._print(f"\n{'=' * 60}\n最终回答: {result}\n{'=' * 60}")
            return result

        except AbortRunError as e:
            # ---- 约束中止 ----
            self.memory.fail_task(str(e), status="aborted")
            self.observer.log("error", message=f"aborted: {e}")
            self.observer.stop()
            self._print(f"\n[!] 运行被中止: {e}")
            return f"[aborted] {e}"
        except Exception as e:
            # ---- 意外失败：演示 Recovery 降级 ----
            self.memory.fail_task(str(e))
            self.observer.log("error", message=f"unexpected error: {e}")
            self.observer.stop()
            self._print(f"\n[!] 运行失败: {e}")
            return f"[failed] {e}"

    # ------------------------------------------------------------------
    # 单步执行: think → act → observe
    # ------------------------------------------------------------------

    def step(self, step: int) -> bool:
        """执行一个推理步。返回 True 表示应继续循环，False 表示任务完成。"""

        # ---- 1. THINK: 调用 LLM ----
        response = self._think(step)

        # ---- 2. ACT: 执行工具调用（如有）----
        if response.tool_calls:
            self._act(step, response)

        # ---- 3. OBSERVE: 观察结果、维护上下文 ----
        return self._observe(step, response)

    # ------------------------------------------------------------------
    # 子步骤
    # ------------------------------------------------------------------

    def _think(self, step: int) -> LLMResponse:
        """调用 LLM 推理，记录观测指标与上下文。"""
        messages = self.context.get_messages()
        tools = self.tools.list_tools()

        # 用 Recovery.safe_execute 包装 LLM 调用：LLM 异常时不至于炸掉整个循环
        # 签名: safe_execute(func, default, *args, **kwargs)
        response = Recovery.safe_execute(
            self.llm.chat, None, messages, tools=tools
        )
        if response is None:
            # 降级：返回一个空的 stop 响应
            response = LLMResponse(
                content="(LLM 调用失败，降级为空响应)", finish_reason="stop"
            )

        # 记录观测
        self.observer.log(
            "think",
            step=step,
            content=response.content[:80] + ("..." if len(response.content) > 80 else ""),
            finish_reason=response.finish_reason,
        )
        self.observer.update_metrics(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # 写入上下文
        self.context.add_assistant_message(response.content, response.tool_calls)

        self._print(
            f"\n--- Step {step} [think] finish_reason={response.finish_reason}\n"
            f"    思考: {response.content[:100]}"
            + (f"..." if len(response.content) > 100 else "")
        )
        if response.tool_calls:
            names = ", ".join(f"{tc.name}({tc.arguments})" for tc in response.tool_calls)
            self._print(f"    工具调用: {names}")

        return response

    def _act(self, step: int, response: LLMResponse) -> list[ToolResult]:
        """执行所有工具调用，结果写回上下文与记忆。"""
        results: list[ToolResult] = []

        for tc in response.tool_calls:
            # Recovery.retry 演示：工具瞬时失败时自动重试（指数退避）
            result = Recovery.retry(
                lambda: self.tools.execute(tc.name, tc.arguments),
                max_retries=2,
                delay=0.01,
            )

            results.append(result)

            # 观测记录
            self.observer.log(
                "act",
                step=step,
                tool=tc.name,
                success=result.success,
                output=(result.output or result.error or "")[:60],
            )

            # 工具失败时记录约束错误
            if not result.success:
                still_ok = self.constraints.record_tool_error()
                if not still_ok:
                    self.observer.log(
                        "warning", step=step, message="tool error budget exhausted"
                    )

            # 结果写回上下文
            tool_output = result.output if result.success else f"Error: {result.error}"
            self.context.add_tool_message(tc.id, tc.name, tool_output)

            # 结果存入短期记忆（供后续步骤或快照恢复使用）
            self.memory.set(
                f"step{step}_{tc.name}", tool_output[:200]
            )

            self._print(
                f"--- Step {step} [act] {tc.name} → "
                f"{'OK' if result.success else 'FAIL'}: {tool_output[:80]}"
            )

        return results

    def _observe(self, step: int, response: LLMResponse) -> bool:
        """观察当前状态：判断是否完成、维护上下文窗口。返回 True=继续。"""
        # 判断任务是否完成
        if not response.tool_calls and response.finish_reason == "stop":
            self._final_answer = response.content
            self.observer.log("observe", step=step, message="final answer received")
            return False

        # 上下文超限时自动摘要压缩
        if self.context.should_summarize():
            self.context.summarize()
            self.observer.log(
                "observe",
                step=step,
                message=f"context summarized, "
                f"compressed={self.context.summarized_count} msgs",
            )
            self._print(
                f"--- Step {step} [observe] 上下文超限，已摘要压缩 "
                f"({self.context.summarized_count} 条历史消息)"
            )
        else:
            self.observer.log(
                "observe",
                step=step,
                tokens=self.context.get_token_count(),
            )

        return True

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _print(self, text: str) -> None:
        if self.verbose:
            print(text)
