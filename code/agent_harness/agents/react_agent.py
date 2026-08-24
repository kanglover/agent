# agent_harness/agents/react_agent.py
"""
ReAct Agent —— 使用 Claude API 的真实 Agent

这个 Agent 的价值：
  1. 展示真实的 LLM Agent 如何接入 Harness 框架
  2. 使用 ReAct 模式（Reasoning + Acting）
  3. 对比 SimpleAgent，看 LLM 强在哪里

策略：
  - 调用 Claude API 进行推理
  - 让 Claude 自主决定调用哪个工具
  - 循环执行直到任务完成或达到步数上限
"""
from __future__ import annotations

import os
import time

from agent_harness.protocol import AgentProtocol, AgentResult, ToolCall
from agent_harness.tools import run_tool


# 最大循环步数（防止无限循环）
MAX_STEPS = 8


class ReactAgent(AgentProtocol):
    """
    基于 Claude API 的 ReAct Agent

    内部使用 observe→think→act 循环：
      1. THINK: 让 Claude 分析当前状态，决定下一步
      2. ACT:   如果 Claude 要调用工具，就执行工具
      3. OBSERVE: 把工具结果喂回给 Claude，继续循环
      4. 如果 Claude 认为任务完成，就输出最终回答
    """

    name = "ReactAgent（LLM型）"

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self._model = model
        self._client = None

    def setup(self) -> None:
        """初始化 Anthropic 客户端"""
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise RuntimeError(
                "需要安装 anthropic 库: pip install anthropic"
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic 客户端初始化失败: {e}")

    def teardown(self) -> None:
        """清理客户端"""
        self._client = None

    def run(
        self,
        task: str,
        tools: list[dict] | None = None,
        context: dict | None = None,
    ) -> AgentResult:
        """使用 Claude API 执行 ReAct 循环"""
        if self._client is None:
            return AgentResult(
                success=False,
                output="",
                error="客户端未初始化，请先调用 setup()",
            )

        # 构建消息历史
        messages = [{"role": "user", "content": task}]
        tool_calls_record: list[ToolCall] = []
        step = 0
        total_input_tokens = 0
        total_output_tokens = 0

        system_prompt = (
            "你是一个助手，使用提供的工具完成用户的任务。\n"
            "请先分析任务需要做什么，然后一步步完成。\n"
            "完成所有步骤后，给用户一个清晰的最终回答。"
        )
        if context and "system" in context:
            system_prompt = context["system"]

        try:
            while step < MAX_STEPS:
                # ── THINK：让 Claude 决定下一步 ──
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1500,
                    system=system_prompt,
                    tools=tools or [],
                    messages=messages,
                )

                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

                # ── 任务完成：Claude 直接回复 ──
                if response.stop_reason == "end_turn":
                    final_text = next(
                        (b.text for b in response.content if hasattr(b, "text")),
                        "任务完成",
                    )
                    return AgentResult(
                        success=True,
                        output=final_text,
                        steps=step + 1,
                        tool_calls=tool_calls_record,
                        metadata={
                            "model": self._model,
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                        },
                    )

                # ── ACT：执行工具调用 ──
                if response.stop_reason == "tool_use":
                    tool_results = []

                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        # 执行工具并计时
                        start = time.perf_counter()
                        observation = run_tool(block.name, block.input)
                        elapsed = (time.perf_counter() - start) * 1000

                        # 记录工具调用
                        tc = ToolCall(
                            tool_name=block.name,
                            arguments=block.input,
                            result=observation,
                            duration_ms=elapsed,
                        )
                        tool_calls_record.append(tc)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": observation,
                        })

                    # OBSERVE：把工具结果喂回 Claude
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                    step += 1

            # 超过最大步数
            return AgentResult(
                success=False,
                output="",
                steps=step,
                tool_calls=tool_calls_record,
                error=f"超过最大步数限制 ({MAX_STEPS} 步)",
                metadata={
                    "model": self._model,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                },
            )

        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                steps=step,
                tool_calls=tool_calls_record,
                error=f"API 调用失败: {type(e).__name__}: {e}",
            )


class MockLLMAgent(AgentProtocol):
    """
    模拟 LLM Agent —— 不调真实 API，但模拟 LLM 的行为模式

    用于：
      - 没有 API key 时测试框架
      - 模拟一个"聪明的 Agent"作为对照
      - 演示 LLM Agent 的行为模式
    """

    name = "MockLLMAgent（模拟型）"

    def run(
        self,
        task: str,
        tools: list[dict] | None = None,
        context: dict | None = None,
    ) -> AgentResult:
        """模拟 LLM Agent 的行为"""
        tool_calls_record: list[ToolCall] = []
        outputs: list[str] = []
        step = 0
        task_lower = task.lower()

        # ── 模拟"思考"：分析任务需要哪些步骤 ──
        needs_search = any(kw in task_lower for kw in ["搜索", "查找", "知识", "内容"])
        needs_calc = any(kw in task_lower for kw in ["计算", "算", "sin", "cos"])
        needs_write = any(kw in task_lower for kw in ["笔记", "保存", "整理", "写"])

        # ── 模拟"推理"：提取关键参数 ──
        search_topics = []
        for topic in ["agent", "workflow", "context engineering", "react", "tool use",
                       "上下文工程", "ReAct", "工具"]:
            if topic.lower() in task_lower:
                search_topics.append(topic)
        if not search_topics and needs_search:
            search_topics = ["agent"]

        # ── 模拟"行动"：按顺序执行工具 ──

        # Step 1: 搜索
        search_results = []
        for topic in search_topics:
            start = time.perf_counter()
            result = run_tool("search_knowledge", {"query": topic})
            elapsed = (time.perf_counter() - start) * 1000

            tool_calls_record.append(ToolCall(
                tool_name="search_knowledge",
                arguments={"query": topic},
                result=result,
                duration_ms=elapsed,
            ))
            step += 1

            import json
            try:
                data = json.loads(result)
                if "results" in data:
                    for r in data["results"]:
                        search_results.append(r)
                        outputs.append(f"【{r['title']}】{r['content']}")
            except json.JSONDecodeError:
                outputs.append(result)

        # Step 2: 计算
        if needs_calc:
            import re
            # 更智能的表达式提取
            expr = None
            if "sin(pi)" in task_lower:
                expr = "sin(pi)"
            elif "cos(pi)" in task_lower:
                expr = "cos(pi)"
            else:
                match = re.search(r"[\d(][^，。？！]*[\d)]", task)
                if match:
                    expr = match.group().replace("（", "(").replace("）", ")")

            if expr:
                start = time.perf_counter()
                result = run_tool("calculate", {"expression": expr})
                elapsed = (time.perf_counter() - start) * 1000

                tool_calls_record.append(ToolCall(
                    tool_name="calculate",
                    arguments={"expression": expr},
                    result=result,
                    duration_ms=elapsed,
                ))
                step += 1

                import json
                try:
                    data = json.loads(result)
                    if data.get("success"):
                        outputs.append(
                            f"计算结果: {expr} = {data['result']}"
                        )
                except json.JSONDecodeError:
                    pass

        # Step 3: 写笔记
        if needs_write and outputs:
            note_content = "\n".join(outputs)
            title = f"关于 {', '.join(search_topics) if search_topics else '任务'} 的笔记"

            start = time.perf_counter()
            result = run_tool("write_note", {
                "title": title,
                "content": note_content,
            })
            elapsed = (time.perf_counter() - start) * 1000

            tool_calls_record.append(ToolCall(
                tool_name="write_note",
                arguments={"title": title, "content": "..."},
                result=result,
                duration_ms=elapsed,
            ))
            step += 1
            outputs.append(f"✅ 笔记「{title}」已保存")

        # ── 模拟"总结"：生成最终回答 ──
        if outputs:
            # 模拟 LLM 的总结能力——加一段综述
            summary = f"根据对{'、'.join(search_topics) if search_topics else '相关主题'}的搜索和分析，"
            summary += "以下是我找到的信息：\n\n"
            summary += "\n\n".join(outputs)
            return AgentResult(
                success=True,
                output=summary,
                steps=step,
                tool_calls=tool_calls_record,
            )
        else:
            return AgentResult(
                success=False,
                output="",
                steps=step,
                tool_calls=tool_calls_record,
                error="未能理解任务意图",
            )
