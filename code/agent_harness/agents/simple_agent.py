# agent_harness/agents/simple_agent.py
"""
简单模拟 Agent —— 不调 LLM，用规则匹配完成任务

这个 Agent 的价值：
  1. 验证 Harness 框架本身能跑通（不依赖 API key）
  2. 作为"基线"对照组（最差能拿多少分）
  3. 演示如何实现 AgentProtocol 接口

策略：
  - 解析任务中的关键词
  - 根据关键词决定调哪个工具
  - 把工具结果拼成回答
"""
from __future__ import annotations

import json
import time

from agent_harness.protocol import AgentProtocol, AgentResult, ToolCall
from agent_harness.tools import run_tool


class SimpleAgent(AgentProtocol):
    """
    基于规则的简单 Agent

    不用 LLM，纯粹通过关键词匹配来决定行为。
    像一个"只会按照说明书操作"的新手。
    """

    name = "SimpleAgent（规则型）"

    # 关键词 → 工具映射
    KEYWORD_TOOL_MAP = {
        "搜索": "search_knowledge",
        "查找": "search_knowledge",
        "查询": "search_knowledge",
        "知识": "search_knowledge",
        "计算": "calculate",
        "算": "calculate",
        "sin": "calculate",
        "cos": "calculate",
        "sqrt": "calculate",
        "笔记": "write_note",
        "保存": "write_note",
        "写": "write_note",
        "整理": "write_note",
    }

    # 常见搜索主题
    TOPIC_KEYWORDS = [
        "agent", "workflow", "context engineering", "上下文工程",
        "react", "ReAct", "tool use", "工具",
    ]

    def run(
        self,
        task: str,
        tools: list[dict] | None = None,
        context: dict | None = None,
    ) -> AgentResult:
        """用规则匹配执行任务"""
        tool_calls: list[ToolCall] = []
        outputs: list[str] = []
        step = 0
        task_lower = task.lower()

        # ── 第一步：判断需要哪些工具 ──
        needed_tools = set()
        for keyword, tool_name in self.KEYWORD_TOOL_MAP.items():
            if keyword in task_lower or keyword in task:
                needed_tools.add(tool_name)

        # ── 第二步：如果需要搜索，先搜索 ──
        if "search_knowledge" in needed_tools:
            # 从任务中提取搜索主题
            query = self._extract_topic(task)
            start = time.perf_counter()
            result = run_tool("search_knowledge", {"query": query})
            elapsed = (time.perf_counter() - start) * 1000

            tc = ToolCall(
                tool_name="search_knowledge",
                arguments={"query": query},
                result=result,
                duration_ms=elapsed,
            )
            tool_calls.append(tc)
            step += 1

            # 解析搜索结果
            try:
                data = json.loads(result)
                if "results" in data:
                    for r in data["results"]:
                        outputs.append(f"【{r['title']}】{r['content']}")
                elif "error" in data:
                    outputs.append(f"搜索失败: {data['message']}")
            except json.JSONDecodeError:
                outputs.append(result)

        # ── 第三步：如果需要计算，执行计算 ──
        if "calculate" in needed_tools:
            expression = self._extract_expression(task)
            if expression:
                start = time.perf_counter()
                result = run_tool("calculate", {"expression": expression})
                elapsed = (time.perf_counter() - start) * 1000

                tc = ToolCall(
                    tool_name="calculate",
                    arguments={"expression": expression},
                    result=result,
                    duration_ms=elapsed,
                )
                tool_calls.append(tc)
                step += 1

                try:
                    data = json.loads(result)
                    if data.get("success"):
                        outputs.append(f"计算 {expression} = {data['result']}")
                    else:
                        outputs.append(f"计算失败: {data.get('message', '未知错误')}")
                except json.JSONDecodeError:
                    outputs.append(result)

        # ── 第四步：如果需要写笔记，用搜索结果写笔记 ──
        if "write_note" in needed_tools and outputs:
            note_content = "\n".join(outputs)
            start = time.perf_counter()
            result = run_tool("write_note", {
                "title": f"关于 {self._extract_topic(task)} 的笔记",
                "content": note_content,
            })
            elapsed = (time.perf_counter() - start) * 1000

            tc = ToolCall(
                tool_name="write_note",
                arguments={"title": "...", "content": note_content[:50] + "..."},
                result=result,
                duration_ms=elapsed,
            )
            tool_calls.append(tc)
            step += 1
            outputs.append("✅ 笔记已保存")

        # ── 汇总输出 ──
        if outputs:
            final_output = "\n\n".join(outputs)
            return AgentResult(
                success=True,
                output=final_output,
                steps=step,
                tool_calls=tool_calls,
            )
        else:
            return AgentResult(
                success=False,
                output="",
                steps=step,
                tool_calls=tool_calls,
                error="无法理解任务，未能执行任何操作",
            )

    def _extract_topic(self, task: str) -> str:
        """从任务描述中提取搜索主题"""
        task_lower = task.lower()
        for topic in self.TOPIC_KEYWORDS:
            if topic.lower() in task_lower:
                return topic
        # 兜底：取任务前几个词
        return task[:20]

    def _extract_expression(self, task: str) -> str | None:
        """从任务描述中提取数学表达式"""
        import re
        # 尝试匹配常见数学表达式模式
        patterns = [
            r"sin\([^)]+\)",
            r"cos\([^)]+\)",
            r"sqrt\([^)]+\)",
            r"[\d\s+\-*/().]+",
        ]
        for pattern in patterns:
            match = re.search(pattern, task)
            if match:
                expr = match.group().strip()
                # 替换中文括号
                expr = expr.replace("（", "(").replace("）", ")")
                # 替换 pi
                expr = expr.replace("pi", "pi")
                if len(expr) >= 3:  # 至少是个有效表达式
                    return expr
        return None
