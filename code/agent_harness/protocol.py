# agent_harness/protocol.py
"""
Agent 协议 —— 所有参赛 Agent 必须遵守的"考生须知"

类比：
  就像驾考要求每个学员都坐进同一种考试车、按同样的流程操作，
  Harness 要求每个 Agent 都实现相同的接口，这样才能公平对比。

核心概念：
  - AgentProtocol: 抽象基类，定义 Agent 必须实现的方法
  - AgentResult:   Agent 运行结果的标准格式
  - ToolCall:      工具调用记录
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── 数据结构：工具调用记录 ───────────────────────────────────────

@dataclass
class ToolCall:
    """一次工具调用的完整记录（Agent 做了什么操作）"""
    tool_name: str          # 调用了哪个工具
    arguments: dict         # 传了什么参数
    result: str             # 工具返回了什么
    duration_ms: float = 0  # 这次调用花了多久（毫秒）


# ── 数据结构：Agent 运行结果 ─────────────────────────────────────

@dataclass
class AgentResult:
    """
    Agent 完成任务后交回的"答卷"

    不管 Agent 内部怎么实现，最终都要包装成这个格式，
    这样 Harness 才能统一评分。
    """
    success: bool                          # 任务是否完成
    output: str                            # Agent 的最终回答
    steps: int = 0                         # 总共走了几步
    tool_calls: list[ToolCall] = field(default_factory=list)  # 所有工具调用记录
    error: str | None = None               # 如果失败了，错误信息是什么
    duration_ms: float = 0                 # 总耗时（毫秒）
    metadata: dict = field(default_factory=dict)  # 额外信息（token 用量等）


# ── 抽象基类：Agent 协议 ─────────────────────────────────────────

class AgentProtocol(ABC):
    """
    Agent 协议——所有"考生"必须实现的接口

    使用方式：
        class MyAgent(AgentProtocol):
            name = "我的Agent"

            def run(self, task, tools, context):
                # ... 你的 Agent 逻辑 ...
                return AgentResult(success=True, output="完成了")
    """

    # 每个 Agent 必须有个名字（用于报告和对比）
    name: str = "UnnamedAgent"

    @abstractmethod
    def run(
        self,
        task: str,
        tools: list[dict] | None = None,
        context: dict | None = None,
    ) -> AgentResult:
        """
        执行一个任务，返回标准化结果

        Args:
            task:    任务描述（自然语言，比如"搜索 agent 笔记并写摘要"）
            tools:   可用工具列表（Harness 提供，格式同 Claude API 的 tools）
            context: 额外上下文（比如系统提示词、约束条件等）

        Returns:
            AgentResult: 标准化的运行结果
        """
        ...

    def setup(self) -> None:
        """
        可选：运行前的初始化（比如加载模型、建立连接）
        默认什么都不做，子类按需覆盖。
        """
        pass

    def teardown(self) -> None:
        """
        可选：运行后的清理（比如关闭连接、释放资源）
        默认什么都不做，子类按需覆盖。
        """
        pass

    def run_with_lifecycle(
        self,
        task: str,
        tools: list[dict] | None = None,
        context: dict | None = None,
    ) -> AgentResult:
        """
        带生命周期管理的运行方法（Harness 实际调用的入口）

        自动完成：
          1. setup()   → 初始化
          2. run()     → 执行任务（计时）
          3. teardown() → 清理
          4. 即使出错也保证 teardown 被调用
        """
        self.setup()
        start = time.perf_counter()
        try:
            result = self.run(task, tools, context)
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False,
                output="",
                error=f"Agent 运行异常: {type(e).__name__}: {e}",
                duration_ms=elapsed,
            )
        finally:
            self.teardown()
