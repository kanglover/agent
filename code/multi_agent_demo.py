"""
多 Agent 协作系统示例（纯模拟版，无需 API Key）
=================================================

学习目标：理解多 Agent 系统的核心概念和端到端运行流程

架构：
  Coordinator（调度员）
    ├── Researcher（研究员）— 收集信息
    ├── Writer（写手）— 产出内容
    └── Reviewer（审核员）— 检查质量

流程：
  用户任务 → Coordinator 拆分 → Worker 执行 → 回报 Coordinator
           → 继续路由 或 FINISH

特点：
  - 纯 Python 实现，不依赖任何外部 API
  - 使用 Mock LLM 模拟每个 Agent 的"思考"过程
  - 完整展示消息传递、状态共享、路由决策
  - 彩色输出，运行效果直观

运行方法：
  cd code && python multi_agent_demo.py
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 数据模型 —— 每个 Agent 的输入输出、全局状态
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """Agent 之间传递的消息"""
    sender: str      # 发送者名称
    receiver: str    # 接收者名称
    content: str     # 消息内容
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskState:
    """全局共享状态 —— 所有 Agent 读写同一份状态"""
    task: str = ""                    # 用户原始任务
    subtasks: list[str] = field(default_factory=list)   # 拆分后的子任务
    research: str = ""                # 研究员的调研结果
    draft: str = ""                   # 写手的草稿
    review: str = ""                  # 审核员的意见
    revision_count: int = 0           # 修订轮次（防止死循环）
    is_approved: bool = False         # 是否通过审核
    needs_revision: bool = False      # 是否需要修改（Writer 用）
    history: list[AgentMessage] = field(default_factory=list)  # 消息历史
    logs: list[str] = field(default_factory=list)             # 运行日志

    def log(self, message: str) -> None:
        """记录一条日志，带时间戳"""
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def add_message(self, sender: str, receiver: str, content: str) -> None:
        """记录 Agent 之间的消息"""
        self.history.append(AgentMessage(sender, receiver, content))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mock LLM —— 模拟 AI 的"思考"，无需真实 API
# ═══════════════════════════════════════════════════════════════════════════════

class MockLLM:
    """
    Mock LLM：用预定义的规则和模板模拟 AI 的回复。

    真实系统中，这里会调用 Claude / GPT / 通义千问 等模型。
    我们用 Mock 的方式，让你在没有 API Key 的情况下也能运行完整流程。
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def generate(self, prompt: str, context: TaskState) -> str:
        """根据 Agent 角色和当前状态，生成模拟回复"""
        return self._route_by_agent(context)

    def _route_by_agent(self, state: TaskState) -> str:
        """根据当前 Agent 角色返回对应的模拟输出"""
        if self.agent_name == "coordinator":
            return self._coordinator_decision(state)
        elif self.agent_name == "researcher":
            return self._researcher_output(state)
        elif self.agent_name == "writer":
            return self._writer_output(state)
        elif self.agent_name == "reviewer":
            return self._reviewer_output(state)
        return "FINISH"

    def _coordinator_decision(self, state: TaskState) -> str:
        """调度员：根据当前进度决定下一步"""
        if not state.subtasks:
            # 第一步：拆分任务
            return "SPLIT_TASK"
        elif not state.research:
            # 第二步：让研究员调研
            return "CALL_RESEARCHER"
        elif not state.draft:
            # 第三步：让写手写第 1 稿
            return "CALL_WRITER"
        elif not state.review:
            # 第四步：让审核员检查第 1 稿
            return "CALL_REVIEWER"
        elif state.is_approved:
            # 审核通过，结束任务
            return "FINISH"
        elif state.revision_count >= 2:
            # 修改次数已达上限，强制结束
            return "FINISH"
        elif state.needs_revision:
            # 审核未通过，需要 Writer 修改
            return "CALL_WRITER"
        else:
            # 审核未通过，标记需要修改，下次进入 CALL_WRITER
            state.needs_revision = True
            return "CALL_WRITER"

    def _researcher_output(self, state: TaskState) -> str:
        """研究员：模拟调研输出"""
        return textwrap.dedent(f"""\
            【调研报告】关于「{state.task}」

            1. 背景信息：
               - 该主题涉及多个技术领域，需要综合考量
               - 当前主流方案有 A、B、C 三种，各有利弊

            2. 关键数据：
               - 方案 A：实现简单，但性能一般
               - 方案 B：性能优秀，但学习曲线陡峭
               - 方案 C：平衡选择，社区支持最好

            3. 推荐方案：
               建议采用方案 C，原因：生态成熟、文档完善、维护成本低。

            4. 参考资料：
               - 官方文档 v2.3
               - 社区最佳实践指南
               - 性能基准测试报告 2024
            """)

    def _writer_output(self, state: TaskState) -> str:
        """写手：模拟写作输出"""
        revision_note = "（第 1 稿）" if state.revision_count == 0 else f"（第 {state.revision_count + 1} 轮修订）"
        review_feedback = ""
        if state.review and not state.is_approved:
            review_feedback = f"\n            【根据审核意见修改】已针对以下问题优化：{state.review[:80]}..."

        return textwrap.dedent(f"""\
            {revision_note}

            # {state.task}

            ## 一、概述
            本文将详细介绍 {state.task} 的核心概念、实现方案及最佳实践。

            ## 二、调研结论
            {state.research[:150] if state.research else '（待补充调研内容）'}

            ## 三、推荐方案
            基于调研结果，建议采用平衡型方案（方案 C），兼顾性能与易用性。

            ## 四、实施步骤
            1. 环境准备与依赖安装
            2. 核心模块开发
            3. 集成测试与验证
            4. 文档编写与交付
            {review_feedback}

            ## 五、总结
            本方案经过多轮评估，具备可行性，建议进入实施阶段。
            """)

    def _reviewer_output(self, state: TaskState) -> str:
        """审核员：模拟审核输出"""
        # 注意：revision_count 在 Writer 修改时已经增加
        # 第 1 稿（revision_count=0）→ 提出修改意见
        # 第 2 稿（revision_count=1）→ 通过审核
        if state.revision_count >= 1:
            return textwrap.dedent("""\
                APPROVED

                审核意见：
                1. 内容结构清晰，逻辑完整
                2. 修改后的版本已解决之前提出的问题
                3. 建议可以进入最终交付阶段

                结论：✅ 审核通过
                """)
        else:
            # 第一次审核，提出修改意见
            return textwrap.dedent("""\
                NEEDS_REVISION

                审核意见：
                1. [内容] 概述部分过于简略，建议补充背景说明
                2. [结构] "实施步骤"章节缺少具体时间节点
                3. [细节] 推荐方案缺少量化对比数据
                4. [格式] 建议增加图表辅助说明

                结论：❌ 需要修改，请 Writer 根据以上意见优化后重新提交。
                """)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Agent 基类 —— 所有 Agent 的通用框架
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    Agent 基类：定义所有 Agent 的通用行为。

    每个 Agent 的核心逻辑：
      1. 接收输入（来自 Coordinator 或其他 Agent）
      2. 思考（调用 Mock LLM 生成回复）
      3. 执行（根据回复更新全局状态）
      4. 汇报（把结果返回给 Coordinator）
    """

    def __init__(self, name: str):
        self.name = name
        self.llm = MockLLM(name)

    def run(self, state: TaskState) -> TaskState:
        """运行 Agent，返回更新后的状态"""
        raise NotImplementedError("子类必须实现 run 方法")

    def _think(self, state: TaskState) -> str:
        """思考：调用 LLM 生成决策/内容"""
        # 模拟网络延迟，让输出更有真实感
        time.sleep(0.3)
        return self.llm.generate("", state)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 具体 Agent 实现
# ═══════════════════════════════════════════════════════════════════════════════

class CoordinatorAgent(BaseAgent):
    """
    调度员 Agent —— 多 Agent 系统的"大脑"。

    职责：
    1. 接收用户任务
    2. 拆分任务为子任务
    3. 根据当前状态，决定下一步调用哪个 Worker
    4. 汇总结果，输出最终答案

    类比：像一个项目经理，不亲自干活，但负责分配任务和把控进度。
    """

    def __init__(self):
        super().__init__("coordinator")

    def run(self, state: TaskState) -> TaskState:
        decision = self._think(state)  # 分析任务状态，做出下一步决策
        state.log(f"🎯 Coordinator 决策：{decision}")  # 记录决策日志
        state.add_message("coordinator", "system", f"决策: {decision}")  # 更新消息历史
        return state  # 返回更新后的状态

    def get_next_action(self, state: TaskState) -> str:
        """获取下一个要执行的动作"""
        return self._think(state)


class ResearcherAgent(BaseAgent):
    """
    研究员 Agent —— 负责信息收集和调研。

    职责：
    1. 根据任务描述收集相关背景知识
    2. 整理技术方案和最佳实践
    3. 输出结构化的调研报告

    类比：像公司里的技术调研工程师，专门负责查资料、做对比、出报告。
    """

    def __init__(self):
        super().__init__("researcher")

    def run(self, state: TaskState) -> TaskState:
        state.log("🔍 Researcher 开始调研...")
        research_result = self._think(state)
        state.research = research_result
        state.log(f"🔍 Researcher 调研完成，输出 {len(research_result)} 字")
        state.add_message("researcher", "coordinator", research_result[:100] + "...")
        return state


class WriterAgent(BaseAgent):
    """
    写手 Agent —— 负责内容产出。

    职责：
    1. 根据调研结果撰写内容
    2. 根据审核意见修改内容
    3. 输出符合要求的文档

    类比：像技术文档工程师，把调研结果写成可读的文章。
    """

    def __init__(self):
        super().__init__("writer")

    def run(self, state: TaskState) -> TaskState:
        # 判断是否是修改模式：needs_revision 为 True 表示审核未通过需要修改
        if state.needs_revision:
            state.log("✏️ Writer 根据审核意见修改中...")
            state.revision_count += 1
        else:
            state.log("✏️ Writer 开始写作...")

        draft = self._think(state)
        state.draft = draft

        # 如果是修改模式，修改完成后清空 review 和 needs_revision
        # 这样 Coordinator 下次会重新调用 Reviewer
        if state.needs_revision:
            state.review = ""
            state.needs_revision = False
            state.log(f"✏️ Writer 修改完成（第 {state.revision_count} 轮），输出 {len(draft)} 字")
        else:
            state.log(f"✏️ Writer 写作完成，输出 {len(draft)} 字")

        state.add_message("writer", "coordinator", draft[:100] + "...")
        return state


class ReviewerAgent(BaseAgent):
    """
    审核员 Agent —— 负责质量检查。

    职责：
    1. 检查内容的正确性、完整性
    2. 提出修改意见
    3. 决定是否通过审核

    类比：像技术评审专家，把关最终质量。
    """

    def __init__(self):
        super().__init__("reviewer")

    def run(self, state: TaskState) -> TaskState:
        state.log("👀 Reviewer 开始审核...")
        review_result = self._think(state)
        state.review = review_result
        state.is_approved = review_result.strip().upper().startswith("APPROVED")
        status = "✅ 通过" if state.is_approved else "❌ 需修改"
        state.log(f"👀 Reviewer 审核完成 → {status}")
        state.add_message("reviewer", "coordinator", review_result[:100] + "...")
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 多 Agent 系统 —— 编排所有 Agent 的协作
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentSystem:
    """
    多 Agent 系统 —— 负责编排 Coordinator 和 Workers 的协作。

    核心方法 run() 的执行流程：
      Step 1: Coordinator 拆分任务
      Step 2: Coordinator 决定调用 Researcher
      Step 3: Researcher 执行调研 → 回报 Coordinator
      Step 4: Coordinator 决定调用 Writer
      Step 5: Writer 写作 → 回报 Coordinator
      Step 6: Coordinator 决定调用 Reviewer
      Step 7: Reviewer 审核 → 回报 Coordinator
      Step 8: Coordinator 判断是否 FINISH
              - 如果通过 → 结束
              - 如果没通过 → 回到 Step 5（修改）
    """

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def run(self, task: str) -> TaskState:
        """
        运行完整的多 Agent 协作流程。

        Args:
            task: 用户输入的任务描述

        Returns:
            最终状态，包含所有 Agent 的产出
        """
        # 初始化全局状态
        state = TaskState(task=task)
        state.log(f"🚀 任务启动：{task}")
        print(f"\n{'='*60}")
        print("🚀 多 Agent 协作系统启动")
        print(f"{'='*60}")
        print(f"📋 任务：{task}\n")

        step = 0
        max_steps = 10  # 安全上限，防止死循环

        while step < max_steps:
            step += 1
            print(f"\n{'─'*60}")
            print(f"📌 Step {step}")
            print(f"{'─'*60}")

            # ── Coordinator 做决策 ──
            self.coordinator.run(state)
            action = self.coordinator.get_next_action(state)

            if action == "SPLIT_TASK":
                state.subtasks = ["调研背景信息", "撰写技术方案", "审核质量"]
                state.log(f"📋 Coordinator 拆分任务为 {len(state.subtasks)} 个子任务")
                print(f"\n📋 任务拆分：")
                for i, st in enumerate(state.subtasks, 1):
                    print(f"   {i}. {st}")
                continue

            elif action == "CALL_RESEARCHER":
                print(f"\n🔄 Coordinator → Researcher")
                self.researcher.run(state)
                print(f"\n📄 调研结果（前 200 字）：")
                print(f"   {state.research[:200]}...")
                continue

            elif action == "CALL_WRITER":
                print(f"\n🔄 Coordinator → Writer")
                self.writer.run(state)
                print(f"\n📄 草稿（前 200 字）：")
                print(f"   {state.draft[:200]}...")
                continue

            elif action == "CALL_REVIEWER":
                print(f"\n🔄 Coordinator → Reviewer")
                self.reviewer.run(state)
                print(f"\n📄 审核意见（前 200 字）：")
                print(f"   {state.review[:200]}...")
                continue

            elif action == "FINISH":
                state.log("🏁 Coordinator 决定结束任务")
                print(f"\n{'='*60}")
                print("🏁 任务完成！")
                print(f"{'='*60}")
                break

            else:
                state.log(f"⚠️ 未知决策：{action}，强制结束")
                break

        return state


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 运行演示
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_report(state: TaskState) -> None:
    """打印最终报告"""
    print(f"\n{'='*60}")
    print("📊 最终产出")
    print(f"{'='*60}")
    print(f"\n{state.draft}")

    print(f"\n{'='*60}")
    print("📊 运行统计")
    print(f"{'='*60}")
    print(f"   总步数：{len(state.history)}")
    print(f"   修订轮次：{state.revision_count}")
    print(f"   审核结果：{'✅ 通过' if state.is_approved else '❌ 未通过'}")
    print(f"   消息数量：{len(state.history)}")

    print(f"\n{'='*60}")
    print("📝 完整运行日志")
    print(f"{'='*60}")
    for log in state.logs:
        print(f"   {log}")

    print(f"\n{'='*60}")
    print("💡 系统架构说明")
    print(f"{'='*60}")
    print(textwrap.dedent("""\
    本系统包含 4 个 Agent：

    1. Coordinator（调度员）
       - 角色：项目经理
       - 职责：拆分任务、决定下一步调用谁
       - 特点：不亲自干活，只负责分配和决策

    2. Researcher（研究员）
       - 角色：技术调研工程师
       - 职责：收集信息、做调研、出报告
       - 特点：专门负责"查资料"

    3. Writer（写手）
       - 角色：技术文档工程师
       - 职责：把调研结果写成文档
       - 特点：根据反馈反复修改

    4. Reviewer（审核员）
       - 角色：技术评审专家
       - 职责：检查质量、提出修改意见
       - 特点：把关最终质量

    协作流程：
      用户任务 → Coordinator 拆分
                → Researcher 调研 → 回报 Coordinator
                → Writer 写作 → 回报 Coordinator
                → Reviewer 审核 → 回报 Coordinator
                → Coordinator 判断是否继续修改或结束

    关键概念：
      - 共享状态（Shared State）：所有 Agent 读写同一份数据
      - 消息传递（Message Passing）：Agent 之间通过消息沟通
      - 路由决策（Routing）：Coordinator 决定下一步走哪条路
      - 迭代改进（Iteration）：审核不通过时循环修改
    """))


def main():
    """主入口"""
    # 用户任务
    task = "帮我写一份关于「如何设计一个高并发缓存系统」的技术方案"

    # 创建并运行多 Agent 系统
    system = MultiAgentSystem()
    final_state = system.run(task)

    # 打印最终报告
    print_final_report(final_state)

    print(f"\n{'='*60}")
    print("✅ 演示结束！")
    print(f"{'='*60}")
    print("\n💡 你可以修改 main() 里的 task 变量，让系统处理不同的任务。")
    print("💡 也可以修改 MockLLM 里的输出模板，自定义每个 Agent 的行为。")


if __name__ == "__main__":
    main()
