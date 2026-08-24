# agent_harness/harness.py
"""
Harness 核心 —— "考场总控"

类比：
  这是整个考试的控制中心，负责：
  1. 发卷（把任务交给 Agent）
  2. 计时（记录每个 Agent 花了多久）
  3. 收卷（收集所有 Agent 的结果）
  4. 阅卷（调用 Evaluator 评分）
  5. 公布成绩（生成对比报告）

核心流程：
  Harness
    ├── 注册 Agent（add_agent）
    ├── 加载任务（add_tasks）
    ├── 运行测评（run）
    │     ├── 对每个 Agent：
    │     │     ├── setup()
    │     │     ├── 逐题 run()
    │     │     └── teardown()
    │     └── 收集所有 AgentResult
    ├── 评分（evaluate）
    └── 生成报告（report）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from agent_harness.protocol import AgentProtocol, AgentResult
from agent_harness.tasks import Task, TASK_SUITE
from agent_harness.evaluator import Evaluator, EvalResult
from agent_harness.tools import TOOL_DEFINITIONS


# ── 数据结构：单个 Agent 的完整测评记录 ────────────────────────

@dataclass
class AgentRunRecord:
    """一个 Agent 做完所有题后的完整记录"""
    agent_name: str
    results: list[AgentResult] = field(default_factory=list)
    eval_results: list[EvalResult] = field(default_factory=list)
    total_score: float = 0.0
    max_possible_score: float = 0.0
    total_duration_ms: float = 0.0

    @property
    def score_pct(self) -> float:
        if self.max_possible_score == 0:
            return 0.0
        return round(self.total_score / self.max_possible_score * 100, 1)

    @property
    def avg_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(self.total_duration_ms / len(self.results), 1)


# ── Harness 主类 ──────────────────────────────────────────────

class Harness:
    """
    Agent 测评框架的核心调度器

    使用方式：
        harness = Harness()
        harness.add_agent(my_agent_1)
        harness.add_agent(my_agent_2)
        harness.add_tasks(TASK_SUITE)    # 或自定义任务列表
        records = harness.run()          # 运行测评
        harness.print_report(records)    # 打印报告
    """

    def __init__(self, evaluator: Evaluator | None = None):
        self._agents: list[AgentProtocol] = []
        self._tasks: list[Task] = []
        self._evaluator = evaluator or Evaluator()

    # ── 配置阶段 ──

    def add_agent(self, agent: AgentProtocol) -> "Harness":
        """注册一个 Agent（支持链式调用）"""
        self._agents.append(agent)
        return self

    def add_tasks(self, tasks: list[Task]) -> "Harness":
        """加载任务列表（支持链式调用）"""
        self._tasks.extend(tasks)
        return self

    def use_default_tasks(self) -> "Harness":
        """使用内置的默认测试套件"""
        self._tasks = list(TASK_SUITE)
        return self

    # ── 运行阶段 ──

    def _run_single_agent(self, agent: AgentProtocol) -> AgentRunRecord:
        """
        让一个 Agent 做完所有题

        流程：
          1. 遍历每道题
          2. 调用 agent.run_with_lifecycle()（自动带 setup/teardown/计时）
          3. 调用 evaluator.evaluate() 评分
          4. 汇总成 AgentRunRecord
        """
        record = AgentRunRecord(agent_name=agent.name)

        for task in self._tasks:
            # 让 Agent 做题（自动带生命周期管理）
            result = agent.run_with_lifecycle(
                task=task.task,
                tools=TOOL_DEFINITIONS,
                context=task.context,
            )
            record.results.append(result)

            # 阅卷打分
            eval_result = self._evaluator.evaluate(task, result)
            record.eval_results.append(eval_result)

            # 累计统计
            record.total_score += eval_result.score
            record.max_possible_score += eval_result.max_score
            record.total_duration_ms += result.duration_ms

        return record

    def run(self) -> list[AgentRunRecord]:
        """
        运行完整测评：所有 Agent × 所有任务

        Returns:
            每个 Agent 的完整测评记录列表
        """
        if not self._agents:
            raise RuntimeError("没有注册任何 Agent，请先调用 add_agent()")
        if not self._tasks:
            raise RuntimeError("没有加载任何任务，请先调用 add_tasks() 或 use_default_tasks()")

        print(f"\n{'=' * 60}")
        print(f"  🏁 Agent Harness 测评开始")
        print(f"  📋 共 {len(self._agents)} 个 Agent × {len(self._tasks)} 道题")
        print(f"  ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        records: list[AgentRunRecord] = []

        for i, agent in enumerate(self._agents, 1):
            print(f"\n{'─' * 50}")
            print(f"  🤖 Agent {i}/{len(self._agents)}: {agent.name}")
            print(f"{'─' * 50}")

            record = self._run_single_agent(agent)
            records.append(record)

            # 每个 Agent 完成后打印简要结果
            for j, (task, eval_r) in enumerate(
                zip(self._tasks, record.eval_results), 1
            ):
                status = "✅" if eval_r.score > 0 else "❌"
                print(
                    f"  {status} [{task.id}] {task.task[:30]}... "
                    f"→ {eval_r.score}/{eval_r.max_score} 分"
                )

            print(f"\n  📊 总分: {record.total_score}/{record.max_possible_score} "
                  f"({record.score_pct}%)")

        return records

    # ── 报告阶段 ──

    def print_report(self, records: list[AgentRunRecord]) -> None:
        """
        打印完整的对比报告

        包含：
          - 各 Agent 总分排名
          - 各维度得分明细
          - 详细评语
        """
        print(f"\n{'=' * 60}")
        print(f"  📊 Agent Harness 测评报告")
        print(f"  ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        # ── 排行榜 ──
        sorted_records = sorted(records, key=lambda r: r.total_score, reverse=True)

        print(f"\n🏆 排行榜:")
        print(f"{'─' * 50}")
        for rank, record in enumerate(sorted_records, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            print(
                f"  {medal} #{rank} {record.agent_name:<20} "
                f"{record.total_score:>6.1f}/{record.max_possible_score:.0f} "
                f"({record.score_pct}%) "
                f"  ⏱️ {record.avg_duration_ms:.0f}ms/题"
            )

        # ── 详细评分 ──
        for record in records:
            print(f"\n{'─' * 50}")
            print(f"  📝 {record.agent_name} 详细评分")
            print(f"{'─' * 50}")

            for task, eval_r in zip(self._tasks, record.eval_results):
                print(f"\n  [{task.id}] {task.task[:40]}...")
                print(f"    得分: {eval_r.score}/{eval_r.max_score} ({eval_r.score_pct}%)")
                print(f"    维度: {eval_r.breakdown}")
                for fb in eval_r.feedback:
                    print(f"    {fb}")

        print(f"\n{'=' * 60}")
        print(f"  ✅ 测评报告完毕")
        print(f"{'=' * 60}\n")
