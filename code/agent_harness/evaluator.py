# agent_harness/evaluator.py
"""
评估器 —— Harness 的"评分标准"

类比：
  阅卷老师根据评分细则给每份答卷打分。
  这里的"评分细则"有两种：
  1. 关键词匹配（自动判卷）：答案里有没有包含必要的关键词
  2. 工具使用检查（过程分）：有没有正确使用指定的工具

设计思路：
  - EvalResult: 单道题的评分结果
  - Evaluator:  评估器，可以对单个或多个结果打分
  - 评分维度是可扩展的，目前包含：
    · 完成度（是否成功）
    · 关键词覆盖率（答案质量）
    · 工具使用正确性（方法是否对）
    · 效率（步骤数是否合理）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_harness.protocol import AgentResult
from agent_harness.tasks import Task


@dataclass
class EvalResult:
    """
    单道题的评分结果

    Attributes:
        task_id:           对应的任务 ID
        score:             最终得分
        max_score:         满分
        breakdown:         各维度得分明细
        feedback:          评语（给人看的文字说明）
    """
    task_id: str
    score: float
    max_score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        """得分率（百分比）"""
        if self.max_score == 0:
            return 0.0
        return round(self.score / self.max_score * 100, 1)


class Evaluator:
    """
    评估器——按多个维度给 Agent 的表现打分

    评分维度和权重：
      - 完成度 (completion):  40%  — 任务是否成功完成
      - 关键词 (keywords):    30%  — 答案是否覆盖了关键内容
      - 工具使用 (tool_use):  20%  — 是否正确使用了必要工具
      - 效率 (efficiency):    10%  — 步骤数是否合理
    """

    # 权重配置（加起来 = 1.0）
    WEIGHTS = {
        "completion": 0.40,
        "keywords":   0.30,
        "tool_use":   0.20,
        "efficiency": 0.10,
    }

    # 效率基准：认为 5 步以内算高效
    EFFICIENT_STEP_LIMIT = 5

    def evaluate(self, task: Task, result: AgentResult) -> EvalResult:
        """
        对一道题的 Agent 结果进行评分

        Args:
            task:   任务定义（"题目"）
            result: Agent 的执行结果（"答卷"）

        Returns:
            EvalResult: 评分结果
        """
        breakdown: dict[str, float] = {}
        feedback: list[str] = []

        # ── 维度 1：完成度（成功就满分，失败就 0）──
        if result.success:
            breakdown["completion"] = 1.0
            feedback.append("✅ 任务完成")
        else:
            breakdown["completion"] = 0.0
            error_msg = result.error or "未知原因"
            feedback.append(f"❌ 任务失败: {error_msg}")

        # ── 维度 2：关键词覆盖率 ──
        if task.expected_keywords and result.output:
            output_lower = result.output.lower()
            matched = [kw for kw in task.expected_keywords if kw.lower() in output_lower]
            coverage = len(matched) / len(task.expected_keywords)
            breakdown["keywords"] = coverage

            if coverage == 1.0:
                feedback.append(f"✅ 关键词全部命中 ({len(matched)}/{len(task.expected_keywords)})")
            elif coverage > 0:
                missed = [kw for kw in task.expected_keywords if kw.lower() not in output_lower]
                feedback.append(
                    f"⚠️ 关键词部分命中 ({len(matched)}/{len(task.expected_keywords)})，"
                    f"缺少: {', '.join(missed)}"
                )
            else:
                feedback.append("❌ 关键词全部未命中")
        else:
            # 没有关键词要求或没有输出，给个基础分
            breakdown["keywords"] = 0.5 if result.success else 0.0
            feedback.append("ℹ️ 无关键词评分标准")

        # ── 维度 3：工具使用正确性 ──
        if task.tools_required:
            used_tools = {tc.tool_name for tc in result.tool_calls}
            required = set(task.tools_required)
            covered = required & used_tools
            tool_score = len(covered) / len(required) if required else 1.0
            breakdown["tool_use"] = tool_score

            if covered == required:
                feedback.append(f"✅ 正确使用了所有必要工具: {', '.join(required)}")
            else:
                missed_tools = required - covered
                feedback.append(
                    f"⚠️ 缺少工具调用: {', '.join(missed_tools)}，"
                    f"已使用: {', '.join(used_tools) if used_tools else '无'}"
                )
        else:
            breakdown["tool_use"] = 1.0 if result.success else 0.0
            feedback.append("ℹ️ 无工具使用要求")

        # ── 维度 4：效率 ──
        if result.steps <= self.EFFICIENT_STEP_LIMIT:
            breakdown["efficiency"] = 1.0
            feedback.append(f"✅ 高效完成 ({result.steps} 步)")
        else:
            # 超出基准的步骤越多，分越低，最低 0.2
            over = result.steps - self.EFFICIENT_STEP_LIMIT
            breakdown["efficiency"] = max(0.2, 1.0 - over * 0.15)
            feedback.append(f"⚠️ 步骤较多 ({result.steps} 步，基准 {self.EFFICIENT_STEP_LIMIT} 步)")

        # ── 加权汇总 ──
        weighted_score = sum(
            breakdown[dim] * self.WEIGHTS[dim]
            for dim in self.WEIGHTS
        )
        final_score = round(weighted_score * task.max_score, 2)

        return EvalResult(
            task_id=task.id,
            score=final_score,
            max_score=task.max_score,
            breakdown={dim: round(val, 3) for dim, val in breakdown.items()},
            feedback=feedback,
        )

    def evaluate_batch(
        self,
        tasks: list[Task],
        results: list[AgentResult],
    ) -> list[EvalResult]:
        """批量评分（任务列表和结果列表必须一一对应）"""
        if len(tasks) != len(results):
            raise ValueError(
                f"任务数 ({len(tasks)}) 和结果数 ({len(results)}) 不匹配"
            )
        return [self.evaluate(t, r) for t, r in zip(tasks, results)]
