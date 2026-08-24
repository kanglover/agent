# agent_harness/tasks.py
"""
任务定义 —— Harness 的"考卷"

类比：
  一张考卷包含多道题，每道题有：
  - 题目内容（task）
  - 题型标签（category）
  - 参考答案要点（expected_keywords）
  - 评分标准（max_score）

设计思路：
  Task 用 dataclass 定义，保证结构统一。
  TASK_SUITE 是一组预定义的测试任务，可以直接用。
  你也可以自定义任务，只要符合 Task 格式就行。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    """
    一道"考题"

    Attributes:
        id:         唯一标识（如 "task_001"）
        task:       任务描述（自然语言，交给 Agent 执行）
        category:   题型分类（如 "搜索"、"计算"、"综合"）
        expected_keywords: 参考答案应包含的关键词（用于自动评分）
        max_score:  这道题的满分
        tools_required: 这道题需要用到哪些工具（用于验证 Agent 是否正确使用）
        context:    额外上下文（可选，传给 Agent 的附加信息）
    """
    id: str
    task: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    max_score: float = 10.0
    tools_required: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


# ── 预定义测试套件 ──────────────────────────────────────────────

TASK_SUITE: list[Task] = [
    # ── 搜索类：考察 Agent 能否正确使用搜索工具 ──
    Task(
        id="search_001",
        task="搜索知识库中关于 Agent 的内容，告诉我 Agent 的核心概念是什么。",
        category="搜索",
        expected_keywords=["observe", "think", "act", "循环"],
        tools_required=["search_knowledge"],
    ),
    Task(
        id="search_002",
        task="查找 ReAct 模式的相关知识，解释什么是 ReAct。",
        category="搜索",
        expected_keywords=["Reasoning", "Acting", "思考", "行动"],
        tools_required=["search_knowledge"],
    ),

    # ── 计算类：考察 Agent 能否正确使用计算工具 ──
    Task(
        id="calc_001",
        task="计算 (15 + 27) * 3 的结果，并解释计算过程。",
        category="计算",
        expected_keywords=["126"],
        tools_required=["calculate"],
    ),
    Task(
        id="calc_002",
        task="计算圆周率 pi 的正弦值 sin(pi)，结果应该接近多少？",
        category="计算",
        expected_keywords=["0"],
        tools_required=["calculate"],
    ),

    # ── 综合类：考察 Agent 能否组合多个工具完成任务 ──
    Task(
        id="combo_001",
        task="搜索关于 Agent 的知识，然后把搜索结果整理成一篇简短的笔记保存起来。",
        category="综合",
        expected_keywords=["Agent"],
        max_score=15.0,
        tools_required=["search_knowledge", "write_note"],
    ),
    Task(
        id="combo_002",
        task="先搜索上下文工程(context engineering)的内容，再搜索工具使用(tool use)的内容，"
             "最后写一篇对比笔记，说明两者的关系。",
        category="综合",
        expected_keywords=["上下文", "工具"],
        max_score=20.0,
        tools_required=["search_knowledge", "write_note"],
    ),
]


def get_tasks_by_category(category: str) -> list[Task]:
    """按分类筛选任务"""
    return [t for t in TASK_SUITE if t.category == category]


def get_task_by_id(task_id: str) -> Task | None:
    """按 ID 查找任务"""
    for t in TASK_SUITE:
        if t.id == task_id:
            return t
    return None
