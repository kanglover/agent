"""
06 - 多节点工作流：编排复杂任务
================================
【这章学什么】
  - 把多个节点串成完整工作流（综合运用前 5 章）
  - 并行执行（Map-Reduce：多个节点同时跑，再汇总）
  - 审校循环（不合格打回重做）

【为什么学它】
  前面学的都是单个技巧。真实项目要拼起来——比如做一个"研究报告生成器"：
  调研 → 写作 → 审校，审校不过就打回重写。
  或者"多角度总结"：让 3 个节点同时从不同角度写，再汇总成一篇。

  LangGraph 的价值就在这：你只管画好"流程图"（谁接谁、哪里分叉、哪里循环），
  执行、并行、状态传递它全包。本章把前面学的 State/节点/边/条件/循环 全用上。

【类比】
  像搭一条生产线：
  - 串行 = 流水线一道接一道（调研→写作→审校）
  - 并行 = 一条传送带分成3条同时加工，最后合流（3个角度同时写→汇总）
  - 循环 = 质检不合格退回上一道工序

【运行】
  cd code
  uv run python learn_langgraph/06_workflow.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END


# ============================================================
# 示例一：串行工作流 + 审校循环
# ============================================================
# 研究报告：调研要点 → 写正文 → 审校（不合格打回重写，最多3次）
print("=" * 60)
print("示例一：研究报告工作流（调研→写作→审校循环）")
print("=" * 60)


class ReportState(TypedDict):
    topic: str
    outline: str       # 调研要点
    content: str       # 正文
    review: str        # 审校意见
    attempts: int      # 重试次数


def research(state: ReportState) -> dict:
    """调研：列出要点。"""
    print(f"  [调研] 收集「{state['topic']}」的要点")
    outline = llm.invoke(f"列出关于 {state['topic']} 的3个核心要点，每个一行。").content
    return {"outline": outline, "attempts": state["attempts"] + 1}


def write(state: ReportState) -> dict:
    """写作：根据要点写正文。"""
    print(f"  [写作] 第{state['attempts']}次写作")
    content = llm.invoke(f"根据以下要点写一段100字以内的正文：\n{state['outline']}").content
    return {"content": content}


def review(state: ReportState) -> dict:
    """审校：判断正文质量。"""
    content = state["content"]
    # 简单规则演示：正文超过 30 字就算通过（实际可让模型审校）
    passed = len(content) > 30 and state["attempts"] < 3  # 超过3次也放行
    opinion = "通过" if passed else "太短，重写"
    print(f"  [审校] {opinion}（正文 {len(content)} 字）")
    return {"review": opinion}


def after_review(state: ReportState) -> str:
    """审校后路由：通过→结束，没过→回写作。"""
    if "通过" in state["review"]:
        return "end"
    return "rewrite"


b1 = StateGraph(ReportState)
b1.add_node("research", research)
b1.add_node("write", write)
b1.add_node("review", review)

b1.add_edge(START, "research")
b1.add_edge("research", "write")
b1.add_edge("write", "review")
b1.add_conditional_edges(
    "review", after_review, {"end": END, "rewrite": "write"}
)

workflow1 = b1.compile()

r = workflow1.invoke({"topic": "人工智能", "outline": "", "content": "", "review": "", "attempts": 0})
print(f"\n📝 最终报告：{r['content'][:80]}")


# ============================================================
# 示例二：并行 Map-Reduce（多个节点同时跑，再汇总）
# ============================================================
# 一个主题，让3个节点分别从"技术/应用/风险"三个角度写，最后汇总成一篇。
# 关键：条件边返回【多个节点名】实现并行 fan-out；用 operator.add 收集各节点结果。
print("\n" + "=" * 60)
print("示例二：并行 Map-Reduce（三角度同时写→汇总）")
print("=" * 60)


class SummaryState(TypedDict):
    topic: str
    # parts 用 operator.add：各并行节点返回的片段会自动拼到一起
    parts: Annotated[list[str], operator.add]
    final: str


def write_tech(state: SummaryState) -> dict:
    print("  [技术角度] 生成中...")
    t = llm.invoke(f"从技术角度用一句话说 {state['topic']}").content
    return {"parts": [f"【技术】{t}"]}


def write_app(state: SummaryState) -> dict:
    print("  [应用角度] 生成中...")
    t = llm.invoke(f"从应用角度用一句话说 {state['topic']}").content
    return {"parts": [f"【应用】{t}"]}


def write_risk(state: SummaryState) -> dict:
    print("  [风险角度] 生成中...")
    t = llm.invoke(f"从风险角度用一句话说 {state['topic']}").content
    return {"parts": [f"【风险】{t}"]}


def assemble(state: SummaryState) -> dict:
    """汇总：把三段拼成最终总结。"""
    print("  [汇总] 合并三角度")
    combined = "\n".join(state["parts"])
    final = llm.invoke(f"把以下三段融合成一段流畅的总结：\n{combined}").content
    return {"final": final}


b2 = StateGraph(SummaryState)
b2.add_node("write_tech", write_tech)
b2.add_node("write_app", write_app)
b2.add_node("write_risk", write_risk)
b2.add_node("assemble", assemble)

b2.add_edge(START, "write_tech")
b2.add_edge(START, "write_app")     # 同一个 START 连三个节点 = 并行
b2.add_edge(START, "write_risk")
# 三个写节点都汇入 assemble（assemble 会等三个都完成才执行）
b2.add_edge("write_tech", "assemble")
b2.add_edge("write_app", "assemble")
b2.add_edge("write_risk", "assemble")
b2.add_edge("assemble", END)

workflow2 = b2.compile()

r2 = workflow2.invoke({"topic": "大语言模型", "parts": [], "final": ""})
print(f"\n📋 三角度汇总：\n{r2['final']}")


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. 工作流 = 多节点 + 边 + 条件 + 循环 的组合应用")
print("2. 串行：add_edge 一道接一道；审校循环用条件边指回写作节点")
print("3. 并行：一个 START 连多个节点 = fan-out；多个节点连同一节点 = fan-in（等齐才执行）")
print("4. 用 Annotated[list, operator.add] 自动合并各并行节点的结果")
print("5. 复杂任务先画流程图（纸笔），再翻译成 add_node/add_edge")
print()
print("🎉 learn_langgraph 6 篇全部学完！你已掌握 LangGraph 核心。")
print("回顾路线：State/节点/边 → 条件循环 → 工具Agent → 记忆 → 人机协作 → 工作流")
print("=" * 60)
