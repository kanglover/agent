"""
01 - LangGraph 基础：State、节点、边，画出第一张图
====================================================
【这章学什么】
  - LangGraph 是什么、跟 LangChain 什么关系
  - 三个核心概念：State（状态）、Node（节点）、Edge（边）
  - 画出并运行第一张图

【为什么学它】
  前面学的 LangChain 链（LCEL）是"一条直线"。但真实任务常常有分支、循环、
  多步骤协作——比如"先判断问题类型，数学走计算、闲聊走回答，必要时反复查资料"。
  这种"流程图"式的复杂逻辑，用 LangChain 的 | 管道写起来很别扭，LangGraph 就是来解决这个的。

  LangGraph 让你像画流程图一样编排 AI 工作流：每个处理步骤是一个节点，
  步骤之间用边连接，所有节点共享一个"状态"白板。

【类比】
  - LangChain 链 = 单行道（一条直线走到底）
  - LangGraph 图 = 城市路网（有分岔、有环岛、能绕路）
  State 就是路网里所有路口共享的"公告板"，每个路口（节点）经过时都能看和改。

【三件套】
  1. State（状态）：一个 TypedDict，定义"公告板"上有哪些字段
  2. Node（节点）：一个普通函数，读 state、返回要更新的字段
  3. Edge（边）：决定执行完一个节点后去哪个节点

【运行】
  cd code
  uv run python learn_langgraph/01_basics.py
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


# ============================================================
# 第1步：定义 State（状态白板）
# ============================================================
# State 用 TypedDict 定义：规定白板上有哪几个字段、什么类型。
# 所有节点都能读这些字段、也能返回新值更新它们。
from typing import TypedDict


class State(TypedDict):
    topic: str        # 用户给的主题
    draft: str        # 写出的初稿
    final: str        # 最终结果


# ============================================================
# 第2步：定义节点（普通函数）
# ============================================================
# 节点函数的约定：接收 state（字典），返回一个【只含要更新字段】的字典。
# 不需要返回整个 state，LangGraph 会帮你合并。

def write_draft(state: State) -> dict:
    """节点1：根据主题写初稿。"""
    topic = state["topic"]
    print(f"  [节点:写初稿] 主题是「{topic}」")
    # 调模型写一段话
    draft = llm.invoke(f"用一句话介绍 {topic}").content
    # 返回要更新的字段
    return {"draft": draft}


def polish(state: State) -> dict:
    """节点2：把初稿润色成最终版。"""
    draft = state["draft"]
    print(f"  [节点:润色] 初稿是「{draft}」")
    final = llm.invoke(f"把这句话改得更生动：{draft}").content
    return {"final": final}


# ============================================================
# 第3步：把节点和边连成图
# ============================================================
from langgraph.graph import StateGraph, START, END

# 创建图的"设计图"（还没编译，不能运行）
graph_builder = StateGraph(State)

# 添加节点：起个名字 + 绑定函数
graph_builder.add_node("write_draft", write_draft)
graph_builder.add_node("polish", polish)

# 添加边：决定执行顺序
#   START → write_draft → polish → END
graph_builder.add_edge(START, "write_draft")   # 从起点进 write_draft
graph_builder.add_edge("write_draft", "polish")  # write_draft 完了去 polish
graph_builder.add_edge("polish", END)            # polish 完了结束

# 编译：把设计图变成可运行的图
graph = graph_builder.compile()


# ============================================================
# 第4步：运行图
# ============================================================
print("=" * 60)
print("运行第一张图：写初稿 → 润色")
print("=" * 60)

# invoke 传入初始 state（必须包含 State 定义的字段）
result = graph.invoke({"topic": "人工智能", "draft": "", "final": ""})

print("\n--- 最终状态 ---")
print("主题：", result["topic"])
print("初稿：", result["draft"])
print("终稿：", result["final"])


# ============================================================
# 第5步：用 stream 看每一步（调试利器）
# ============================================================
# invoke 只给最终结果；stream 一步步吐出每个节点执行后的状态，调试时很有用。
print("\n" + "=" * 60)
print("用 stream 看每一步")
print("=" * 60)

for chunk in graph.stream({"topic": "区块链", "draft": "", "final": ""}):
    # 每个 chunk 是 {节点名: 该节点返回的更新}
    for node_name, update in chunk.items():
        print(f"  步骤 [{node_name}] 更新了: {update}")


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. State(TypedDict) = 共享白板；Node(函数) = 处理步骤；Edge = 步骤顺序")
print("2. 节点函数：读 state，返回【要更新的字段】的字典（不用返回全部）")
print("3. 流程：add_node 加节点 → add_edge 连边 → compile 编译 → invoke 运行")
print("4. stream() 一步步看执行过程，调试必备")
print("5. 下一章：条件边——根据情况走不同分支（不再只是直线）")
print("=" * 60)
