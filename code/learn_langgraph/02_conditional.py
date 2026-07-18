"""
02 - 条件边：根据情况走不同分支
================================
【这章学什么】
  - 条件边 add_conditional_edges：让图能"分岔"
  - 用循环（边指回前面的节点）反复迭代直到满意

【为什么学它】
  上一章的图是直线：START → A → B → END。但真实任务常需要"看情况"：
  - 用户问数学 → 走计算分支；问闲聊 → 走聊天分支
  - 写出来的东西不合格 → 回去重写；合格 → 结束

  条件边就是"路口的红绿灯"：根据当前状态决定下一步去哪个节点。
  当边指回前面的节点时，就形成了"循环"——这是 LangGraph 比 LangChain 链强大的关键。

【类比】
  普通边 = 直行道（必定走到下个节点）
  条件边 = 岔路口（看路标决定走哪条，路标由你写的函数决定）
  循环   = 环岛（转一圈再出来，转几圈由条件决定）

【运行】
  cd code
  uv run python learn_langgraph/02_conditional.py
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

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 示例一：分类路由（条件分支）
# ============================================================
# 场景：用户提问 → 模型判断是"数学"还是"闲聊" → 走不同节点回答
print("=" * 60)
print("示例一：条件分支（按问题类型走不同节点）")
print("=" * 60)


class State(TypedDict):
    question: str   # 用户问题
    category: str   # 分类结果
    answer: str     # 最终回答


def classify(state: State) -> dict:
    """节点1：判断问题类型。"""
    q = state["question"]
    # 让模型只回答 "数学" 或 "闲聊"
    cat = llm.invoke(
        f"判断这个问题属于【数学】还是【闲聊】，只回答这两个词之一：{q}"
    ).content.strip()
    print(f"  [分类] 问题「{q}」→ {cat}")
    return {"category": cat}


def math_answer(state: State) -> dict:
    """节点2a：数学分支。"""
    print("  [数学分支] 走计算路线")
    ans = llm.invoke(f"你是数学老师，解答：{state['question']}。简洁。").content
    return {"answer": ans}


def chat_answer(state: State) -> dict:
    """节点2b：闲聊分支。"""
    print("  [闲聊分支] 走聊天路线")
    ans = llm.invoke(f"友好地回答：{state['question']}").content
    return {"answer": ans}


def route(state: State) -> str:
    """条件函数：根据分类结果返回下一个节点的名字。"""
    cat = state["category"]
    if "数学" in cat:
        return "math_answer"
    return "chat_answer"


# 建图
b = StateGraph(State)
b.add_node("classify", classify)
b.add_node("math_answer", math_answer)
b.add_node("chat_answer", chat_answer)

b.add_edge(START, "classify")
# 条件边：从 classify 出发，由 route 函数决定去哪
# path_map 把 route 返回的字符串映射到节点名
b.add_conditional_edges("classify", route, {"math_answer": "math_answer", "chat_answer": "chat_answer"})
b.add_edge("math_answer", END)
b.add_edge("chat_answer", END)

graph1 = b.compile()

# 跑两个不同类型的问题
for q in ["3加5等于几", "今天天气真好啊"]:
    print(f"\n问：{q}")
    r = graph1.invoke({"question": q, "category": "", "answer": ""})
    print(f"答：{r['answer'][:40]}")


# ============================================================
# 示例二：循环（反复修改直到通过检查）
# ============================================================
# 场景：写笑话 → 检查是否够短 → 不够短就重写（循环）→ 够短就结束
# 关键：检查节点用条件边，不通过就指回"写笑话"节点，形成循环
print("\n" + "=" * 60)
print("示例二：循环（写笑话 → 检查长度 → 不合格重写）")
print("=" * 60)


class JokeState(TypedDict):
    topic: str
    joke: str
    attempts: int   # 重试次数，防止无限循环


def write_joke(state: JokeState) -> dict:
    """写一个笑话。"""
    print(f"  [写笑话] 第{state['attempts']+1}次尝试")
    joke = llm.invoke(f"写一个关于{state['topic']}的笑话，一句话。").content
    return {"joke": joke, "attempts": state["attempts"] + 1}


def check_joke(state: JokeState) -> dict:
    """检查节点：这里不做修改，只占位（判断在条件边函数里做）。"""
    # 实际判断逻辑放在 route_after_check 里
    return {}


def route_after_check(state: JokeState) -> str:
    """条件函数：笑话够短(≤30字)就结束，否则回去重写。"""
    joke = state["joke"]
    print(f"  [检查] 笑话长度 {len(joke)} 字：「{joke[:30]}...」")
    # 防止无限循环：超过 3 次就强制结束
    if state["attempts"] >= 3:
        print("  [检查] 达到最大次数，结束")
        return "end"
    if len(joke) <= 30:
        print("  [检查] 够短，通过！")
        return "end"
    print("  [检查] 太长，重写")
    return "rewrite"


b2 = StateGraph(JokeState)
b2.add_node("write_joke", write_joke)
b2.add_node("check_joke", check_joke)

b2.add_edge(START, "write_joke")
b2.add_edge("write_joke", "check_joke")
# 条件边：检查后，通过→END，不通过→回 write_joke（循环！）
b2.add_conditional_edges(
    "check_joke",
    route_after_check,
    {"end": END, "rewrite": "write_joke"},
)

graph2 = b2.compile()

r = graph2.invoke({"topic": "程序员", "joke": "", "attempts": 0})
print(f"\n最终笑话：{r['joke']}")


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. add_conditional_edges：从某节点出发，由函数决定去哪个节点")
print("2. 条件函数返回节点名字符串，path_map 映射到实际节点（或 END）")
print("3. 边指回前面的节点 = 循环（反复迭代直到满足条件）")
print("4. 循环一定要加『最大次数』保护，防止无限循环")
print("5. 下一章：用条件边 + 工具节点，搭一个会调工具的 Agent")
print("=" * 60)
