"""
05 - 人机协作：关键步骤暂停，等人确认再继续
============================================
【这章学什么】
  - Human-in-the-loop（人在回路）：让图在关键步骤暂停，等人介入
  - 方式一：interrupt_before（在某个节点前暂停）
  - 方式二：interrupt()（在节点内部主动暂停，拿人工输入）
  - 用 Command 恢复执行

【为什么学它】
  自动化的 Agent 很方便，但有些步骤"搞砸了代价很大"——比如发邮件、删数据、
  执行花钱的操作。你希望 Agent 做到"准备发邮件"就停下，把草稿给你看，
  你点"同意"它才真发，点"改"它就回去重写。

  这就是人机协作：Agent 自动跑到危险/关键步骤 → 暂停 → 人审核 → 继续。
  既享受自动化，又保留人的控制权。

【类比】
  自动驾驶的"人工接管"：平时自己开，遇到复杂路况暂停提示司机，司机确认后继续。
  interrupt_before = 到某个路口必停（比如收费站前停下问"要过吗"）
  interrupt()       = 开到一半觉得不对，主动停下来打电话问老板

【前提】interrupt 必须配 checkpointer（要靠记忆记住"暂停在哪"才能恢复）

【运行】
  cd code
  uv run python learn_langgraph/05_human_in_loop.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict
import operator

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class EmailState(TypedDict):
    messages: Annotated[list, operator.add]
    draft: str       # 邮件草稿
    approved: bool   # 是否已批准


# ============================================================
# 方式一：interrupt_before（节点前暂停）
# ============================================================
# 场景：写邮件草稿 → 【暂停，等人确认】→ 发送
# 我们在"发送"节点前强制暂停，人看了草稿再决定继续不继续。
print("=" * 60)
print("方式一：interrupt_before（发送前暂停等确认）")
print("=" * 60)


def write_email(state: EmailState) -> dict:
    """写邮件草稿。"""
    topic = state["messages"][-1].content
    draft = llm.invoke(f"写一封简短的邮件草稿，主题：{topic}。直接输出正文。").content
    print(f"  [写邮件] 草稿已生成：{draft[:40]}...")
    return {"draft": draft}


def send_email(state: EmailState) -> dict:
    """发送邮件（演示用，实际会真的发）。"""
    print(f"  [发送] ✉️ 邮件已发送！内容：{state['draft'][:40]}...")
    return {"approved": True}


b1 = StateGraph(EmailState)
b1.add_node("write_email", write_email)
b1.add_node("send_email", send_email)
b1.add_edge(START, "write_email")
b1.add_edge("write_email", "send_email")
b1.add_edge("send_email", END)

# 关键：compile 时加 interrupt_before=["send_email"]
# 并必须传 checkpointer（暂停要靠它记住状态）
graph1 = b1.compile(checkpointer=MemorySaver(), interrupt_before=["send_email"])

config = {"configurable": {"thread_id": "email-1"}}

# 第一次调用：会跑到 send_email 前就停下（暂停）
print("\n第1步：让 Agent 写邮件（会在发送前暂停）")
graph1.invoke({"messages": [HumanMessage(content="请假一天")], "draft": "", "approved": False}, config=config)

# 查看暂停时的状态
snap = graph1.get_state(config)
print(f"\n⏸️ 当前暂停在：下一步是 {snap.next}")
print(f"   草稿内容：{snap.values['draft']}")

# 人审核后，决定继续（不传新输入，直接 None 接着跑）
print("\n第2步：人工确认草稿，让它继续发送")
graph1.invoke(None, config=config)  # 传 None = 从暂停处继续
print("→ 发送完成\n")


# ============================================================
# 方式二：interrupt() 函数（节点内主动暂停，拿人工输入）
# ============================================================
# 比 interrupt_before 更灵活：在节点内部任何时候暂停，还能从人那里"问"一个值回来。
# 场景：执行危险工具前，interrupt 问人"允许吗"，人回答后才继续。
print("=" * 60)
print("方式二：interrupt()（主动暂停问人，拿回输入再继续）")
print("=" * 60)

from langgraph.types import interrupt, Command


class ToolState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_result: str


def risky_tool(state: ToolState) -> dict:
    """一个"危险"操作：执行前先用 interrupt 问人是否允许。"""
    # interrupt(问题) 会暂停图，把问题抛出去；恢复时把人的回答作为返回值
    user_answer = interrupt("这个操作会删数据，允许执行吗？(yes/no)")
    print(f"  [工具] 收到人工回复：{user_answer}")
    if str(user_answer).lower().startswith("yes") or "是" in str(user_answer):
        return {"tool_result": "✅ 已执行（数据已删）"}
    return {"tool_result": "❌ 已取消（人工拒绝）"}


b2 = StateGraph(ToolState)
b2.add_node("risky_tool", risky_tool)
b2.add_edge(START, "risky_tool")
b2.add_edge("risky_tool", END)
graph2 = b2.compile(checkpointer=MemorySaver())

config2 = {"configurable": {"thread_id": "tool-1"}}

# 第一次：会触发 interrupt 暂停
print("\n第1步：调用危险工具（会暂停问人）")
graph2.invoke({"messages": [HumanMessage(content="删数据")], "tool_result": ""}, config=config2)

snap2 = graph2.get_state(config2)
print(f"⏸️ 暂停中，等待人工输入。下一步: {snap2.next}")

# 恢复：用 Command(resume=...) 把人的回答传回去
print("\n第2步：人工回复「yes」，继续执行")
result = graph2.invoke(Command(resume="yes"), config=config2)
print(f"最终结果：{result['tool_result']}")


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. interrupt 让图在关键步骤暂停，等人介入——保留对 Agent 的控制权")
print("2. interrupt_before=['节点']：在某节点前必停（简单）")
print("3. interrupt(问题)：节点内主动停，还能用 Command(resume=) 拿人工输入回来（灵活）")
print("4. 暂停后用 invoke(None, config) 或 invoke(Command(resume=...), config) 恢复")
print("5. interrupt 必须配 checkpointer（要记住暂停在哪才能恢复）")
print("6. 下一章：多节点编排——把多个步骤拼成完整工作流")
print("=" * 60)
