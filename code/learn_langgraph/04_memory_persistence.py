"""
04 - 记忆与持久化：让 Agent 记住对话
=====================================
【这章学什么】
  - Checkpointer：LangGraph 的"记忆"机制
  - MemorySaver：内存版记忆（程序结束就忘）
  - thread_id：用"会话 ID"区分不同对话
  - SqliteSaver：磁盘持久化（程序重启也记得）

【为什么学它】
  上一章的 Agent 每次 invoke 都是"失忆"的——你告诉它名字，下一句问它就忘了。
  因为图本身不存历史，每次调用都是全新开始。

  Checkpointer 就是给图加"记忆"：每执行一步就把当前 state 存下来。
  下次用同一个 thread_id 调用时，自动加载之前的 state 接着聊。
  这就是多轮对话的本质。

【类比】
  没记忆的图 = 每次打电话都换一个客服（从头解释）
  Checkpointer = 客服做了笔记，下次你报"会话号"(thread_id) 就能接着上次聊
  MemorySaver = 笔记写在便签上（关机就没）；SqliteSaver = 写进笔记本（永久保存）

【运行】
  cd code
  uv run python learn_langgraph/04_memory_persistence.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Annotated
from typing_extensions import TypedDict
import operator

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def call_model(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


from langgraph.graph import StateGraph, START, END

b = StateGraph(AgentState)
b.add_node("agent", call_model)
b.add_edge(START, "agent")
b.add_edge("agent", END)


# ============================================================
# 1. 没有记忆：每次都失忆
# ============================================================
print("=" * 60)
print("1. 没加 checkpointer：图是失忆的")
print("=" * 60)

no_mem_graph = b.compile()  # 不传 checkpointer
no_mem_graph.invoke({"messages": [HumanMessage(content="我叫小明，记住了")]})
r = no_mem_graph.invoke({"messages": [HumanMessage(content="我叫什么？")]})
print("问名字：", r["messages"][-1].content)
print("→ 答：不记得，因为第二次调用是全新的\n")


# ============================================================
# 2. MemorySaver：内存记忆 + thread_id
# ============================================================
# 加了 checkpointer 后，图会记住每个 thread 的状态。
# thread_id 像会话号：不同 thread_id 互不干扰，同 thread_id 接着上次聊。
print("=" * 60)
print("2. MemorySaver：用 thread_id 记住对话")
print("=" * 60)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
mem_graph = b.compile(checkpointer=memory)  # ← 传入 checkpointer

# config 里指定 thread_id
config = {"configurable": {"thread_id": "session-1"}}

def chat(text: str):
    r = mem_graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    print(f"用户：{text}")
    print(f"助手：{r['messages'][-1].content}\n")

chat("我叫小明")
chat("我喜欢吃火锅")
chat("我叫什么？我喜欢吃什么？")  # 这次该记得了

# 换个 thread_id，就是全新对话
print("--- 换 thread_id = session-2（全新对话）---")
config2 = {"configurable": {"thread_id": "session-2"}}
r = mem_graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config=config2)
print("助手：", r["messages"][-1].content)
print("→ 另一个会话不知道，记忆按 thread_id 隔离\n")


# ============================================================
# 3. 查看记忆里存了什么
# ============================================================
print("=" * 60)
print("3. 查看 checkpointer 里的记忆")
print("=" * 60)
# get_state 能拿到当前 thread 的完整状态
snap = mem_graph.get_state(config)
print(f"session-1 当前存了 {len(snap.values.get('messages', []))} 条消息")
for m in snap.values["messages"]:
    print(f"  [{m.type}] {m.content[:40]}")


# ============================================================
# 4. SqliteSaver：持久化到磁盘（重启也记得）
# ============================================================
# MemorySaver 存内存，程序退出就没了。要永久保存用 SqliteSaver（存数据库文件）。
print("\n" + "=" * 60)
print("4. SqliteSaver：持久化到磁盘")
print("=" * 60)

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 建一个 sqlite 数据库文件（实际存在当前目录）
conn = sqlite3.connect(":memory:", check_same_thread=False)  # 演示用内存库；真实用文件路径
# 真实项目: conn = sqlite3.connect("agent_memory.db", check_same_thread=False)

sqlite_saver = SqliteSaver(conn)
sqlite_saver.setup()  # 建表
persist_graph = b.compile(checkpointer=sqlite_saver)
cfg = {"configurable": {"thread_id": "persist-1"}}

persist_graph.invoke({"messages": [HumanMessage(content="我叫小红")]}, config=cfg)
# 模拟"重启"：重新用同一个数据库+thread_id 加载
snap2 = persist_graph.get_state(cfg)
print("重启前存的最后消息:", snap2.values["messages"][-1].content)

r = persist_graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config=cfg)
print("重启后问名字:", r["messages"][-1].content)
print("→ 存数据库后，只要数据库文件还在，跨进程/重启都能恢复记忆\n")


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. checkpointer = 图的记忆；compile(checkpointer=...) 开启")
print("2. thread_id 区分会话：同 id 接着聊，不同 id 互不干扰")
print("3. MemorySaver 存内存（快但易失）；SqliteSaver 存磁盘（持久）")
print("4. get_state(config) 查看某会话当前记忆")
print("5. 上一章的 Agent + 本章的记忆 = 一个完整可多轮对话的智能体")
print("6. 下一章：人机协作——在关键步骤暂停，等人确认再继续")
print("=" * 60)
