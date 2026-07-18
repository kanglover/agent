"""
06 - 记忆与多轮对话：让模型记住前面说过的话
============================================
【这章学什么】
  - 为什么需要"记忆"：模型本身是无记忆的
  - 用消息列表手动管理对话历史
  - 用 RunnableWithMessageHistory 自动维护记忆
  - 裁剪历史消息，防止超出上下文窗口

【为什么学它】
  大模型是"无状态"的——每次调用都是全新的，它不记得上一句你说了啥。
  你说"我叫小明"，下一句问"我叫什么"，它一脸懵。

  所谓"记忆"，其实是个工程技巧：把之前的对话存下来，每次调用时连同历史一起
  发给模型，让它"看起来"有记忆。本质是"把聊天记录塞进 prompt"。

【类比】
  模型像个"金鱼记忆"的客服，每次电话挂断就忘光。
  记忆系统 = 一个帮客服做笔记的助理，每次新电话来，先把之前的笔记递给他看。

【运行】
  cd code
  uv run python learn_langchain/06_memory.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ============================================================
# 1. 先看问题：模型本没有记忆
# ============================================================
print("=" * 60)
print("1. 没有记忆：模型不记得上一句")
print("=" * 60)

# 第一次告诉它名字
llm.invoke("我叫小明，记住了")  # 模型回复了，但没存
# 第二次问，它不知道
r = llm.invoke("我叫什么名字？")
print("没记忆时问名字：", r.content)
print("→ 它答不上来，因为第二次调用时它根本不知道第一次说过啥\n")


# ============================================================
# 2. 手动记忆：自己存消息列表，每次带上历史
# ============================================================
print("=" * 60)
print("2. 手动记忆：带上历史消息")
print("=" * 60)

# 一个"记忆"就是消息列表
history = [
    ("system", "你是一个友好的助手。")
]

def chat_manual(user_input: str) -> str:
    # 把新输入加进历史
    history.append(("human", user_input))
    # 把全部历史拼成消息列表发给模型
    msgs = [{"role": role, "content": c} for role, c in history]
    reply = llm.invoke(msgs).content
    # 把模型回复也存进历史，下一轮才能记住
    history.append(("assistant", reply))
    return reply

print("用户：我叫小明")
print("助手：", chat_manual("我叫小明"))
print("用户：我喜欢学 Python")
print("助手：", chat_manual("我喜欢学 Python"))
print("用户：我叫什么？我喜欢什么？")
print("助手：", chat_manual("我叫什么？我喜欢什么？"))
print("→ 这次记住了！因为每次都把历史带上了\n")


# ============================================================
# 3. 自动记忆：RunnableWithMessageHistory（更省事）
# ============================================================
# 手动管理历史很繁琐。LangChain 提供 RunnableWithMessageHistory，
# 你只管 invoke，它自动帮你存历史、下次自动带上。
print("=" * 60)
print("3. 自动记忆：RunnableWithMessageHistory")
print("=" * 60)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# 带占位符的模板：{history} 处会自动填入历史消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息插这里
    ("human", "{input}"),
])

# 基础链
chain = prompt | llm | StrOutputParser()

# 每个会话(session)一个历史记录容器。session_id 用来区分不同用户的对话
store = {}  # 实际项目可换成数据库

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 把普通链包装成"带记忆的链"
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# config 里的 session_id 决定用哪份记忆
config = {"configurable": {"session_id": "user_001"}}

def chat_auto(user_input: str):
    reply = chain_with_memory.invoke({"input": user_input}, config=config)
    print(f"用户：{user_input}")
    print(f"助手：{reply}\n")

chat_auto("我叫小红")
chat_auto("我在学 LangChain")
chat_auto("我叫什么？我在学什么？")  # 应该答得上来

# 换个 session_id，就是另一段独立的对话（互不干扰）
print("--- 切换到另一个会话 user_002（不知道小红说过啥）---")
config2 = {"configurable": {"session_id": "user_002"}}
print("助手：", chain_with_memory.invoke({"input": "我叫什么？"}, config=config2))
print("→ 另一个会话不知道，因为记忆是按 session_id 隔离的\n")


# ============================================================
# 4. 裁剪历史：防止对话太长超出上下文窗口
# ============================================================
# 对话越长，历史消息越多，最终会超出模型的上下文上限（且越贵）。
# 解决：只保留最近 N 条消息。trim_messages 是 LangChain 的裁剪工具。
print("=" * 60)
print("4. 裁剪历史：只保留最近的消息")
print("=" * 60)

from langchain_core.messages import trim_messages

# 模拟一段长历史
long_history = InMemoryChatMessageHistory()
for i in range(5):
    long_history.add_user_message(f"这是第 {i+1} 条用户消息")
    long_history.add_ai_message(f"这是第 {i+1} 条回复")

# 裁剪：只保留最近若干条
# ⚠️ 避坑：trim_messages 的 token_counter 用模型数 token 时，通义千问会报
#   NotImplementedError（它不知道 qwen-max 的分词规则）。这里改用按"消息条数"裁剪：
#   strategy="last" + 传一个简单的 token_counter（按字符数估算），绕开模型计数。
def char_counter(messages):
    # 粗略估算：一条消息算它的字符数 + 4（开销）。够用于演示。
    return sum(len(m.content) + 4 for m in messages)

trimmed = trim_messages(
    long_history.messages,
    max_tokens=40,              # 按上面的粗略计数设上限
    strategy="last",            # 保留最后的
    token_counter=char_counter, # 自己提供计数器，不用模型
    start_on="human",           # 从一条 human 消息开始（保证对话完整）
)
print(f"裁剪前 {len(long_history.messages)} 条 → 裁剪后 {len(trimmed)} 条")
for m in trimmed:
    print(f"  [{m.type}] {m.content}")
print("→ 长对话时这样做，既省 token 又防超限\n")


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. 模型本身无记忆；'记忆'= 把历史消息每次一起发给模型")
print("2. 手动：自己维护消息列表；自动：RunnableWithMessageHistory")
print("3. session_id 隔离不同对话；trim_messages 裁剪过长历史防超限")
print("4. 下一章学 RAG：另一种'记忆'——让模型记住你的私有文档")
print("=" * 60)
