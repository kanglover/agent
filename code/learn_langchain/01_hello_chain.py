"""
01 - Hello, LangChain！第一次调用
==================================
【这章学什么】
  - 用 LangChain 调用大模型，拿到回复
  - 理解 LangChain 里的两个最基本概念：ChatModel（模型）、Message（消息）

【为什么学它】
  万丈高楼平地起。后面所有花哨的功能（链、工具、记忆、RAG），
  本质上都是在"调模型拿回复"这件事上一步步加东西。先把这步跑通、看懂。

【类比】
  LangChain 就像一个"电话总机"——你不用自己直接拨号到大模型公司，
  而是通过它来拨号。好处是：换一家模型公司（比如从通义千问换到 Claude），
  你只要换个"插头"，其他代码基本不用改。

【环境配置】
  在 code/.env 里写好（这是你之前已经配好的通义千问）：
    OPENAI_API_KEY=sk-你的通义千问Key
    OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

【运行】
  cd code
  uv run python learn_langchain/01_hello_chain.py
"""

import os
from dotenv import load_dotenv

# load_dotenv() 会自动读取当前目录(code/)下的 .env 文件，把里面的配置变成环境变量
# 这样代码里就能用 os.getenv("OPENAI_API_KEY") 拿到 Key，而不用把 Key 写死在代码里
load_dotenv()

# ChatOpenAI 是 LangChain 提供的"OpenAI 兼容模型"客户端。
# 通义千问支持 OpenAI 的接口格式，所以我们用 ChatOpenAI 来连它（换个地址即可）。
from langchain_openai import ChatOpenAI

# ── 创建一个"模型对象" ──────────────────────────────────────
# 这一步相当于"准备好一部电话，拨通通义千问"。
# 三个参数的意思：
#   model      → 用哪个模型（qwen-max 是通义千问的旗舰款，最聪明）
#   api_key    → 你的身份凭证（从 .env 读）
#   base_url   → 拨到哪个地址（通义千问的地址，不是 OpenAI 的）
#   temperature→ 0 表示输出稳定不随机；越大越有创意/随机。学习阶段先用 0
llm = ChatOpenAI(
    model="qwen-max",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

# ── 方式一：最简单的调用，直接传一句话 ──────────────────────
print("=" * 60)
print("方式一：直接传一句话")
print("=" * 60)

# invoke() = "调用"。传进去一句话，它返回模型的回复。
response = llm.invoke("用一句话介绍一下你自己")

# 注意：response 不是字符串，而是一个 AIMessage 对象（里面除了文本还有其他信息）。
# 想拿到纯文本，用 .content
print("回复类型：", type(response).__name__)
print("回复内容：", response.content)
print()


# ── 方式二：用"消息列表"调用，更标准 ────────────────────────
# 实际项目里，我们通常按"角色"组织消息。最常见的两种角色：
#   system → 设定 AI 的"人设/规则"（你是谁、怎么回答）
#   human  → 用户说的话
# 这就像给服务员两张纸条：一张写"你今天扮演礼貌的迎宾"，一张写"顾客的话"。
print("=" * 60)
print("方式二：用消息列表（system + human）")
print("=" * 60)

from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个说话非常简短的助手，每次回答不超过 15 个字。"),
    HumanMessage(content="什么是人工智能？"),
]

response2 = llm.invoke(messages)
print("回复内容：", response2.content)
print()


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. ChatOpenAI 是模型客户端；invoke() 是调用方法")
print("2. 返回的是 AIMessage 对象，用 .content 取文本")
print("3. 可以直接传字符串，也可以传 [SystemMessage, HumanMessage] 消息列表")
print("4. 后面所有功能都建立在 invoke 这个基础动作之上")
print("=" * 60)
