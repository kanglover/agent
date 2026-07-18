"""
02 - 提示词模板 PromptTemplate
==============================
【这章学什么】
  - 用"模板"复用提示词，而不是每次手动拼字符串
  - 区分 PromptTemplate（纯文本）和 ChatPromptTemplate（按角色分消息）

【为什么学它】
  假设你要让模型解释 10 个不同概念。如果手动写：
      llm.invoke("用3句话解释 向量数据库")
      llm.invoke("用3句话解释 微调")
      llm.invoke("用3句话解释 RAG")
  ……这句"用3句话解释"要重复写 10 遍，格式稍一改动就得改 10 处。

  模板就是把这个"句式"抽出来，留个填空位 {topic}，每次只填空：
      template.format(topic="向量数据库")
  改格式时只改模板一处。这就是"复用"。

【类比】
  PromptTemplate 就像一张"填空题试卷"：题目框架印好了，每次考试只换填空内容。

【运行】
  cd code
  uv run python learn_langchain/02_prompt_template.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ── 1. PromptTemplate：纯文本模板 ──────────────────────────
# {topic} 是占位符（变量名自己取），调用 format() 时填进去
print("=" * 60)
print("1. PromptTemplate：纯文本模板")
print("=" * 60)

explain_template = PromptTemplate(
    input_variables=["topic"],          # 声明有哪些变量（供检查用）
    template="请用3句话解释 {topic}，要通俗易懂，适合初学者。",
)

# format() 只是把变量填进去，生成一段字符串，【不调用模型】
prompt_str = explain_template.format(topic="向量数据库")
print("生成的提示词：", prompt_str)

# 这才真正调用模型
response = llm.invoke(prompt_str)
print("模型回复：", response.content)
print()


# ── 2. ChatPromptTemplate：按角色分消息（更常用）────────────
# 实际项目几乎都用这个，因为它能清晰设定 system（人设）和 human（用户输入）
print("=" * 60)
print("2. ChatPromptTemplate：按角色分消息")
print("=" * 60)

chat_template = ChatPromptTemplate.from_messages(
    [
        # system：设定 AI 是什么角色、按什么风格回答
        ("system", "你是一位专业的 {domain} 顾问，回答要简洁专业，不超过50字。"),
        # human：用户的问题
        ("human", "请解释 {concept} 的核心原理。"),
    ]
)

# format_messages() 生成消息列表（不调用模型）
messages = chat_template.format_messages(domain="AI 工程", concept="RAG")
print("生成的消息：")
for msg in messages:
    print(f"  [{msg.__class__.__name__}] {msg.content}")

# 调用模型
response2 = llm.invoke(messages)
print("模型回复：", response2.content)
print()


# ── 3. 同一个模板，填不同的值复用 ──────────────────────────
# 这就是模板的价值：写一次，用多次
print("=" * 60)
print("3. 复用同一个模板解释 3 个概念")
print("=" * 60)

concepts = ["Embedding", "微调", "Token"]
for c in concepts:
    msgs = chat_template.format_messages(domain="AI 工程", concept=c)
    reply = llm.invoke(msgs)
    print(f"【{c}】{reply.content}")
    print()


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. PromptTemplate：纯文本模板，format() 填变量得到字符串")
print("2. ChatPromptTemplate：按 system/human 角色组织消息，更常用")
print("3. 模板的价值 = 复用：写一次句式，换不同变量反复调用")
print("4. 下一章我们把『模板→模型→取文本』串成一条流水线（LCEL）")
print("=" * 60)
