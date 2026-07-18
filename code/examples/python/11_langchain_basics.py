"""
LangChain 入门示例
==================
本文件演示 LangChain 的核心概念，配合 Anthropic Claude 模型使用。

目录：
  1. 安装和配置（ChatAnthropic）
  2. PromptTemplate 和 ChatPromptTemplate
  3. LCEL 管道（.pipe() 和 | 操作符）
  4. OutputParser（StrOutputParser、JsonOutputParser）
  5. RunnableParallel（并行运行多个链）
  6. RunnablePassthrough（透传输入）
  7. 流式输出（chain.stream()）
  8. 与原生 API 对比（LCEL vs 手写）

安装依赖：
  pip install langchain langchain-anthropic

运行前请设置环境变量：
  export ANTHROPIC_API_KEY="your-api-key"
"""

# ============================================================
# 1. 安装和配置
# ============================================================
# pip install langchain langchain-anthropic
#
# LangChain 是一个把 LLM（大语言模型）各种能力"积木化"的框架。
# langchain-anthropic 是专门连接 Claude 系列模型的适配器。

import os
import json
from typing import Any

# ChatAnthropic：LangChain 包装的 Claude 客户端
from langchain_anthropic import ChatAnthropic

# Prompt 相关
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 输出解析器
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Runnable 组件（LCEL 核心）
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 用于 JsonOutputParser 的类型提示（可选，但推荐）
from langchain_core.pydantic_v1 import BaseModel, Field


# ---------- 初始化模型 ----------
# model：指定要用的 Claude 版本
# max_tokens：最多生成多少个 token（大致对应汉字/单词数量）
# temperature：0 = 输出更确定，1 = 输出更随机/有创意
llm = ChatAnthropic(
    model="claude-opus-4-5",
    max_tokens=1024,
    temperature=0,
)

print("=" * 60)
print("1. ChatAnthropic 基础调用")
print("=" * 60)

# 最简单的调用方式：直接传字符串
response = llm.invoke("用一句话解释什么是 LangChain？")
# response 是 AIMessage 对象，.content 取出文本内容
print("直接调用结果：", response.content)
print()


# ============================================================
# 2. PromptTemplate 和 ChatPromptTemplate
# ============================================================
# PromptTemplate：把"模板字符串 + 变量"封装成可复用的提示词构件。
# 好比一张"填空题"，每次只需填入不同变量，生成不同的提示词。

print("=" * 60)
print("2. PromptTemplate 和 ChatPromptTemplate")
print("=" * 60)

# ---------- PromptTemplate（纯文本模板）----------
# {topic} 是占位变量，调用时传入具体值
simple_template = PromptTemplate(
    input_variables=["topic"],
    template="请用3句话解释 {topic}，要求通俗易懂，适合初学者理解。",
)

# format() 生成最终字符串，但不调用模型
formatted_prompt = simple_template.format(topic="向量数据库")
print("PromptTemplate 生成的提示词：")
print(formatted_prompt)
print()

# ---------- ChatPromptTemplate（对话消息模板）----------
# 更常用的方式：按角色（system / human / ai）组织多轮消息
chat_template = ChatPromptTemplate.from_messages(
    [
        # ("system", ...) 设定 AI 的身份和行为规则
        ("system", "你是一位专业的 {domain} 领域顾问，回答要简洁专业。"),
        # ("human", ...) 代表用户输入
        ("human", "请解释 {concept} 的核心原理。"),
    ]
)

# format_messages() 生成消息列表（不调用模型）
messages = chat_template.format_messages(domain="AI 工程", concept="RAG（检索增强生成）")
print("ChatPromptTemplate 生成的消息列表：")
for msg in messages:
    print(f"  [{msg.__class__.__name__}] {msg.content}")
print()


# ============================================================
# 3. LCEL 管道（| 操作符 和 .pipe()）
# ============================================================
# LCEL = LangChain Expression Language，LangChain 的"流水线语言"。
# 核心思想：把多个处理步骤用 | 连接成一条链，数据从左流到右。
#
#   提示词模板 → 模型 → 输出解析器
#
# 类比：Unix 管道 `cat file.txt | grep "key" | wc -l`
#       数据一步步流过每个处理节点。

print("=" * 60)
print("3. LCEL 管道（| 操作符）")
print("=" * 60)

# 用 | 操作符把三个组件串成一条链
# StrOutputParser 把 AIMessage 对象转成纯字符串
chain_with_pipe = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位技术写作专家。"),
            ("human", "用一句话定义 {term}。"),
        ]
    )
    | llm
    | StrOutputParser()
)

result = chain_with_pipe.invoke({"term": "Embedding（向量嵌入）"})
print("LCEL 管道执行结果：", result)
print()

# ---------- 等价写法：.pipe() ----------
# | 和 .pipe() 完全等价，选自己喜欢的风格即可
prompt_step = ChatPromptTemplate.from_messages(
    [("human", "用比喻解释 {concept}，一句话即可。")]
)
chain_with_pipe_method = prompt_step.pipe(llm).pipe(StrOutputParser())

result2 = chain_with_pipe_method.invoke({"concept": "Token（令牌）"})
print(".pipe() 写法执行结果：", result2)
print()


# ============================================================
# 4. OutputParser（输出解析器）
# ============================================================
# OutputParser 负责把模型输出转成我们想要的格式。
# - StrOutputParser：最简单，直接返回字符串
# - JsonOutputParser：把模型输出的 JSON 字符串解析成 Python 字典

print("=" * 60)
print("4. OutputParser")
print("=" * 60)

# ---------- StrOutputParser（已在上面演示，这里补充说明）----------
str_parser = StrOutputParser()
# StrOutputParser 接收 AIMessage，返回 .content 字符串
# 相当于手写：lambda msg: msg.content

# ---------- JsonOutputParser ----------
# 先定义期望的输出结构（可选，但能让模型输出更规范）
class ConceptExplanation(BaseModel):
    concept: str = Field(description="概念名称")
    definition: str = Field(description="一句话定义")
    analogy: str = Field(description="生活类比")
    difficulty: str = Field(description="难度等级：入门/中级/高级")


json_parser = JsonOutputParser(pydantic_object=ConceptExplanation)

# get_format_instructions() 自动生成告诉模型"请输出 JSON"的指令
format_instructions = json_parser.get_format_instructions()

json_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是 AI 概念解释器，必须严格按照指定 JSON 格式回答。"),
        (
            "human",
            "请解释 {concept}。\n\n输出格式要求：\n{format_instructions}",
        ),
    ]
)

json_chain = json_prompt | llm | json_parser

json_result = json_chain.invoke(
    {
        "concept": "Fine-tuning（微调）",
        "format_instructions": format_instructions,
    }
)
print("JsonOutputParser 解析结果（Python 字典）：")
print(json.dumps(json_result, ensure_ascii=False, indent=2))
print()


# ============================================================
# 5. RunnableParallel（并行运行多个链）
# ============================================================
# RunnableParallel 让多个链同时处理同一个输入，最后把结果合并成字典。
# 类比：同时让多位专家分析同一份报告，最后汇总各自的意见。
#
# 适用场景：需要对同一输入从多个角度分析，或者需要同时获取多项信息。

print("=" * 60)
print("5. RunnableParallel（并行执行）")
print("=" * 60)

# 链1：生成优点
pros_chain = (
    ChatPromptTemplate.from_messages(
        [("human", "列出 {technology} 的3个核心优点，每条一句话。")]
    )
    | llm
    | StrOutputParser()
)

# 链2：生成缺点
cons_chain = (
    ChatPromptTemplate.from_messages(
        [("human", "列出 {technology} 的3个主要缺点，每条一句话。")]
    )
    | llm
    | StrOutputParser()
)

# 链3：生成使用场景
use_case_chain = (
    ChatPromptTemplate.from_messages(
        [("human", "列出 {technology} 最适合的2个使用场景，每条一句话。")]
    )
    | llm
    | StrOutputParser()
)

# 用 RunnableParallel 同时运行三条链
# 输入 {"technology": "..."} 会同时传给三条链
parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    use_cases=use_case_chain,
)

parallel_result = parallel_chain.invoke({"technology": "向量数据库"})
print("RunnableParallel 并行执行结果：")
print("【优点】\n", parallel_result["pros"])
print("【缺点】\n", parallel_result["cons"])
print("【使用场景】\n", parallel_result["use_cases"])
print()


# ============================================================
# 6. RunnablePassthrough（透传输入）
# ============================================================
# RunnablePassthrough 的作用是"原样传递输入，不做任何处理"。
# 最常见用途：在管道中保留原始输入，同时传递给下游步骤。
#
# 类比：快递分拣时，在包裹上贴标签的同时，包裹本身原封不动地继续传送。

print("=" * 60)
print("6. RunnablePassthrough（透传输入）")
print("=" * 60)

# 场景：生成回答的同时，把原始问题也保留在输出中
passthrough_chain = RunnableParallel(
    # "original_question" 字段：原样保留输入中的 question 字段
    original_question=RunnablePassthrough() | (lambda x: x["question"]),
    # "answer" 字段：经过模型生成的回答
    answer=(
        ChatPromptTemplate.from_messages(
            [("human", "{question}")]
        )
        | llm
        | StrOutputParser()
    ),
)

passthrough_result = passthrough_chain.invoke(
    {"question": "用一句话解释什么是 Agent（智能体）？"}
)
print("RunnablePassthrough 示例：")
print("原始问题：", passthrough_result["original_question"])
print("模型回答：", passthrough_result["answer"])
print()


# ============================================================
# 7. 流式输出（chain.stream()）
# ============================================================
# stream() 让模型边生成边输出，而不是等全部生成完再返回。
# 效果就像 ChatGPT 网页上文字一个个出现的效果。
#
# 类比：广播直播 vs 录播——直播边说边听，录播等节目录完才能看。

print("=" * 60)
print("7. 流式输出（stream()）")
print("=" * 60)

stream_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位 AI 科普作者。"),
            ("human", "用100字以内介绍 {topic}，要生动有趣。"),
        ]
    )
    | llm
    | StrOutputParser()
)

print("流式输出效果（文字逐步出现）：")
# stream() 返回一个生成器，每次 yield 一小段文本
for chunk in stream_chain.stream({"topic": "提示词工程（Prompt Engineering）"}):
    # end="" 和 flush=True 实现实时打印（不换行、不缓冲）
    print(chunk, end="", flush=True)
print("\n")


# ============================================================
# 8. 与原生 API 对比：LCEL vs 手写
# ============================================================
# 这一节通过同一个任务，展示两种写法的对比，
# 帮助理解 LCEL 到底帮我们省了什么。

print("=" * 60)
print("8. LCEL vs 原生 API 对比")
print("=" * 60)

TASK_INPUT = "机器学习"

# ---------- 方式 A：原生 API 手写 ----------
# 需要手动组装消息、调用模型、提取结果——每一步都是命令式代码
print("【方式 A：手写原生调用】")

from anthropic import Anthropic

native_client = Anthropic()

# 手动构建消息列表
native_messages = [
    {
        "role": "user",
        "content": f"用一句话定义 {TASK_INPUT}，并给出一个生活类比。",
    }
]

# 手动调用 API
native_response = native_client.messages.create(
    model="claude-opus-4-5",
    max_tokens=256,
    messages=native_messages,
)

# 手动提取文本内容
native_text = native_response.content[0].text
print("原生 API 结果：", native_text)
print()

# ---------- 方式 B：LCEL 链式写法 ----------
# 声明式：只描述"做什么"，不关心"怎么做"
# 更易读、更易复用、天然支持流式/并行/批量
print("【方式 B：LCEL 链式写法】")

lcel_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "用一句话定义 {topic}，并给出一个生活类比。",
            )
        ]
    )
    | llm
    | StrOutputParser()
)

lcel_text = lcel_chain.invoke({"topic": TASK_INPUT})
print("LCEL 结果：", lcel_text)
print()

# ---------- 对比总结 ----------
print("【对比总结】")
comparison = """
┌─────────────────┬──────────────────────────┬──────────────────────────┐
│ 维度            │ 手写原生 API             │ LCEL 链式写法            │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ 代码风格        │ 命令式（一步一步写）     │ 声明式（描述做什么）     │
│ 可读性          │ 较低（夹杂格式细节）     │ 较高（结构一目了然）     │
│ 复用性          │ 低（每次重写）           │ 高（链可以拼接/嵌套）    │
│ 流式支持        │ 需手动处理 stream        │ .stream() 开箱即用       │
│ 并行执行        │ 需手写 asyncio           │ RunnableParallel 直接用  │
│ 批量处理        │ 需手写循环               │ .batch() 一行搞定        │
│ 调试追踪        │ 手动 print               │ LangSmith 可视化追踪     │
│ 学习成本        │ 低（只需了解 API）       │ 中（需学 LCEL 概念）     │
└─────────────────┴──────────────────────────┴──────────────────────────┘

结论：
- 简单脚本、一次性任务 → 手写原生 API 更直接
- 生产级应用、需要复用和扩展 → LCEL 优势明显
"""
print(comparison)

print("全部示例运行完毕！")
