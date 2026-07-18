"""
04 - 结构化输出：让模型吐对象，而不是一坨文字
==============================================
【这章学什么】
  - with_structured_output：让模型按你定义的数据结构返回
  - 用 Pydantic 定义"我要什么样的数据"

【为什么学它】
  前几章模型返回的都是"一坨自由文字"。但实际项目里，我们往往要的是"结构化数据"：
  比如做一个"书籍信息提取"功能，希望模型返回：
      {书名, 作者, 年份, 标签}  ← 字段固定、类型固定
  而不是一段"《三体》是刘慈欣在 2008 年写的……"的文字（还得自己写正则去抠）。

  with_structured_output 让模型直接返回一个 Python 对象，字段名、类型都对好，
  拿来就能用，不用解析。

【类比】
  - 普通调用 = 让人写篇作文（格式自由，难提取信息）
  - 结构化输出 = 给人一张表格让他填（字段固定，直接拿数据）

【运行】
  cd code
  uv run python learn_langchain/04_structured_output.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ── 1. 用 Pydantic 定义你想要的数据结构 ─────────────────────
# Pydantic 的 BaseModel 就是"数据表格的定义"：每个字段是什么名、什么类型。
# Field(description=...) 是给模型看的说明，告诉它这个字段该填什么。
print("=" * 60)
print("1. 定义结构：书籍信息")
print("=" * 60)


class BookInfo(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    year: int = Field(description="出版年份，纯数字")
    tags: list[str] = Field(description="3 个分类标签")


# with_structured_output 把"模型"变成"会按 BookInfo 结构输出的模型"
# ⚠️ 避坑：通义千问必须用 method="function_calling"。
#   默认的 json_mode 会让 Qwen 输出中文 key（如"姓名"），解析会失败。
#   function_calling 通过函数签名约束输出，最稳。
structured_llm = llm.with_structured_output(BookInfo, method="function_calling")

result = structured_llm.invoke("介绍一下《三体》这本书，刘慈欣写的，2008 年出版")
print("结果类型：", type(result).__name__)  # 直接是 BookInfo 对象
print(f"书名：{result.title}")
print(f"作者：{result.author}")
print(f"年份：{result.year}（类型 {type(result.year).__name__}，自动转成了数字）")
print(f"标签：{result.tags}")
print()


# ── 2. 接进 LCEL 链：模板 | 结构化模型 ──────────────────────
print("=" * 60)
print("2. 结构化输出 + LCEL 链")
print("=" * 60)

from langchain_core.prompts import ChatPromptTemplate

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个信息提取助手，从用户描述中提取人物信息。"),
        ("human", "{text}"),
    ]
)


class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄，纯数字")
    occupation: str = Field(description="职业")


# 链：模板 → 结构化模型（注意这里不用 StrOutputParser，因为要的是对象不是字符串）
extract_chain = extract_prompt | llm.with_structured_output(
    PersonInfo, method="function_calling"
)

text = "张三今年 28 岁，是一名前端工程师，最近在学 AI。"
person = extract_chain.invoke({"text": text})
print(f"提取出：{person.name}，{person.age} 岁，职业 {person.occupation}")
print()


# ── 3. 为什么不直接用字符串 + 正则？────────────────────────
print("=" * 60)
print("3. 对比：结构化输出 vs 手动解析")
print("=" * 60)
print("【手动解析】要写正则抠 '28'、判断是年龄还是别的数字，脆弱且易错")
print("【结构化输出】模型直接给 age=28（int 类型），拿来即用")
print("  → 信息提取、表单填充、数据入库等场景，结构化输出是首选")
print()


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. Pydantic BaseModel 定义数据结构；Field(description) 指导模型填值")
print("2. with_structured_output 让模型直接返回该结构的对象")
print("3. 配通义千问要用 method='function_calling'（json_mode 会失败）")
print("4. 下一章：让模型不只是回答，还能『调用工具』完成任务")
print("=" * 60)
