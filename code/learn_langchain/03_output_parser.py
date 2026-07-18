"""
03 - 输出解析器 & LCEL 管道
============================
【这章学什么】
  - LCEL（LangChain 的"流水线"写法）：用 | 把步骤串成一条链
  - 输出解析器 OutputParser：把模型的输出转成想要的格式

【为什么学它】
  前两章每次调用都要分三步：
      1. 组装提示词
      2. 调用模型，拿到 AIMessage
      3. 从 AIMessage 里 .content 取出文本
  写三遍就烦了。LCEL 把这三步用 | 串成一条"流水线"：
      提示词模板 | 模型 | 输出解析器
  以后一行定义、到处复用，还天然支持流式、批量、并行。

【类比】
  LCEL 就像 Unix 管道 `cat 文件 | grep 关键词 | wc -l`：
  数据从左流到右，每一步处理一下传给下一步。 | 就是"交给下一步"的意思。
  输出解析器就是流水线最后一道工序——把半成品包装成你要的成品（纯文本/字典）。

【运行】
  cd code
  uv run python learn_langchain/03_output_parser.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ── 1. 最简单的链：模板 | 模型 | 解析器 ─────────────────────
print("=" * 60)
print("1. LCEL 基本链：模板 | 模型 | StrOutputParser")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages(
    [("human", "用一句话定义 {term}。")]
)

# 用 | 把三步串起来，这就叫一条"链"(chain)
# StrOutputParser 的作用：自动从 AIMessage 里取出 .content，变成纯字符串
chain = prompt | llm | StrOutputParser()

# 调用：只传变量字典，链会自动走完三步，直接返回字符串
result = chain.invoke({"term": "Embedding"})
print("结果类型：", type(result).__name__)  # 直接是 str，不再是 AIMessage
print("结果：", result)
print()


# ── 2. 为什么用 StrOutputParser？对比一下 ───────────────────
# 不用解析器：拿到的是 AIMessage，要手动 .content
print("=" * 60)
print("2. 对比：有/无 StrOutputParser")
print("=" * 60)

chain_no_parser = prompt | llm
r1 = chain_no_parser.invoke({"term": "Token"})
print("无解析器：", type(r1).__name__, "→", r1.content)  # AIMessage，要取 .content

chain_with_parser = prompt | llm | StrOutputParser()
r2 = chain_with_parser.invoke({"term": "Token"})
print("有解析器：", type(r2).__name__, "→", r2)  # 直接是 str
print()


# ── 3. 链可以复用：换个变量就跑 ─────────────────────────────
print("=" * 60)
print("3. 同一条链，解释多个术语")
print("=" * 60)

for term in ["向量数据库", "微调", "RAG"]:
    print(f"【{term}】", chain.invoke({"term": term}))
print()


# ── 4. .pipe() 写法：和 | 完全等价 ─────────────────────────
# 不喜欢用 | 的，也可以用 .pipe()，效果一样，选顺手的
print("=" * 60)
print("4. .pipe() 写法（和 | 等价）")
print("=" * 60)

chain_pipe = ChatPromptTemplate.from_messages(
    [("human", "用比喻解释 {concept}，一句话。")]
).pipe(llm).pipe(StrOutputParser())

print(chain_pipe.invoke({"concept": "Token"}))
print()


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. LCEL 用 | 把『模板 → 模型 → 解析器』串成一条链")
print("2. StrOutputParser 自动取 .content，让链直接返回字符串")
print("3. 链定义一次可反复 invoke，只换变量")
print("4. | 和 .pipe() 等价，任选其一")
print("5. 下一章：让模型直接吐结构化数据（字典/对象），而不只是字符串")
print("=" * 60)
