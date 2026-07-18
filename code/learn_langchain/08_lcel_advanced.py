"""
08 - LCEL 进阶：自定义、降级、路由、并行
=========================================
【这章学什么】
  - RunnableLambda：把普通函数接进链
  - with_fallbacks：出错时自动降级到备选方案
  - RunnableBranch：按条件路由到不同链
  - RunnableParallel：并行跑多个链再合并

【为什么学它】
  前面学的链是"一条直线走到底"。真实场景更复杂：
  - 链里需要插一段自定义逻辑（比如把输出转大写）→ RunnableLambda
  - 模型偶尔抽风报错，要有 Plan B 兜底 → with_fallbacks
  - 不同类型的问题要走不同处理流程（分类路由）→ RunnableBranch
  - 一个输入要同时算多个结果再合并 → RunnableParallel
  这些是 LCEL 的"积木"，掌握后能拼出任意复杂的工作流。

【类比】
  LCEL 链就像搭乐高。前面学的是直条积木（模板|模型|解析器），
  这章学的是转接头(RunnableLambda)、备胎(with_fallbacks)、
  分岔路(RunnableBranch)、多车道(RunnableParallel)。

【前置】先掌握第03章（LCEL 基础）再读本篇。

【运行】
  cd code
  uv run python learn_langchain/08_lcel_advanced.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch

llm = ChatOpenAI(
    model="qwen-max",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ============================================================
# 1. RunnableLambda：把普通函数接进链
# ============================================================
# 链里的每一段都必须是"Runnable"（可运行对象）。普通函数用 RunnableLambda 包一下就能进链。
print("=" * 60)
print("1. RunnableLambda：自定义函数接入链")
print("=" * 60)

# 一个普通函数：把字符串转大写并加前缀
def emphasize(text: str) -> str:
    return f"【重点】{text.upper()}"

# 不用 RunnableLambda，普通函数不能直接 | 进链；包一下就行
chain = (
    ChatPromptTemplate.from_messages([("human", "用一句话定义 {term}")])
    | llm
    | StrOutputParser()
    | RunnableLambda(emphasize)   # 把模型输出再加工一下
)

print(chain.invoke({"term": "Agent"}))
print()


# ============================================================
# 2. with_fallbacks：出错自动降级
# ============================================================
# 生产环境模型可能偶发报错（限流、超时）。with_fallbacks 让主链失败时自动用备选链。
print("=" * 60)
print("2. with_fallbacks：主链失败自动降级")
print("=" * 60)

# 主链：故意用一个会报错的"模型"（空 base_url，必失败）
bad_llm = ChatOpenAI(
    model="qwen-max",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="http://故意写错的地址",  # 故意让它连不上
    temperature=0,
)
primary = ChatPromptTemplate.from_messages([("human", "用一句话解释 {x}")]) | bad_llm | StrOutputParser()

# 备链：用正常模型
fallback = ChatPromptTemplate.from_messages([("human", "用一句话解释 {x}")]) | llm | StrOutputParser()

# 主链挂上备链：主链抛异常时自动切备链
chain_with_fallback = primary.with_fallbacks([fallback])

print("用故意失败的主链 + 正常备链：")
print(chain_with_fallback.invoke({"x": "降级机制"}))
print("→ 主链失败了，但因为有 fallback，用户无感知地拿到了备链结果\n")


# ============================================================
# 3. RunnableBranch：按条件路由到不同链
# ============================================================
# 不同类型的问题走不同处理。比如：问数学走计算链，问翻译走翻译链，其它走默认。
print("=" * 60)
print("3. RunnableBranch：条件路由")
print("=" * 60)

# 判断函数：返回 True/False 决定走不走这条分支
def is_math(question: str) -> bool:
    return any(op in question for op in ["加", "减", "乘", "除", "+", "-", "*", "/"])

def is_translate(question: str) -> bool:
    return "翻译" in question

math_chain = (
    ChatPromptTemplate.from_messages([("human", "你是数学老师，回答：{q}。简洁。")])
    | llm | StrOutputParser()
)
translate_chain = (
    ChatPromptTemplate.from_messages([("human", "你是翻译，把以下内容翻译成英文：{q}")])
    | llm | StrOutputParser()
)
default_chain = (
    ChatPromptTemplate.from_messages([("human", "简洁回答：{q}")])
    | llm | StrOutputParser()
)

# RunnableBranch 接收 (条件, 链) 列表，最后一个是不带条件的默认分支
router = RunnableBranch(
    (is_math, math_chain),
    (is_translate, translate_chain),
    default_chain,  # 默认
)

for q in ["3 加 5 等于多少", "把'你好世界'翻译", "什么是 RAG"]:
    # 注意：分支链的输入是原始的 question 字符串，链里用 {q} 接收，所以传 {"q": q}
    print(f"问：{q}")
    print(f"答：{router.invoke({'q': q})}\n")


# ============================================================
# 4. RunnableParallel：并行跑多个链再合并
# ============================================================
# 同一个输入，同时从多个角度处理，结果合并成字典。比串行跑快得多。
print("=" * 60)
print("4. RunnableParallel：并行多角度分析")
print("=" * 60)

pros = ChatPromptTemplate.from_messages([("human", "列出 {tech} 的2个优点，每条一句")]) | llm | StrOutputParser()
cons = ChatPromptTemplate.from_messages([("human", "列出 {tech} 的2个缺点，每条一句")]) | llm | StrOutputParser()
use_cases = ChatPromptTemplate.from_messages([("human", "列出 {tech} 的2个适用场景，每条一句")]) | llm | StrOutputParser()

# 三个链同时跑，结果按 key 合并成 {pros:..., cons:..., use_cases:...}
parallel = RunnableParallel(pros=pros, cons=cons, use_cases=use_cases)

result = parallel.invoke({"tech": "向量数据库"})
print("优点：", result["pros"])
print("缺点：", result["cons"])
print("场景：", result["use_cases"])
print()


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：")
print("1. RunnableLambda：普通函数包一下就能进链（加工/转换）")
print("2. with_fallbacks：主链失败自动切备链，生产环境必备容错")
print("3. RunnableBranch：按条件把请求路由到不同链（分类处理）")
print("4. RunnableParallel：多链并行，结果合并成字典（多角度分析）")
print("5. 这些积木 + 03章的 | 管道，能拼出任意复杂的工作流")
print("=" * 60)
print("\n🎉 恭喜！学完这 8 篇，LangChain 的核心你已经掌握了。")
print("   下一步建议：把第05章(工具)和第06章(记忆)结合 → 这就是 Agent！")
print("   可参考 examples/python/12_langchain_agents.py 看 Agent 完整实现。")
