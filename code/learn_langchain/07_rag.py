"""
07 - RAG：检索增强生成，让模型照着你的资料回答
==============================================
【这章学什么】
  - RAG 是什么、为什么需要它
  - 完整 RAG 流程：文档 → 切块 → 向量化 → 存向量库 → 检索 → 回答
  - 防幻觉的关键：prompt 里要求"只根据资料回答"

【为什么学它】
  大模型有两个毛病：
  1. 知识有截止日期（不知道昨天的事）
  2. 不知道你的私有信息（公司制度、个人笔记），还可能瞎编（幻觉）

  RAG 的解法：给模型一个"参考资料库"。提问时先从库里搜出相关段落，
  连同问题一起发给模型，让它"开卷作答"。

  这是目前企业用 AI 最落地的技术——做"基于自家文档的问答机器人"。

【类比】
  - 普通模型 = 闭卷考试的学生（凭记忆答，可能记错或瞎编）
  - RAG     = 开卷考试的学生（先翻你给的资料，照着答）
  Embedding + 向量库 = 帮学生快速翻到最相关那页的"索引"

【完整流程图】
  原始文档
    ↓ 切块（split，切成小段）
  文本块
    ↓ 向量化（embedding，每段变一串数字）
  向量
    ↓ 存入向量库（chroma）
  ────── 提问 ──────
  问题 → 向量化 → 在向量库找最相似的几段 → 拼进 prompt → 模型作答

【运行】
  cd code
  uv run python learn_langchain/07_rag.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

# Embedding 模型：把文字变成向量（一串数字），用来算"语义相似度"
# ⚠️ 避坑：配 Qwen 必须设 check_embedding_ctx_length=False。
#   默认 True 时 langchain 会先把文本切成 token 列表再发，但 Qwen 接口只收字符串，会报 400。
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False,
)


# ============================================================
# 第1步：准备原始文档（你的"资料库"）
# ============================================================
# 这里用一段虚构的公司制度。真实项目可换成 PDF/Word/网页，用文档加载器读入。
from langchain_core.documents import Document

raw_text = """
公司年假制度（2024 版）
1. 入职满 1 年的员工，享有 5 天带薪年假。
2. 入职满 3 年的员工，享有 10 天带薪年假。
3. 入职满 5 年及以上的员工，享有 15 天带薪年假。
4. 年假当年未休完的，可在次年第一季度补休，逾期作废。

公司报销制度（2024 版）
1. 出差交通费：高铁二等座实报实销，飞机经济舱实报实销。
2. 出差住宿费：一线城市上限 500 元/晚，其他城市上限 350 元/晚。
3. 报销需在出差结束后 7 个工作日内提交，逾期不予受理。
"""

# Document = "一段文本 + 元数据"。metadata 可存来源、章节，检索时能用来过滤
documents = [Document(page_content=raw_text, metadata={"source": "公司制度手册"})]
print(f"第1步：原始文档 {len(documents)} 篇，{len(raw_text)} 字")


# ============================================================
# 第2步：切块（Splitting）
# ============================================================
# 为什么要切？整篇太长塞不进模型，且检索时只想找"最相关的一小段"。
# RecursiveCharacterTextSplitter 会按 段落→换行→句号 的顺序尽量自然地切。
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,     # 每块最多 120 字
    chunk_overlap=20,   # 相邻块重叠 20 字，防止句子被从中间切断丢上下文
)
chunks = splitter.split_documents(documents)
print(f"第2步：切成 {len(chunks)} 块")


# ============================================================
# 第3步：向量化 + 存入向量库
# ============================================================
# from_documents 自动：每块调 embedding 得到向量 → 存进 Chroma
# Chroma 是轻量向量库；这里用内存模式（程序结束即丢失），真实项目可持久化到磁盘
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy",
)
print(f"第3步：向量库建立，存入 {len(chunks)} 个向量")


# ============================================================
# 第4步：检索器（Retriever）
# ============================================================
# 检索器 = 向量库的查询接口。给个问题，返回最相关的 k 段。
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # k=2：只要最相关 2 段


# ============================================================
# 第5步：组装 RAG 链
# ============================================================
# 逻辑：问题 → 检索相关段 → 【资料+问题】拼进 prompt → 模型作答
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """把多个 Document 拼成一段纯文本。"""
    return "\n\n---\n\n".join(d.page_content for d in docs)

# 防幻觉关键：prompt 里明确"只根据资料回答，没有就说不知道"
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是公司制度问答助手。请【只根据】下面资料回答，"
     "资料没有就答'根据现有资料无法回答'，不要编造。\n\n资料：\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {
        # context：检索器找资料 → 拼成文本
        "context": retriever | format_docs,
        # question：原样保留输入的问题
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 第6步：提问验证
# ============================================================
def ask(question: str):
    print(f"\n❓ {question}")
    # 先看检索器找了啥（调试 RAG 必看：检索准不准决定答案准不准）
    retrieved = retriever.invoke(question)
    print(f"   [检索到 {len(retrieved)} 段]")
    for i, d in enumerate(retrieved):
        print(f"    资料{i}: {d.page_content[:35].replace(chr(10),' ')}...")
    print(f"💡 {rag_chain.invoke(question)}")

print("\n" + "=" * 60)
print("RAG 问答演示")
print("=" * 60)

ask("入职满3年能休几天年假？")       # 资料有 → 准确答
ask("出差住宿费上限多少？")          # 资料有 → 准确答
ask("公司年会有什么奖品？")          # 资料没有 → 如实说不知道（防幻觉生效）

print("\n" + "=" * 60)
print("小结：")
print("1. RAG = 先检索相关资料，再让模型开卷作答")
print("2. 流程：文档→切块→向量化→向量库→检索→拼prompt→回答")
print("3. 切块参数(chunk_size/overlap)和检索 k 值，直接影响回答质量")
print("4. prompt 写'只根据资料回答'是防幻觉的关键")
print("5. 换真实资料：把 raw_text 换成文档加载器(如 PyPDFLoader)读 PDF 即可")
print("=" * 60)
