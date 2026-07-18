"""
05_rag_simple.py — 纯 Python 实现最简 RAG（检索增强生成）

为什么需要 RAG，而不是把全文塞进 context？
─────────────────────────────────────────
1. Token 上限：Claude 的 context window 再大也有上限，一本书几十万字直接放不进去。
2. 成本：Token 越多，API 费用越高；把 1000 页文档全传进去非常贵。
3. 注意力稀释：研究表明，当 context 很长时，模型对中间部分的关注度会下降
   （"Lost in the Middle" 问题），反而不如只放最相关的几段效果好。
4. 速度：更短的 prompt 响应更快。

RAG 的核心思路：
  用户提问 → 在知识库里找最相关的几段 → 只把这几段 + 问题送给 LLM → 得到回答

本文件用 numpy 实现余弦相似度，不依赖向量数据库，适合入门理解原理。
"""

import os
import math
import re
import anthropic
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 第一部分：文档加载
# ─────────────────────────────────────────────────────────────

def load_text_file(filepath: str) -> str:
    """从 txt 文件读取全部文本内容"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
# 第二部分：文档分块（Chunking）
#
# 为什么要分块？
# 把整篇文章当一个单元太大，检索时无法精准定位。
# 把每个句子当一个单元太小，缺少上下文。
# 分块是在"精准"和"上下文完整"之间找平衡。
# ─────────────────────────────────────────────────────────────

def chunk_by_paragraph(text: str) -> list[str]:
    """
    方式一：按段落分块
    以连续空行作为段落分隔符，过滤掉太短的段落（少于 10 个字）。
    优点：保留语义完整性；缺点：段落长度参差不齐。
    """
    # 用两个以上换行符切分段落
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for para in paragraphs:
        cleaned = para.strip()
        if len(cleaned) >= 10:   # 过滤空行 / 极短片段
            chunks.append(cleaned)
    return chunks


def chunk_by_word_count(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    方式二：按字数分块（中文按字，英文按词）

    参数：
        chunk_size : 每块的目标字数
        overlap    : 相邻块之间重叠的字数
                     重叠的目的是避免一句话被截断导致语义丢失

    实现思路：
        把文本拆成字列表，按 chunk_size 步长滑动窗口切分。
    """
    # 去除多余空白，把文本视为字符序列
    chars = list(text.replace("\n", " "))
    chunks = []
    start = 0
    while start < len(chars):
        end = start + chunk_size
        chunk = "".join(chars[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap   # 向前移动时保留 overlap 个字的重叠
    return chunks


# ─────────────────────────────────────────────────────────────
# 第三部分：TF-IDF 简化版关键词匹配
#
# TF-IDF 是什么？
# TF  = 词在当前文档中出现的频率（Term Frequency）
# IDF = 词在所有文档中的稀有程度（Inverse Document Frequency）
#       越稀有的词权重越高，避免"的/是/在"这类高频词干扰
# ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    简单分词：
    - 中文：按字切分
    - 英文：按空格/标点切分，转小写
    过滤掉长度 < 2 的 token（单字中文除外，单字可能有意义）
    """
    # 提取所有中文字符和英文单词
    tokens = re.findall(r"[一-鿿]|[a-zA-Z]{2,}", text)
    return [t.lower() for t in tokens]


def build_tfidf(chunks: list[str]) -> tuple[list[dict], dict]:
    """
    为所有分块构建 TF-IDF 向量（用 dict 稀疏表示）。

    返回：
        tf_list : 每个块的词频字典列表
        idf_map : 全局 IDF 字典
    """
    N = len(chunks)
    tf_list = []
    df_map = {}   # 每个词出现在多少块中

    # 第一步：计算每块的词频（TF）
    for chunk in chunks:
        tokens = tokenize(chunk)
        if not tokens:
            tf_list.append({})
            continue
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # 归一化：除以该块的总词数
        total = len(tokens)
        tf = {k: v / total for k, v in tf.items()}
        tf_list.append(tf)
        # 记录 DF：每个词在哪些块里出现过（只记录一次）
        for t in set(tokens):
            df_map[t] = df_map.get(t, 0) + 1

    # 第二步：计算 IDF
    idf_map = {}
    for term, df in df_map.items():
        # 加 1 平滑，避免除零
        idf_map[term] = math.log((N + 1) / (df + 1)) + 1

    return tf_list, idf_map


def tfidf_score(query: str, tf: dict, idf_map: dict) -> float:
    """计算 query 与某个块的 TF-IDF 相关度（点积）"""
    q_tokens = tokenize(query)
    score = 0.0
    for t in q_tokens:
        if t in tf and t in idf_map:
            score += tf[t] * idf_map[t]
    return score


# ─────────────────────────────────────────────────────────────
# 第四部分：余弦相似度（用纯 Python，不依赖 numpy）
#
# 余弦相似度衡量两个向量方向的接近程度（与长度无关），
# 非常适合文本相似度计算。值域 [-1, 1]，越接近 1 越相似。
# ─────────────────────────────────────────────────────────────

def cosine_similarity_dict(vec_a: dict, vec_b: dict) -> float:
    """
    计算两个稀疏向量（用 dict 表示）的余弦相似度。

    公式：cos(θ) = (A·B) / (|A| × |B|)
    """
    # 点积：只遍历共同的 key
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_b)
    # 各自的模长
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def query_tfidf_vector(query: str, idf_map: dict) -> dict:
    """把 query 也转成 TF-IDF 向量，以便计算余弦相似度"""
    tokens = tokenize(query)
    if not tokens:
        return {}
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens)
    tfidf = {}
    for t, cnt in tf.items():
        tfidf[t] = (cnt / total) * idf_map.get(t, 1.0)
    return tfidf


# ─────────────────────────────────────────────────────────────
# 第五部分：检索函数
# ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    chunks: list[str],
    tf_list: list[dict],
    idf_map: dict,
    top_k: int = 3,
) -> list[tuple[float, str]]:
    """
    检索与 query 最相关的 top_k 个文本块。

    策略：用 TF-IDF 余弦相似度打分，取分最高的 top_k 块。

    返回：[(score, chunk_text), ...] 按 score 降序排列
    """
    q_vec = query_tfidf_vector(query, idf_map)
    if not q_vec:
        return []

    scores = []
    for i, (chunk, tf) in enumerate(zip(chunks, tf_list)):
        # 把块的 TF 也乘以 IDF，得到块的 TF-IDF 向量
        chunk_vec = {t: v * idf_map.get(t, 1.0) for t, v in tf.items()}
        score = cosine_similarity_dict(q_vec, chunk_vec)
        scores.append((score, chunk))

    # 按分数降序排列，取前 top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# ─────────────────────────────────────────────────────────────
# 第六部分：完整 RAG 流程
# ─────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: list[str],
    tf_list: list[dict],
    idf_map: dict,
    top_k: int = 3,
) -> str:
    """
    完整的 RAG 流程：
    1. 检索最相关的 top_k 块
    2. 把这些块拼成「参考资料」
    3. 连同问题一起送给 Claude
    4. 返回 Claude 的回答

    这就是 RAG 的精髓：LLM 只看最相关的片段，不看全文。
    """
    client = anthropic.Anthropic()

    # Step 1: 检索
    results = retrieve(question, chunks, tf_list, idf_map, top_k=top_k)
    if not results:
        return "抱歉，知识库中未找到相关内容。"

    # Step 2: 构建 context
    context_parts = []
    for rank, (score, chunk) in enumerate(results, 1):
        context_parts.append(f"【参考片段 {rank}（相关度 {score:.3f}）】\n{chunk}")
    context = "\n\n".join(context_parts)

    # Step 3: 构建 prompt
    prompt = f"""请根据以下参考资料回答问题。如果参考资料中没有足够信息，请如实说明，不要编造。

参考资料：
{context}

问题：{question}

请用简洁清晰的语言回答："""

    # Step 4: 调用 Claude
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ─────────────────────────────────────────────────────────────
# 第七部分：演示
# ─────────────────────────────────────────────────────────────

def create_demo_knowledge_base(filepath: str) -> None:
    """创建一个示例知识库文件（关于 AI 工具的简单介绍）"""
    content = """Claude 是 Anthropic 公司开发的 AI 助手。
Claude 擅长写作、分析、编程和问答等任务。
Claude 支持长上下文，能处理大量文本。

Cursor 是一款 AI 驱动的代码编辑器，基于 VSCode 开发。
Cursor 内置了代码补全、对话式编程和代码解释功能。
使用 Cursor 可以大幅提升编程效率，尤其适合初学者。

RAG 是 Retrieval-Augmented Generation（检索增强生成）的缩写。
RAG 的核心思想是：先从知识库中检索相关内容，再交给 LLM 生成回答。
RAG 可以解决 LLM 知识截止日期的问题，让模型回答最新信息。
RAG 比直接把全文塞进 context 更节省 token，也更精准。

向量数据库是专门存储和检索向量（embedding）的数据库。
常见的向量数据库有 ChromaDB、Pinecone、Weaviate 等。
向量数据库的核心操作是「近似最近邻搜索」（ANN search）。

Embedding 是把文本转换成数字向量的过程。
语义相近的文本，其 embedding 向量在空间中距离也更近。
常用的 embedding 模型有 OpenAI text-embedding-3、VoyageAI 等。

Claude Code 是 Anthropic 推出的 AI 编程助手命令行工具。
Claude Code 可以读取项目文件、运行命令、修改代码。
Claude Code 适合完成需要跨文件理解的复杂编程任务。
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已创建示例知识库：{filepath}")


def main() -> None:
    print("=" * 60)
    print("05_rag_simple.py — 纯 Python 最简 RAG 演示")
    print("=" * 60)

    # 1. 创建示例知识库
    kb_path = "/tmp/demo_knowledge_base.txt"
    create_demo_knowledge_base(kb_path)

    # 2. 加载文档
    text = load_text_file(kb_path)
    print(f"\n知识库字数：{len(text)} 字")

    # 3. 两种分块方式对比
    para_chunks = chunk_by_paragraph(text)
    word_chunks = chunk_by_word_count(text, chunk_size=100, overlap=15)
    print(f"\n按段落分块：{len(para_chunks)} 块")
    print(f"按字数分块（每块100字，重叠15字）：{len(word_chunks)} 块")

    # 使用段落分块进行演示
    chunks = para_chunks
    print(f"\n使用段落分块，共 {len(chunks)} 块")
    print("─" * 40)
    for i, c in enumerate(chunks):
        print(f"块 {i+1}: {c[:50]}...")

    # 4. 构建 TF-IDF 索引
    tf_list, idf_map = build_tfidf(chunks)
    print(f"\nTF-IDF 索引词汇量：{len(idf_map)} 个词")

    # 5. 测试检索
    test_queries = [
        "RAG 是什么？有什么好处？",
        "Cursor 编辑器有哪些功能？",
        "向量数据库怎么用？",
    ]

    print("\n" + "=" * 60)
    print("纯检索测试（不调用 LLM）")
    print("=" * 60)
    for query in test_queries:
        print(f"\n查询：{query}")
        results = retrieve(query, chunks, tf_list, idf_map, top_k=2)
        for rank, (score, chunk) in enumerate(results, 1):
            print(f"  Top{rank} (score={score:.4f}): {chunk[:80]}...")

    # 6. 完整 RAG 问答（需要 ANTHROPIC_API_KEY）
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 60)
        print("完整 RAG 问答（调用 Claude）")
        print("=" * 60)
        question = "RAG 相比直接把全文塞进 context 有什么优势？"
        print(f"\n问题：{question}\n")
        answer = generate_answer(question, chunks, tf_list, idf_map, top_k=3)
        print(f"回答：\n{answer}")
    else:
        print("\n（未设置 ANTHROPIC_API_KEY，跳过 LLM 问答部分）")
        print("提示：设置环境变量后可看到完整 RAG 效果")


if __name__ == "__main__":
    main()
