"""
06_rag_advanced.py — 生产级 RAG：ChromaDB + VoyageAI + 混合检索 + LLM 重排序

依赖安装：
    pip install chromadb anthropic voyageai pypdf

本文件涵盖：
  1. RecursiveChunker 类（带重叠分块）
  2. VoyageAI Embedding（无 key 时退回 hash embedding）
  3. ChromaDB 向量数据库（持久化到本地磁盘）
  4. 混合检索：向量相似度 + BM25，用 RRF 融合
  5. LLM 重排序（把 top-5 交给 Claude 选 top-2）
  6. RAG 质量评估（faithfulness 0-10 分）
  7. PDF 支持（可选，依赖 pypdf）
"""

from __future__ import annotations

import os
import re
import json
import math
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import anthropic

# ── 可选依赖：graceful import ──────────────────────────────────
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("警告：chromadb 未安装，将使用内存模拟向量存储（pip install chromadb）")

try:
    import voyageai
    HAS_VOYAGE = True
except ImportError:
    HAS_VOYAGE = False
    print("警告：voyageai 未安装，将使用 hash embedding（pip install voyageai）")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("警告：pypdf 未安装，将跳过 PDF 支持（pip install pypdf）")


# ─────────────────────────────────────────────────────────────
# 第一部分：RecursiveChunker
#
# 什么是递归分块？
# 先尝试用最大分隔符（段落）切，如果块还是太大，
# 再用次级分隔符（句子）切，以此类推。
# 这样能保证每块不超过 max_size，同时尽量保留语义完整性。
# ─────────────────────────────────────────────────────────────

class RecursiveChunker:
    """
    递归字符分块器，支持重叠（overlap）。

    参数：
        max_size  : 每块最大字数（默认 300）
        overlap   : 相邻块重叠字数（默认 50）
                    重叠的目的：避免一个完整的语句被切断到两个块里，
                    检索时两个块都能覆盖到这句话。
        separators: 分割符优先级列表，从大到小
    """

    def __init__(
        self,
        max_size: int = 300,
        overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.max_size = max_size
        self.overlap = overlap
        # 按语义粒度从粗到细排列分隔符
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def chunk(self, text: str) -> list[str]:
        """把文本切成不超过 max_size 的块，相邻块有 overlap 字重叠"""
        chunks = self._recursive_split(text, self.separators)
        return self._apply_overlap(chunks)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """递归地用 separators[0] 切分，太大的块继续用更细的分隔符切"""
        if len(text) <= self.max_size:
            return [text] if text.strip() else []

        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            # 最后兜底：直接按字数硬切
            return [text[i:i + self.max_size] for i in range(0, len(text), self.max_size)]

        parts = text.split(sep)
        result = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.max_size:
                current = candidate
            else:
                if current.strip():
                    result.append(current.strip())
                # 如果单个 part 还是太大，用更细的分隔符递归切
                if len(part) > self.max_size and next_seps:
                    result.extend(self._recursive_split(part, next_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            result.append(current.strip())

        return result

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        在相邻块之间加入重叠：
        对于第 i 块（i > 0），在它前面拼接第 i-1 块的末尾 overlap 个字。
        """
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.overlap:]   # 取上一块末尾
            merged = prev_tail + chunks[i]
            result.append(merged)
        return result


# ─────────────────────────────────────────────────────────────
# 第二部分：Embedding 模型
#
# Embedding 是把文本变成数字向量的过程。
# 语义相近的文本，向量之间的夹角更小（余弦相似度更高）。
# VoyageAI 是专门为 RAG 优化的 embedding 服务。
# ─────────────────────────────────────────────────────────────

def get_voyage_embedding(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    调用 VoyageAI 获取 embedding。

    input_type:
        "document" — 用于索引知识库（建库时用）
        "query"    — 用于查询（检索时用）
        这个区分能提升检索精度。
    """
    vo = voyageai.Client()   # 自动读取 VOYAGE_API_KEY 环境变量
    result = vo.embed(texts, model="voyage-3", input_type=input_type)
    return result.embeddings


def hash_embedding(text: str, dim: int = 128) -> list[float]:
    """
    Fallback：用 MD5 哈希生成伪 embedding（仅演示用，语义无意义）。
    当没有 VoyageAI key 时使用，保证程序能跑通但检索质量很低。
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng_state = seed
    vec = []
    for _ in range(dim):
        # 线性同余伪随机数
        rng_state = (rng_state * 1664525 + 1013904223) % (2 ** 32)
        vec.append((rng_state / 2 ** 32) * 2 - 1)   # 归一化到 [-1, 1]
    # 归一化为单位向量（余弦相似度计算要求）
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def get_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """统一接口：优先用 VoyageAI，否则 fallback 到 hash embedding"""
    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if HAS_VOYAGE and voyage_key:
        print(f"使用 VoyageAI embedding（{len(texts)} 条）...")
        return get_voyage_embedding(texts, input_type=input_type)
    else:
        print(f"使用 hash embedding fallback（{len(texts)} 条，仅演示）...")
        return [hash_embedding(t) for t in texts]


# ─────────────────────────────────────────────────────────────
# 第三部分：ChromaDB 向量数据库
#
# ChromaDB 是什么？
# 一个开源的本地向量数据库，可以持久化到磁盘。
# 你可以把 embedding 存进去，之后用向量相似度快速检索。
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    封装 ChromaDB 操作。
    如果 chromadb 未安装，退回到内存 list 线性扫描。
    """

    def __init__(self, persist_dir: str = "/tmp/chroma_rag_demo"):
        self.persist_dir = persist_dir
        self._memory_store: list[dict] = []   # fallback 用

        if HAS_CHROMA:
            # PersistentClient：数据保存到磁盘，重启后不丢失
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="rag_demo",
                metadata={"hnsw:space": "cosine"},   # 用余弦距离
            )
            print(f"ChromaDB 持久化路径：{persist_dir}")
        else:
            print("ChromaDB 不可用，使用内存向量存储")

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """把分块和对应的 embedding 存入数据库"""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = metadatas or [{"source": "unknown"}] * len(chunks)

        if HAS_CHROMA:
            # ChromaDB 批量添加
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            # 内存存储
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self._memory_store.append({
                    "id": ids[i],
                    "document": chunk,
                    "embedding": emb,
                    "metadata": metadatas[i],
                })

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        向量检索：找出与 query_embedding 最相似的 top_k 个文档。
        返回格式：[{"text": ..., "score": ..., "metadata": ...}, ...]
        """
        if HAS_CHROMA:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            docs = results["documents"][0]
            distances = results["distances"][0]   # ChromaDB 余弦距离：越小越相似
            metadatas = results["metadatas"][0]
            return [
                {"text": doc, "score": 1 - dist, "metadata": meta}   # 转成相似度
                for doc, dist, meta in zip(docs, distances, metadatas)
            ]
        else:
            # 内存线性扫描（O(n)，数据量大时很慢）
            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(x * x for x in b))
                return dot / (na * nb + 1e-9)

            scored = [
                {"text": item["document"], "score": cosine(query_embedding, item["embedding"]), "metadata": item["metadata"]}
                for item in self._memory_store
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

    def count(self) -> int:
        """返回数据库中的文档数量"""
        if HAS_CHROMA:
            return self.collection.count()
        return len(self._memory_store)

    def clear(self) -> None:
        """清空数据库（重建演示时用）"""
        if HAS_CHROMA:
            self.client.delete_collection("rag_demo")
            self.collection = self.client.get_or_create_collection(
                name="rag_demo",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._memory_store.clear()


# ─────────────────────────────────────────────────────────────
# 第四部分：BM25 关键词检索
#
# BM25 是什么？
# 经典的信息检索算法，是 TF-IDF 的改进版。
# 加入了文档长度归一化（长文档不会因为词出现次数多就得高分）。
# 混合检索 = 向量检索 + BM25，两者互补：
#   向量检索擅长语义匹配（换了说法也能找到）
#   BM25 擅长精确关键词匹配（原词更准）
# ─────────────────────────────────────────────────────────────

class BM25:
    """
    BM25 简化实现。
    参数 k1=1.5, b=0.75 是经验值，一般不需要调整。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_freqs: list[dict] = []
        self.idf: dict = {}
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[一-鿿]|[a-zA-Z]{2,}", text.lower())

    def fit(self, documents: list[str]) -> None:
        """用文档集合构建 BM25 索引"""
        self.corpus = [self._tokenize(doc) for doc in documents]
        N = len(documents)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(N, 1)

        # 计算 IDF
        df: dict = {}
        for doc_tokens in self.corpus:
            for t in set(doc_tokens):
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for t, freq in df.items()
        }
        # 计算每文档的词频
        self.doc_freqs = []
        for doc_tokens in self.corpus:
            freq: dict = {}
            for t in doc_tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)

    def score(self, query: str, doc_idx: int) -> float:
        """计算 query 对第 doc_idx 篇文档的 BM25 得分"""
        q_tokens = self._tokenize(query)
        doc_len = len(self.corpus[doc_idx])
        score = 0.0
        freq = self.doc_freqs[doc_idx]
        for t in q_tokens:
            if t not in freq:
                continue
            tf = freq[t]
            idf = self.idf.get(t, 0)
            # BM25 公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def query(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(doc_idx, score), ...] 按分数降序"""
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─────────────────────────────────────────────────────────────
# 第五部分：RRF 融合（Reciprocal Rank Fusion）
#
# RRF 是什么？
# 把多个排序列表合并成一个的简单算法。
# 核心思想：排名越靠前，贡献越大；用名次的倒数加权。
# 公式：RRF_score(d) = Σ 1/(k + rank_i(d))
# k=60 是经验常数，用于平滑排名差距。
# ─────────────────────────────────────────────────────────────

def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    把多个文本排名列表融合为一个。

    参数：
        ranked_lists : 多个排序结果（每个是文本列表，从高到低）
        k            : RRF 平滑常数（默认 60）

    返回：
        [(text, rrf_score), ...] 按分数降序
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, text in enumerate(ranked, start=1):
            scores[text] = scores.get(text, 0) + 1 / (k + rank)
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result


# ─────────────────────────────────────────────────────────────
# 第六部分：LLM 重排序
#
# 为什么需要 LLM 重排序？
# 向量/BM25 检索只看字面或向量相似度，
# 但 Claude 能理解语义、逻辑关系，判断哪段真正能回答问题。
# 把 top-5 候选段落给 Claude 看，让它选出最有用的 top-2。
# ─────────────────────────────────────────────────────────────

def llm_rerank(
    query: str,
    candidates: list[str],
    top_k: int = 2,
) -> list[str]:
    """
    用 Claude 对候选段落重排序，返回最相关的 top_k 个。

    实现方式：让 Claude 以 JSON 格式返回最相关段落的编号。
    """
    client = anthropic.Anthropic()

    numbered = "\n\n".join(
        f"[{i+1}] {text}" for i, text in enumerate(candidates)
    )

    prompt = f"""下面是 {len(candidates)} 段参考文本，请判断哪些段落最能回答用户问题。

用户问题：{query}

候选段落：
{numbered}

请从中选出最相关的 {top_k} 段，按相关度从高到低排列。
只返回 JSON 格式，例如：{{"selected": [2, 1]}}
不要解释，只返回 JSON。"""

    message = client.messages.create(
        model="claude-haiku-4-5",   # 重排序用轻量模型节省成本
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw = message.content[0].text.strip()
        # 提取 JSON 部分
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            indices = data.get("selected", [])
            return [candidates[i - 1] for i in indices if 1 <= i <= len(candidates)]
    except Exception as e:
        print(f"重排序解析失败：{e}，返回原序前 {top_k} 个")

    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────
# 第七部分：RAG 质量评估 — Faithfulness 评分
#
# Faithfulness 是什么？
# 衡量模型的回答是否完全基于提供的参考资料，没有"自编"内容。
# 分数 0-10：10 分表示完全忠实于参考资料，0 分表示完全编造。
# ─────────────────────────────────────────────────────────────

def evaluate_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> dict:
    """
    用 Claude 评估 RAG 回答的 faithfulness（忠实度）。
    返回：{"score": int, "reason": str}
    """
    client = anthropic.Anthropic()

    prompt = f"""你是 RAG 质量评估专家。请评估以下回答的"忠实度"（faithfulness）。

忠实度定义：回答中的每个陈述都能在参考资料中找到依据，没有基于模型自身知识编造内容。

问题：{question}

参考资料：
{context}

回答：
{answer}

请按以下格式给出评分（JSON）：
{{
  "score": <0到10的整数，10分最忠实>,
  "reason": "<50字以内的简短理由>"
}}
只返回 JSON，不要其他内容。"""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw = message.content[0].text.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"评估解析失败：{e}")

    return {"score": -1, "reason": "解析失败"}


# ─────────────────────────────────────────────────────────────
# 第八部分：PDF 支持
# ─────────────────────────────────────────────────────────────

def load_pdf(filepath: str) -> str:
    """从 PDF 文件提取纯文本（需要 pypdf）"""
    if not HAS_PYPDF:
        raise ImportError("请先安装 pypdf：pip install pypdf")
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


# ─────────────────────────────────────────────────────────────
# 第九部分：完整 Advanced RAG 流程
# ─────────────────────────────────────────────────────────────

class AdvancedRAG:
    """把所有组件整合在一起的完整 RAG 系统"""

    def __init__(self, persist_dir: str = "/tmp/chroma_rag_advanced"):
        self.chunker = RecursiveChunker(max_size=300, overlap=50)
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.bm25 = BM25()
        self.chunks: list[str] = []

    def build_index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """
        从文档列表构建检索索引。
        步骤：分块 → 生成 embedding → 存入 ChromaDB + 构建 BM25 索引
        """
        print("\n🔨 构建索引...")

        # 1. 分块
        all_chunks = []
        all_metas = []
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
            meta = metadatas[doc_idx] if metadatas else {"doc_idx": doc_idx}
            all_metas.extend([meta] * len(chunks))

        self.chunks = all_chunks
        print(f"分块完成：{len(all_chunks)} 块")

        # 2. 生成 embedding（向量化）
        embeddings = get_embeddings(all_chunks, input_type="document")

        # 3. 存入向量数据库
        self.vector_store.clear()   # 清除旧数据（演示用）
        self.vector_store.add_documents(all_chunks, embeddings, all_metas)
        print(f"向量数据库已存储 {self.vector_store.count()} 条记录")

        # 4. 构建 BM25 索引
        self.bm25.fit(all_chunks)
        print("BM25 索引构建完成")

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> list[str]:
        """
        混合检索：向量 + BM25，用 RRF 融合排名。

        1. 向量检索：生成 query embedding，查向量数据库
        2. BM25 检索：关键词匹配
        3. RRF 融合两个排名列表
        """
        # Step 1: 向量检索
        q_emb = get_embeddings([query], input_type="query")[0]
        vector_results = self.vector_store.query(q_emb, top_k=top_k)
        vector_ranked = [r["text"] for r in vector_results]

        # Step 2: BM25 检索
        bm25_results = self.bm25.query(query, top_k=top_k)
        bm25_ranked = [self.chunks[idx] for idx, _ in bm25_results]

        # Step 3: RRF 融合
        fused = rrf_fusion([vector_ranked, bm25_ranked])
        return [text for text, _ in fused[:top_k]]

    def answer(
        self,
        question: str,
        use_rerank: bool = True,
        evaluate: bool = False,
    ) -> dict:
        """
        完整 RAG 问答流程：
        检索 → (重排序) → 生成回答 → (质量评估)
        """
        client = anthropic.Anthropic()

        # 1. 混合检索 top-5
        candidates = self.retrieve_hybrid(question, top_k=5)
        print(f"\n混合检索到 {len(candidates)} 个候选段落")

        # 2. LLM 重排序（可选）
        if use_rerank and os.environ.get("ANTHROPIC_API_KEY"):
            print("LLM 重排序中...")
            final_chunks = llm_rerank(question, candidates, top_k=2)
        else:
            final_chunks = candidates[:2]

        context = "\n\n".join(final_chunks)

        # 3. 生成回答
        prompt = f"""请根据以下参考资料回答问题。如参考资料信息不足，请说明。

参考资料：
{context}

问题：{question}"""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = message.content[0].text

        result = {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": final_chunks,
        }

        # 4. 质量评估（可选）
        if evaluate:
            print("评估 faithfulness...")
            eval_result = evaluate_faithfulness(question, context, answer_text)
            result["faithfulness"] = eval_result
            print(f"Faithfulness 评分：{eval_result['score']}/10 — {eval_result['reason']}")

        return result


# ─────────────────────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────────────────────

DEMO_DOCUMENTS = [
    """ChromaDB 是一个开源的向量数据库，专为 AI 应用设计。
它支持本地运行（无需联网），也支持部署为服务。
ChromaDB 的核心功能是存储 embedding 向量并进行高效的相似度检索。
使用 PersistentClient 可以把数据持久化到磁盘，重启后不丢失。
ChromaDB 支持余弦距离、欧氏距离等多种距离函数。""",

    """VoyageAI 是专为 RAG 优化的 embedding 服务。
它的 voyage-3 模型在 MTEB 检索基准上表现优秀。
VoyageAI 区分 document 和 query 两种输入类型，分别优化建库和检索阶段。
使用方式：先 pip install voyageai，再设置 VOYAGE_API_KEY 环境变量。""",

    """BM25 是经典的信息检索算法，是 TF-IDF 的改进版本。
BM25 加入了文档长度归一化，避免长文档因词频高而得分虚高。
BM25 擅长精确关键词匹配，向量检索擅长语义匹配。
混合检索结合两者优势，通常比单一方法效果更好。""",

    """RRF（Reciprocal Rank Fusion）是一种简单有效的排名融合算法。
公式：对每个文档，把它在各排名列表中名次的倒数求和。
k=60 是经验常数，用于减小头部排名之间的差距。
RRF 不需要归一化各系统的分数，使用非常方便。""",

    """RAG 质量评估包括多个维度：
- Faithfulness（忠实度）：回答是否基于参考资料，没有编造
- Relevance（相关性）：检索到的段落是否与问题相关
- Completeness（完整性）：回答是否覆盖了问题的所有方面
用 LLM 作为评估器（LLM-as-Judge）是目前主流的自动化评估方式。""",
]


def main() -> None:
    print("=" * 60)
    print("06_rag_advanced.py — 生产级 Advanced RAG 演示")
    print("=" * 60)

    # 演示 RecursiveChunker
    print("\n── RecursiveChunker 演示 ──")
    chunker = RecursiveChunker(max_size=100, overlap=20)
    sample = DEMO_DOCUMENTS[0]
    chunks = chunker.chunk(sample)
    print(f"原文 {len(sample)} 字，分成 {len(chunks)} 块：")
    for i, c in enumerate(chunks):
        print(f"  块{i+1}（{len(c)}字）: {c[:50]}...")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n（未设置 ANTHROPIC_API_KEY，跳过 LLM 相关演示）")
        print("设置 ANTHROPIC_API_KEY 后可体验完整 Advanced RAG 流程")

        # 非 LLM 部分：演示混合检索
        print("\n── 混合检索演示（无需 API Key）──")
        rag = AdvancedRAG()
        rag.build_index(DEMO_DOCUMENTS)
        results = rag.retrieve_hybrid("BM25 和向量检索有什么区别？", top_k=3)
        print("检索结果：")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r[:80]}...")
        return

    # 完整流程
    rag = AdvancedRAG()
    rag.build_index(DEMO_DOCUMENTS)

    question = "混合检索相比单一检索方式有什么优势？"
    print(f"\n问题：{question}")
    result = rag.answer(question, use_rerank=True, evaluate=True)
    print(f"\n回答：\n{result['answer']}")


if __name__ == "__main__":
    main()
