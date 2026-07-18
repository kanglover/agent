# 运行: python examples/python/_billing_patch.py examples/python/06b_rag_runnable.py
# 依赖: pip install chromadb   (embedding 用阿里 DashScope，key 已在环境变量里)
#
# 说明: 这是 06_rag_advanced.py 的「可运行版」。
#       原版依赖 VoyageAI（需要单独 key），这里改用环境里现成的 DashScope，
#       所以能真正跑通「语义检索」——不是原版的 hash 假向量。
#       为了好懂，这里砍掉了 BM25 / RRF / 重排序等复杂部分，
#       只保留 RAG 最核心的一条线：分块 → 向量化 → 存库 → 检索 → 回答。
#
# 类比：06 像一辆改装赛车（零件多但难启动），这个版本像一辆能直接上路的家用车。
#       先把基本流程跑通看懂，再回去读 06 的进阶部分。

from __future__ import annotations

import os
import json
import math
import hashlib
import urllib.request
from typing import Any

import anthropic

# ── 可选依赖：chromadb 没装就用内存存储 ──────────────────────
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("提示：chromadb 未安装，将用内存存储（pip install chromadb 体验持久化）")


# ============================================================
# 第一步：把长文章切成小块（Chunking）
# ============================================================
# 为什么要切块？
#   一整篇文章太长，检索时不好定位「哪一段和问题相关」。
#   切成小块后，可以精准找出最相关的几块喂给 AI。
#   类比：一本书不拆分，你只能整本翻；拆成章节段落，就能直接翻到要找的那页。

def chunk_text(text: str, max_size: int = 200, overlap: int = 40) -> list[str]:
    """
    按段落 + 句号切分，每块不超过 max_size 字，相邻块有 overlap 字重叠。

    overlap（重叠）的作用：
      避免一句话正好被切到两块里、谁都不完整。
      让相邻块共享一小段尾巴，检索时两边都能覆盖到这句话。
    """
    # 先按空行/换行切成粗块
    rough = [p for p in text.replace("\n\n", "\n").split("\n") if p.strip()]
    chunks: list[str] = []
    for para in rough:
        # 段落太长就按句号继续切
        if len(para) <= max_size:
            chunks.append(para.strip())
            continue
        sentences = para.split("。")
        current = ""
        for s in sentences:
            piece = s + "。"
            if len(current) + len(piece) <= max_size:
                current += piece
            else:
                if current:
                    chunks.append(current.strip())
                current = piece
        if current:
            chunks.append(current.strip())

    # 加重叠：每块前面拼上上一块的末尾 overlap 个字
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + chunks[i])
    return result


# ============================================================
# 第二步：把文字变成向量（Embedding）
# ============================================================
# 什么是 embedding？
#   把一段文字转成一串数字（向量）。语义相近的文字，向量也相近。
#   这样「找相关段落」就变成「找距离最近的向量」，计算机很擅长算距离。
#   类比：给每段文字标一个 GPS 坐标，意思接近的段落坐标也接近，
#         查资料就像在地图上找最近的几个点。

def dashscope_embed(texts: list[str]) -> list[list[float]]:
    """
    用阿里 DashScope 的 text-embedding-v3 模型生成向量。

    为什么用 DashScope 而不是原版的 VoyageAI？
      VoyageAI 需要单独申请 key，公司 Claude 网关也不提供 embeddings。
      而 DASHSCOPE_API_KEY 在当前环境已经配好，直接能用，且是真正的语义向量。
      用标准库 urllib 调它的 OpenAI 兼容接口，不用装额外 SDK。

    小知识：DashScope 单次最多 25 条文本，所以这里分批发送。
    """
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i + 25]
        payload = json.dumps({
            "model": "text-embedding-v3",
            "input": batch,
            "dimensions": 1024,
            "encoding_format": "float",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        # 按 index 排好序，保证顺序和输入一致
        batch_sorted = sorted(body["data"], key=lambda d: d["index"])
        all_embeddings.extend([d["embedding"] for d in batch_sorted])
    return all_embeddings


def hash_embed(text: str, dim: int = 128) -> list[float]:
    """
    降级方案：没有 DashScope key 时，用哈希造一个假向量。
    注意：这种向量没有真正的语义含义，只是保证程序能跑，检索质量很差。
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    state = seed
    vec = []
    for _ in range(dim):
        state = (state * 1664525 + 1013904223) % (2 ** 32)
        vec.append((state / 2 ** 32) * 2 - 1)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def embed(texts: list[str]) -> list[list[float]]:
    """统一入口：有 DashScope key 就用真向量，否则降级到 hash 假向量。"""
    if os.environ.get("DASHSCOPE_API_KEY"):
        print(f"  → 使用 DashScope text-embedding-v3 向量化（{len(texts)} 条）")
        return dashscope_embed(texts)
    print(f"  → 未配置 DASHSCOPE_API_KEY，使用 hash 假向量（{len(texts)} 条，质量低）")
    return [hash_embed(t) for t in texts]


# ============================================================
# 第三步：把向量存进数据库（向量检索）
# ============================================================
# 什么是向量数据库？
#   专门存「向量」并支持「快速找最近向量」的数据库。
#   这里用 ChromaDB（开源、能存到本地磁盘）。
#   类比：普通数据库按「关键字」查，向量数据库按「意思像不像」查。

class VectorStore:
    """封装向量存储：有 chromadb 就用真数据库，没有就用内存列表线性扫描。"""

    def __init__(self, persist_dir: str = "/tmp/chroma_rag_runnable"):
        self.persist_dir = persist_dir
        self._mem: list[dict] = []  # 内存降级用
        if HAS_CHROMA:
            self.client = chromadb.PersistentClient(path=persist_dir)
            # 每次演示都清掉旧数据，避免维度不一致（hash=128 vs dashscope=1024）
            try:
                self.client.delete_collection("rag_runnable")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="rag_runnable",
                metadata={"hnsw:space": "cosine"},  # 用余弦距离衡量相似度
            )
            print(f"  → ChromaDB 数据库就绪（持久化到 {persist_dir}）")
        else:
            print("  → chromadb 不可用，使用内存向量存储")

    def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        if HAS_CHROMA:
            self.collection.add(
                ids=[f"c{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
            )
        else:
            for i, (c, e) in enumerate(zip(chunks, embeddings)):
                self._mem.append({"id": f"c{i}", "doc": c, "emb": e})

    def search(self, query_emb: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        """返回最相似的 top_k 个块：[(文本, 相似度分数), ...]"""
        if HAS_CHROMA:
            res = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
            docs = res["documents"][0]
            dists = res["distances"][0]  # 余弦距离：越小越像
            return [(d, 1 - dist) for d, dist in zip(docs, dists)]

        # 内存降级：手算余弦相似度，逐个比较（数据多时慢）
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb + 1e-9)

        scored = [(m["doc"], cosine(query_emb, m["emb"])) for m in self._mem]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 第四步：让 Claude 根据检索到的资料回答问题
# ============================================================
def answer_with_context(question: str, context: str) -> str:
    """
    把检索到的资料塞进 prompt，让 Claude「照着资料回答」。
    这一步是 RAG 的「G」(Generation)。
    """
    client = anthropic.Anthropic()
    prompt = f"""请只根据下面【参考资料】回答问题。
如果资料里没有相关信息，就直接说「资料中未提及」，不要自己编造。

【参考资料】
{context}

【问题】
{question}"""

    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ============================================================
# 知识库：5 段关于 RAG 各组件的介绍（演示用的小图书馆）
# ============================================================
KNOWLEDGE_BASE = [
    "ChromaDB 是一个开源的向量数据库，专为 AI 应用设计。它支持本地运行，无需联网。"
    "核心功能是存储 embedding 向量并做高效的相似度检索。用 PersistentClient 可以把数据存到磁盘，重启不丢失。",

    "VoyageAI 是专为 RAG 优化的 embedding 服务，voyage-3 模型在检索基准上表现优秀。"
    "它区分 document 和 query 两种输入类型，分别优化建库和检索阶段。",

    "BM25 是经典的信息检索算法，是 TF-IDF 的改进版。它加入了文档长度归一化，"
    "避免长文档因词频高而得分虚高。BM25 擅长精确关键词匹配。",

    "向量检索擅长语义匹配——换了说法也能找到相关内容。"
    "混合检索把向量检索和 BM25 结合，通常比单一方法效果更好。",

    "RAG 质量评估常用 LLM-as-Judge：让大模型当评委打分。"
    "主要看忠实度（回答是否基于资料、没编造）、相关性、完整性等维度。",
]


# ============================================================
# 主流程：把上面四步串起来
# ============================================================
def main() -> None:
    print("=" * 60)
    print("06b_rag_runnable.py — 可直接运行的 RAG 演示（DashScope + ChromaDB + Claude）")
    print("=" * 60)

    # 1. 切块
    print("\n【1】切块")
    chunks = chunk_text("\n".join(KNOWLEDGE_BASE))
    print(f"  知识库 {len(KNOWLEDGE_BASE)} 段 → 切成 {len(chunks)} 块")

    # 2. 向量化 + 存库
    print("\n【2】向量化并入库")
    embeddings = embed(chunks)
    store = VectorStore()
    store.add(chunks, embeddings)
    print(f"  已存入 {len(chunks)} 条向量记录")

    # 3. 提问 → 检索 → 回答
    question = "向量检索和 BM25 各擅长什么？混合检索为什么更好？"
    print(f"\n【3】提问：{question}")

    print("\n【3.1】把问题也变成向量，去库里找最相关的 2 块")
    q_emb = embed([question])[0]
    hits = store.search(q_emb, top_k=2)
    context_parts = []
    for i, (text, score) in enumerate(hits, 1):
        print(f"  命中 [{i}] 相似度={score:.3f}：{text[:60]}...")
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    print("\n【3.2】把检索到的资料交给 Claude 回答")
    answer = answer_with_context(question, context)
    print("  Claude 回答：")
    for line in answer.splitlines():
        print(f"    {line}")

    # 4. 对比演示：问一个知识库里没有的问题，看 RAG 如何「拒绝编造」
    print("\n【4】对比：问一个资料里没有的问题")
    off_question = "今天北京的天气怎么样？"
    print(f"  提问：{off_question}")
    off_emb = embed([off_question])[0]
    off_hits = store.search(off_emb, top_k=2)
    off_context = "\n\n".join(t for t, _ in off_hits)
    off_answer = answer_with_context(off_question, off_context)
    print(f"  Claude 回答：{off_answer}")
    print("  （理想情况：Claude 会说资料中未提及，而不是瞎编天气）")

    print("\n✅ RAG 演示完成！")
    print("\n学到了什么？")
    print("  1. 切块 chunk_text     = 把长文章切成能精准定位的小段")
    print("  2. 向量化 embed        = 把文字变成可比较距离的数字向量")
    print("  3. 存储 VectorStore    = 用 ChromaDB 按「意思像不像」快速检索")
    print("  4. 回答 answer_with_context = 让 Claude 只根据资料回答，减少编造")


if __name__ == "__main__":
    main()
