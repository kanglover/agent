"""
Embedding 与向量存储完整示例
涵盖：
  1. Voyage / OpenAI Embedding API 调用
  2. 余弦相似度、点积、欧氏距离对比
  3. ChromaDB 完整 CRUD
  4. 元数据过滤（source、date、type）
  5. 批量向量化（大量文档）
  6. 增量更新（新增文档不重建全库）
  7. 持久化（ChromaDB persist）
  8. 多集合管理
  9. 向量库迁移（从内存到磁盘到云端）

依赖安装：
  pip install chromadb openai voyageai numpy
"""

import os
import math
import time
import uuid
import tempfile
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────
# 第 1 节：Embedding API 调用（Voyage / OpenAI）
# ─────────────────────────────────────────────

def embed_with_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """使用 OpenAI Embedding API 将文本转为向量。"""
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"[OpenAI Embed] 跳过（未配置 API Key 或网络错误）: {e}")
        # 返回假向量，方便本地演示
        return [[0.1 * i for i in range(1536)] for _ in texts]


def embed_with_voyage(texts: List[str], model: str = "voyage-3") -> List[List[float]]:
    """使用 Voyage Embedding API 将文本转为向量。"""
    try:
        import voyageai
        client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", "pa-placeholder"))
        result = client.embed(texts, model=model, input_type="document")
        return result.embeddings
    except Exception as e:
        print(f"[Voyage Embed] 跳过（未配置 API Key 或未安装 voyageai）: {e}")
        return [[0.2 * i for i in range(1024)] for _ in texts]


# ─────────────────────────────────────────────
# 第 2 节：相似度计算（余弦 / 点积 / 欧氏距离）
# ─────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度：衡量向量方向的接近程度，范围 [-1, 1]。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(a: List[float], b: List[float]) -> float:
    """点积：向量内积，适合已归一化的向量（此时等价于余弦相似度）。"""
    return sum(x * y for x, y in zip(a, b))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """欧氏距离：向量在空间中的直线距离，越小越相似。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def demo_similarity():
    """演示三种相似度的差异。"""
    print("\n=== 相似度对比演示 ===")
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.5, 0.0]
    vec_c = [0.0, 0.0, 1.0]

    pairs = [("A vs B", vec_a, vec_b), ("A vs C", vec_a, vec_c), ("B vs C", vec_b, vec_c)]
    for label, x, y in pairs:
        cos  = cosine_similarity(x, y)
        dot  = dot_product(x, y)
        eucl = euclidean_distance(x, y)
        print(f"  {label} → 余弦={cos:.4f}  点积={dot:.4f}  欧氏={eucl:.4f}")


# ─────────────────────────────────────────────
# 第 3 节：ChromaDB 完整 CRUD
# ─────────────────────────────────────────────

def get_chroma_in_memory():
    """创建内存中的 ChromaDB 客户端（不落盘，适合快速测试）。"""
    import chromadb
    return chromadb.Client()


def get_chroma_persistent(persist_dir: str):
    """创建持久化 ChromaDB 客户端（数据保存到磁盘）。"""
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


def demo_chroma_crud(client):
    """演示 ChromaDB 的 Create / Read / Update / Delete 操作。"""
    print("\n=== ChromaDB CRUD 演示 ===")
    collection_name = "demo_crud"

    # 如果已存在则删除重建
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 使用余弦距离
    )

    # --- Create：插入文档 ---
    docs = [
        "Python 是一门简洁易学的编程语言",
        "机器学习需要大量数据和计算资源",
        "向量数据库专门用于存储和检索高维向量",
        "Transformer 架构彻底改变了 NLP 领域",
        "ChromaDB 是一个开源的向量数据库",
    ]
    ids   = [f"doc_{i}" for i in range(len(docs))]
    metas = [
        {"source": "textbook", "date": "2024-01-01", "type": "intro"},
        {"source": "paper",    "date": "2024-03-15", "type": "research"},
        {"source": "blog",     "date": "2024-05-20", "type": "tutorial"},
        {"source": "paper",    "date": "2023-11-10", "type": "research"},
        {"source": "docs",     "date": "2024-06-01", "type": "reference"},
    ]

    col.add(documents=docs, ids=ids, metadatas=metas)
    print(f"  已插入 {col.count()} 条文档")

    # --- Read：语义检索 ---
    results = col.query(
        query_texts=["什么是向量数据库"],
        n_results=2,
    )
    print("\n  语义检索「什么是向量数据库」的 Top-2 结果：")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"    [{dist:.4f}] {doc}")

    # --- Update：更新已有文档 ---
    col.update(
        ids=["doc_0"],
        documents=["Python 是一门简洁易学、生态丰富的编程语言，广泛用于 AI 开发"],
        metadatas=[{"source": "textbook", "date": "2024-07-01", "type": "intro"}],
    )
    updated = col.get(ids=["doc_0"])
    print(f"\n  更新后 doc_0：{updated['documents'][0]}")

    # --- Delete：删除文档 ---
    col.delete(ids=["doc_4"])
    print(f"  删除 doc_4 后，集合共 {col.count()} 条文档")

    return col


# ─────────────────────────────────────────────
# 第 4 节：元数据过滤
# ─────────────────────────────────────────────

def demo_metadata_filter(col):
    """演示按 source、date、type 等元数据过滤查询。"""
    print("\n=== 元数据过滤演示 ===")

    # 只查 source=paper 的文档
    res = col.query(
        query_texts=["深度学习模型"],
        n_results=5,
        where={"source": {"$eq": "paper"}},
    )
    print("  过滤 source=paper 的结果：")
    for doc in res["documents"][0]:
        print(f"    - {doc}")

    # 只查 type=tutorial 或 type=reference 的文档
    res2 = col.query(
        query_texts=["数据库使用方法"],
        n_results=5,
        where={"type": {"$in": ["tutorial", "reference"]}},
    )
    print("\n  过滤 type in [tutorial, reference] 的结果：")
    for doc in res2["documents"][0]:
        print(f"    - {doc}")


# ─────────────────────────────────────────────
# 第 5 节：批量向量化（大量文档）
# ─────────────────────────────────────────────

def batch_embed(
    texts: List[str],
    batch_size: int = 50,
    embed_fn=None,
    delay: float = 0.1,
) -> List[List[float]]:
    """
    将大量文本分批向量化，避免超出 API 单次请求限制。

    参数：
      texts      — 待向量化的文本列表
      batch_size — 每批大小（默认 50）
      embed_fn   — 向量化函数，接受 List[str] 返回 List[List[float]]
      delay      — 批次间等待秒数，避免触发限速
    """
    if embed_fn is None:
        # 本地 mock：生成随机向量，供演示用
        import random
        embed_fn = lambda ts: [[random.random() for _ in range(128)] for _ in ts]

    all_vectors: List[List[float]] = []
    total_batches = math.ceil(len(texts) / batch_size)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = start + batch_size
        chunk = texts[start:end]
        vectors = embed_fn(chunk)
        all_vectors.extend(vectors)
        print(f"  批次 {batch_idx + 1}/{total_batches}：向量化 {len(chunk)} 条，累计 {len(all_vectors)} 条")
        if batch_idx < total_batches - 1:
            time.sleep(delay)

    return all_vectors


def demo_batch_embed():
    """生成 200 条示例文档并分批向量化。"""
    print("\n=== 批量向量化演示（200 条文档）===")
    fake_docs = [f"这是第 {i} 号示例文档，内容主题为：{['AI', '数据库', '编程', '自然语言处理'][i % 4]}" for i in range(200)]
    vectors = batch_embed(fake_docs, batch_size=50)
    print(f"  完成，共 {len(vectors)} 个向量，每个维度 {len(vectors[0])}")
    return fake_docs, vectors


# ─────────────────────────────────────────────
# 第 6 节：增量更新（新增文档不重建全库）
# ─────────────────────────────────────────────

def incremental_upsert(col, new_docs: List[str], new_metas: Optional[List[Dict]] = None):
    """
    增量更新：只向量化并插入新文档，不影响已有数据。
    使用 upsert 语义：若 ID 已存在则覆盖，不存在则新增。
    """
    print("\n=== 增量更新演示 ===")
    if new_metas is None:
        new_metas = [{"source": "incremental", "date": "2024-07-03", "type": "update"} for _ in new_docs]

    # 用时间戳生成唯一 ID，确保不与旧数据冲突
    new_ids = [f"inc_{int(time.time() * 1000)}_{i}" for i in range(len(new_docs))]

    col.upsert(documents=new_docs, ids=new_ids, metadatas=new_metas)
    print(f"  增量插入 {len(new_docs)} 条，集合现有 {col.count()} 条")


# ─────────────────────────────────────────────
# 第 7 节：持久化（ChromaDB persist）
# ─────────────────────────────────────────────

def demo_persistence():
    """演示向量库的磁盘持久化与重新加载。"""
    print("\n=== 持久化演示 ===")
    import chromadb

    persist_dir = tempfile.mkdtemp(prefix="chroma_persist_")
    print(f"  持久化目录：{persist_dir}")

    # 写入阶段
    client_w = chromadb.PersistentClient(path=persist_dir)
    col_w = client_w.get_or_create_collection("persist_demo")
    col_w.add(
        documents=["持久化测试文档 A", "持久化测试文档 B"],
        ids=["p1", "p2"],
        metadatas=[{"type": "test"}, {"type": "test"}],
    )
    print(f"  写入 {col_w.count()} 条并关闭客户端")
    del client_w  # 关闭连接，数据已落盘

    # 重新加载阶段
    client_r = chromadb.PersistentClient(path=persist_dir)
    col_r = client_r.get_collection("persist_demo")
    print(f"  重新打开后读取到 {col_r.count()} 条（数据持久化成功）")

    result = col_r.get(ids=["p1"])
    print(f"  p1 内容：{result['documents'][0]}")


# ─────────────────────────────────────────────
# 第 8 节：多集合管理
# ─────────────────────────────────────────────

def demo_multi_collection(client):
    """演示在同一个 ChromaDB 实例中管理多个集合。"""
    print("\n=== 多集合管理演示 ===")

    # 创建三个用途不同的集合
    collections_config = [
        ("products",  {"domain": "ecommerce",  "hnsw:space": "cosine"}),
        ("articles",  {"domain": "knowledge",  "hnsw:space": "cosine"}),
        ("questions", {"domain": "qa",         "hnsw:space": "cosine"}),
    ]

    for col_name, meta in collections_config:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        col = client.create_collection(name=col_name, metadata=meta)
        # 向每个集合写入示例数据
        col.add(
            documents=[f"{col_name} 示例文档 {i}" for i in range(3)],
            ids=[f"{col_name}_{i}" for i in range(3)],
        )

    # 列出所有集合
    all_cols = client.list_collections()
    print(f"  当前共 {len(all_cols)} 个集合：")
    for c in all_cols:
        loaded = client.get_collection(c.name)
        print(f"    - {c.name}（{loaded.count()} 条）")

    # 跨集合独立查询
    for col_name, _ in collections_config:
        col = client.get_collection(col_name)
        res = col.query(query_texts=[f"查询 {col_name}"], n_results=1)
        print(f"  [{col_name}] 检索结果：{res['documents'][0][0]}")


# ─────────────────────────────────────────────
# 第 9 节：向量库迁移（内存 → 磁盘 → 云端示意）
# ─────────────────────────────────────────────

def migrate_collection(src_col, dst_col, batch_size: int = 100):
    """
    将 src_col 的全部数据迁移到 dst_col。
    支持跨客户端（内存→磁盘、磁盘→磁盘、磁盘→云端）。
    """
    total = src_col.count()
    if total == 0:
        print("  源集合为空，无需迁移")
        return

    # ChromaDB get() 的 limit 参数控制每批拉取量
    offset = 0
    migrated = 0
    while offset < total:
        batch = src_col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        if not batch["ids"]:
            break
        # 如果源集合有预计算向量则直接复用，否则让目标集合自动向量化
        if batch.get("embeddings") and batch["embeddings"][0] is not None:
            dst_col.upsert(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                embeddings=batch["embeddings"],
            )
        else:
            dst_col.upsert(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
            )
        migrated += len(batch["ids"])
        offset   += batch_size
        print(f"  迁移进度：{migrated}/{total}")

    print(f"  迁移完成，目标集合共 {dst_col.count()} 条")


def demo_migration():
    """演示向量库迁移：内存 → 磁盘。"""
    print("\n=== 向量库迁移演示（内存 → 磁盘）===")
    import chromadb

    # 内存源库
    mem_client = chromadb.Client()
    src = mem_client.create_collection("migration_src")
    src.add(
        documents=[f"迁移测试文档 {i}" for i in range(10)],
        ids=[f"m{i}" for i in range(10)],
        metadatas=[{"batch": "initial"} for _ in range(10)],
    )
    print(f"  源库（内存）：{src.count()} 条")

    # 磁盘目标库
    persist_dir = tempfile.mkdtemp(prefix="chroma_migrate_")
    disk_client = chromadb.PersistentClient(path=persist_dir)
    dst = disk_client.get_or_create_collection("migration_dst")

    migrate_collection(src, dst, batch_size=5)

    # 验证迁移后数据完整性
    sample = dst.get(ids=["m0"])
    print(f"  验证 m0：{sample['documents'][0]}")
    print(f"  磁盘持久化路径：{persist_dir}")

    # 云端迁移说明（需要 chromadb HttpClient，此处仅展示代码结构）
    print("\n  云端迁移示例（需要部署 Chroma 服务端）：")
    print("    cloud_client = chromadb.HttpClient(host='your-server', port=8000)")
    print("    cloud_col = cloud_client.get_or_create_collection('cloud_col')")
    print("    migrate_collection(dst, cloud_col)")


# ─────────────────────────────────────────────
# 主入口：按顺序运行所有演示
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Embedding 与向量存储完整演示")
    print("=" * 60)

    # 1. 相似度对比
    demo_similarity()

    # 2. ChromaDB CRUD（使用内存客户端，无需额外配置）
    mem_client = get_chroma_in_memory()
    col = demo_chroma_crud(mem_client)

    # 3. 元数据过滤
    demo_metadata_filter(col)

    # 4. 批量向量化
    demo_batch_embed()

    # 5. 增量更新
    incremental_upsert(
        col,
        new_docs=["增量文档：大型语言模型的应用场景", "增量文档：RAG 检索增强生成技术"],
        new_metas=[
            {"source": "blog", "date": "2024-07-03", "type": "tutorial"},
            {"source": "paper", "date": "2024-07-03", "type": "research"},
        ],
    )

    # 6. 持久化
    demo_persistence()

    # 7. 多集合管理
    demo_multi_collection(get_chroma_in_memory())

    # 8. 向量库迁移
    demo_migration()

    print("\n" + "=" * 60)
    print("  所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
