"""
demo_02_long_memory.py — 长期记忆：持久化、语义检索、跨会话调用

长期记忆 = 存在磁盘（JSON文件）+ 向量库（自实现 TF-IDF）
跨会话：程序重启后记忆依然在

架构：
  结构化事实 → JSON 文件（用户名、偏好等精确信息）
  语义内容   → TF-IDF 向量 + 余弦相似度（对话历史、知识，支持模糊检索）

注意：真实工程用 ChromaDB / Pinecone 等向量库替代这里的 TF-IDF
      TF-IDF 的作用完全等价：把文本变成向量 → 计算相似度

运行：
  cd code && python examples/python/demo_02_long_memory.py
"""

import json
import math
import os
import re
import time
from collections import Counter


# ─────────────────────────────────────────────────────────────
# TF-IDF 向量引擎（替代 ChromaDB Embedding，无需下载模型）
# ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """简单分词：按字/词切分，过滤停用词"""
    # 中文按字切，英文按词切
    tokens = re.findall(r'[一-鿿]|[a-zA-Z]+', text.lower())
    stopwords = {'的', '了', '在', '是', '我', '你', '他', '她', '它',
                 'a', 'the', 'is', 'in', 'of', 'to', 'and'}
    return [t for t in tokens if t not in stopwords and len(t) > 0]


def build_tfidf(docs: list[str]) -> tuple[list[dict], dict]:
    """
    建立 TF-IDF 向量索引

    TF-IDF = Term Frequency × Inverse Document Frequency
    TF：某词在当前文档中的频率（这篇文章多常提到这个词）
    IDF：log(总文档数 / 含该词的文档数)（越稀有的词权重越高）

    类比：
      TF  = 这个词在这本书里出现了多少次
      IDF = 这个词有多罕见（常见词如"的"权重低，专业词权重高）
    """
    n = len(docs)
    tokenized = [tokenize(doc) for doc in docs]

    # 计算 IDF
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    idf = {term: math.log((n + 1) / (df + 1)) for term, df in doc_freq.items()}

    # 计算每个文档的 TF-IDF 向量
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {t: (tf[t] / total) * idf.get(t, 1.0) for t in tf}
        vectors.append(vec)

    return vectors, idf


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """余弦相似度：衡量两个向量方向的相似程度（取值 0~1，越大越相似）"""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0


def query_vector(query: str, idf: dict) -> dict:
    """把查询词也变成 TF-IDF 向量"""
    tokens = tokenize(query)
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (tf[t] / total) * idf.get(t, 1.0) for t in tf}


# ─────────────────────────────────────────────────────────────
# 长期记忆管理器
# ─────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    长期记忆：跨会话持久化

    双存储策略：
      1. JSON 文件  → 结构化精确信息（姓名、偏好、关键事实）
      2. 向量索引   → 对话语义检索（"找和RAG相关的历史对话"）
    """

    def __init__(self, user_id: str, persist_dir: str = "/tmp/agent_memory"):
        self.user_id = user_id
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # ① JSON 文件：结构化用户档案
        self.profile_path = f"{persist_dir}/{user_id}_profile.json"
        self.profile = self._load_profile()

        # ② 向量库：对话片段 + TF-IDF 索引
        self.episodes_path = f"{persist_dir}/{user_id}_episodes.json"
        self.episodes: list[dict] = self._load_episodes()   # 原始文本
        self.vectors: list[dict] = []                       # TF-IDF 向量
        self.idf: dict = {}                                 # IDF 表
        self._rebuild_index()                               # 建立索引

        count = len(self.episodes)
        print(f"  📂 长期记忆初始化: user={user_id}")
        print(f"     JSON 档案: {self.profile_path}")
        print(f"     向量库:    {self.episodes_path} (已有 {count} 条记忆)")

    def _load_profile(self) -> dict:
        if os.path.exists(self.profile_path):
            with open(self.profile_path) as f:
                return json.load(f)
        return {"user_id": self.user_id, "facts": {}}

    def _save_profile(self):
        with open(self.profile_path, "w") as f:
            json.dump(self.profile, f, ensure_ascii=False, indent=2)

    def _load_episodes(self) -> list[dict]:
        if os.path.exists(self.episodes_path):
            with open(self.episodes_path) as f:
                return json.load(f)
        return []

    def _save_episodes(self):
        with open(self.episodes_path, "w") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)

    def _rebuild_index(self):
        """每次增加新记忆后重建 TF-IDF 索引（小规模下没问题）"""
        if not self.episodes:
            self.vectors, self.idf = [], {}
            return
        texts = [ep["content"] for ep in self.episodes]
        self.vectors, self.idf = build_tfidf(texts)

    # ─── 存储 ─────────────────────────────────────────────────

    def remember_fact(self, key: str, value: str):
        """存结构化事实到 JSON"""
        self.profile["facts"][key] = value
        self._save_profile()
        print(f"  💾 [JSON存储] {key} = {value}")

    def remember_episode(self, content: str, tags: list[str] = None):
        """存语义内容到向量库"""
        episode = {
            "id": f"{self.user_id}_{int(time.time() * 1000)}",
            "content": content,
            "tags": tags or [],
            "timestamp": time.time(),
        }
        self.episodes.append(episode)
        self._save_episodes()
        self._rebuild_index()  # 更新索引
        print(f"  💾 [向量存储] tags={tags} 内容={content[:40]!r}")

    # ─── 调用 / 检索 ──────────────────────────────────────────

    def recall_fact(self, key: str) -> str | None:
        return self.profile["facts"].get(key)

    def recall_semantic(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索：TF-IDF + 余弦相似度"""
        if not self.episodes:
            return []

        q_vec = query_vector(query, self.idf)
        scored = []
        for ep, vec in zip(self.episodes, self.vectors):
            sim = cosine_similarity(q_vec, vec)
            scored.append({**ep, "similarity": round(sim, 3)})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def build_context(self, current_query: str) -> str:
        """根据当前问题检索相关记忆，构建 system prompt 注入内容"""
        parts = []
        if self.profile["facts"]:
            facts_str = "、".join(f"{k}={v}" for k, v in self.profile["facts"].items())
            parts.append(f"用户信息：{facts_str}")

        memories = self.recall_semantic(current_query, top_k=3)
        relevant = [m for m in memories if m["similarity"] > 0.05]
        if relevant:
            mem_lines = "\n".join(
                f"  - [相似度{m['similarity']}] {m['content'][:60]}"
                for m in relevant
            )
            parts.append(f"相关历史记忆：\n{mem_lines}")

        return "\n".join(parts)

    def show_all(self):
        print(f"\n  📖 所有长期记忆 (user={self.user_id})")
        print(f"  ─ JSON档案 ─────────────────────────────")
        print(f"  {json.dumps(self.profile['facts'], ensure_ascii=False, indent=4)}")
        print(f"  ─ 向量库 ({len(self.episodes)}条) ──────────────────")
        for ep in self.episodes:
            print(f"  [{','.join(ep['tags']):12}] {ep['content'][:60]}")


# ─────────────────────────────────────────────────────────────
# Demo 运行
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Demo 02 — 长期记忆：持久化、检索、跨会话")
    print("=" * 60)

    # ── 会话一：存入记忆 ──────────────────────────────────────
    print("\n【会话一】第一次见面，存入信息")
    print("─" * 50)

    mem = LongTermMemory(user_id="user_xiaoming")

    mem.remember_fact("name", "小明")
    mem.remember_fact("occupation", "学生")
    mem.remember_fact("goal", "学习AI开发")

    mem.remember_episode(
        "用户问：什么是RAG？回答：RAG是检索增强生成，让模型在回答前先查文档。",
        tags=["RAG", "技术"]
    )
    mem.remember_episode(
        "用户问：Python适合做AI吗？回答：非常适合，大多数AI框架都用Python。",
        tags=["Python", "技术"]
    )
    mem.remember_episode(
        "用户表示对向量数据库很感兴趣，想深入了解ChromaDB的使用方法。",
        tags=["ChromaDB", "兴趣"]
    )

    mem.show_all()

    # ── 会话二：模拟重启后，从记忆恢复上下文 ─────────────────
    print("\n\n【会话二】程序重启，新会话开始（记忆仍在）")
    print("─" * 50)

    mem2 = LongTermMemory(user_id="user_xiaoming")  # 重新初始化，从磁盘加载

    current_question = "向量数据库怎么用？"
    print(f"\n  用户问: {current_question}")

    context = mem2.build_context(current_question)
    print(f"\n  🔍 检索到的相关记忆（将注入 system prompt）:")
    print(f"  {'─'*40}")
    for line in context.split("\n"):
        print(f"    {line}")

    name = mem2.recall_fact("name")
    goal = mem2.recall_fact("goal")
    print(f"\n  📌 精确查找 name={name}, goal={goal}")

    print(f"\n  🔎 语义检索「向量数据库 ChromaDB」Top-2:")
    results = mem2.recall_semantic("向量数据库 ChromaDB", top_k=2)
    for r in results:
        print(f"     相似度={r['similarity']} | {r['content'][:60]}")

    print("\n" + "=" * 60)
    print("  关键结论：")
    print("  1. JSON 文件存精确事实，随时精确查找（O(1)）")
    print("  2. TF-IDF/向量库存语义内容，支持模糊语义检索")
    print("  3. 程序重启后记忆仍在（写入磁盘持久化）")
    print("  4. 新对话开始 → build_context() → 自动注入历史")
    print("  5. 生产环境用 ChromaDB/Pinecone 替代 TF-IDF，原理相同")
    print("=" * 60)

    # 清理临时文件
    import shutil
    shutil.rmtree("/tmp/agent_memory", ignore_errors=True)
    print("  🧹 已清理临时文件")


if __name__ == "__main__":
    main()


"""
============================================================
  Demo 02 — 长期记忆：持久化、检索、跨会话
============================================================

【会话一】第一次见面，存入信息
──────────────────────────────────────────────────
  📂 长期记忆初始化: user=user_xiaoming
     JSON 档案: /tmp/agent_memory/user_xiaoming_profile.json
     向量库:    /tmp/agent_memory/user_xiaoming_episodes.json (已有 0 条记忆)
  💾 [JSON存储] name = 小明
  💾 [JSON存储] occupation = 学生
  💾 [JSON存储] goal = 学习AI开发
  💾 [向量存储] tags=['RAG', '技术'] 内容='用户问：什么是RAG？回答：RAG是检索增强生成，让模型在回答前先查文档。'
  💾 [向量存储] tags=['Python', '技术'] 内容='用户问：Python适合做AI吗？回答：非常适合，大多数AI框架都用Python'
  💾 [向量存储] tags=['ChromaDB', '兴趣'] 内容='用户表示对向量数据库很感兴趣，想深入了解ChromaDB的使用方法。'

  📖 所有长期记忆 (user=user_xiaoming)
  ─ JSON档案 ─────────────────────────────
  {
    "name": "小明",
    "occupation": "学生",
    "goal": "学习AI开发"
}
  ─ 向量库 (3条) ──────────────────
  [RAG,技术      ] 用户问：什么是RAG？回答：RAG是检索增强生成，让模型在回答前先查文档。
  [Python,技术   ] 用户问：Python适合做AI吗？回答：非常适合，大多数AI框架都用Python。
  [ChromaDB,兴趣 ] 用户表示对向量数据库很感兴趣，想深入了解ChromaDB的使用方法。


【会话二】程序重启，新会话开始（记忆仍在）
──────────────────────────────────────────────────
  📂 长期记忆初始化: user=user_xiaoming
     JSON 档案: /tmp/agent_memory/user_xiaoming_profile.json
     向量库:    /tmp/agent_memory/user_xiaoming_episodes.json (已有 3 条记忆)

  用户问: 向量数据库怎么用？

  🔍 检索到的相关记忆（将注入 system prompt）:
  ────────────────────────────────────────
    用户信息：name=小明、occupation=学生、goal=学习AI开发
    相关历史记忆：
      - [相似度0.354] 用户表示对向量数据库很感兴趣，想深入了解ChromaDB的使用方法。
      - [相似度0.08] 用户问：什么是RAG？回答：RAG是检索增强生成，让模型在回答前先查文档。

  📌 精确查找 name=小明, goal=学习AI开发

  🔎 语义检索「向量数据库 ChromaDB」Top-2:
     相似度=0.519 | 用户表示对向量数据库很感兴趣，想深入了解ChromaDB的使用方法。
     相似度=0.015 | 用户问：Python适合做AI吗？回答：非常适合，大多数AI框架都用Python。

============================================================
  关键结论：
  1. JSON 文件存精确事实，随时精确查找（O(1)）
  2. TF-IDF/向量库存语义内容，支持模糊语义检索
  3. 程序重启后记忆仍在（写入磁盘持久化）
  4. 新对话开始 → build_context() → 自动注入历史
  5. 生产环境用 ChromaDB/Pinecone 替代 TF-IDF，原理相同
============================================================
"""