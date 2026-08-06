"""
通义千问 Embedding 模型调用示例
================================

什么是 Embedding（嵌入/向量化）？
  简单说，就是把一段文字变成一串数字（向量）。
  意思相近的文字，变出来的数字也会很接近。
  这样计算机就能"理解"文字之间的语义关系了。

用途举例：
  - 语义搜索：输入"如何做红烧肉"，能找到"猪肉炖菜的做法"
  - 文档分类：自动把文章归到合适的类别
  - RAG（检索增强生成）：先搜到相关资料，再让 AI 回答问题

千问 Embedding 模型：
  - text-embedding-v3（推荐）：最新款，效果最好，支持多种维度
  - text-embedding-v2：上一代，稳定可用
  - 它们都兼容 OpenAI 的接口格式，调用方式一样

运行方法：
  1. 确保 .env 文件里配好了 OPENAI_API_KEY（千问的 API Key）
  2. 安装依赖：pip install openai python-dotenv
  3. 运行：cd code && python examples/python/qwen_embedding_demo.py

文档参考：
  https://help.aliyun.com/zh/model-studio/text-embedding
"""

import os
import math
from openai import OpenAI
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 准备工作：加载配置、创建客户端
# ────────────────────────────────────────────────────────

# 从 .env 文件读取 API Key（就是你在阿里云百炼平台申请的那个密钥）
load_dotenv()

# 千问是国内服务，不需要走代理
# 如果你的电脑配了代理（VPN），这里临时关掉，避免连接失败
for key in ["http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

# 创建客户端，指向千问的接口地址
# 千问的接口和 OpenAI 格式一样，所以可以直接用 openai 库来调用
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 使用的 Embedding 模型
EMBED_MODEL = "text-embedding-v3"


# ────────────────────────────────────────────────────────
# 第 1 步：最基本的调用 —— 把一句话变成向量
# ────────────────────────────────────────────────────────

def basic_embedding():
    """
    最简单的例子：传入一句话，拿到一串数字（向量）。

    就像给每句话拍一张"语义指纹"，
    意思相近的话，指纹长得就很像。
    """
    print("=" * 50)
    print("第 1 步：把一句话变成向量")
    print("=" * 50)

    text = "人工智能正在改变世界"

    # 调用千问 Embedding API
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,            # 要转换的文字
        dimensions=256,        # 向量维度（数字的个数），越大越精细，但也越占空间
    )

    # 从返回结果中取出向量
    vector = response.data[0].embedding

    print(f"\n原文：「{text}」")
    print(f"向量维度：{len(vector)} 个数字")
    print(f"前 10 个数字：{vector[:10]}")
    print(f"\n（完整向量太长了，这里只展示前 10 个数字作为示意）")

    return vector


# ────────────────────────────────────────────────────────
# 第 2 步：批量转换 —— 一次把多句话都变成向量
# ────────────────────────────────────────────────────────

def batch_embedding():
    """
    实际使用中，通常一次要处理很多句话。
    千问 API 支持一次传入多条文本，效率更高。
    """
    print("\n" + "=" * 50)
    print("第 2 步：批量把多句话变成向量")
    print("=" * 50)

    texts = [
        "今天天气真好，适合出去走走",
        "外面阳光明媚，去公园散步吧",
        "Python 是最流行的编程语言之一",
        "机器学习需要大量训练数据",
    ]

    # 一次性把所有文本都转成向量
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=256,
    )

    vectors = [item.embedding for item in response.data]

    print(f"\n一共转换了 {len(vectors)} 句话")
    for i, text in enumerate(texts):
        print(f"  第 {i+1} 句：「{text}」→ {len(vectors[i])} 维向量")

    return texts, vectors


# ────────────────────────────────────────────────────────
# 第 3 步：计算相似度 —— 看看哪些话意思接近
# ────────────────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """
    余弦相似度：衡量两个向量"方向"有多接近。

    打个比方：
      两个人站在原点，各指向一个方向。
      如果指的方向完全一样 → 相似度 = 1（完全相同）
      如果指的方向完全相反 → 相似度 = -1（完全相反）
      如果互相垂直（毫无关系）→ 相似度 = 0

    在语义搜索里，相似度越接近 1，说明两句话意思越像。
    """
    # 计算点积（两个向量对应位置的数字相乘，再全部加起来）
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    # 计算各自的长度
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def similarity_demo(texts, vectors):
    """
    用实际例子展示：意思相近的话，相似度确实更高。
    """
    print("\n" + "=" * 50)
    print("第 3 步：计算语义相似度")
    print("=" * 50)

    print("\n我们来两两对比每句话的相似度：\n")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = cosine_similarity(vectors[i], vectors[j])
            # 相似度大于 0.7 的标记为"相似"
            tag = " ← 意思相近！" if score > 0.7 else ""
            print(f"  「{texts[i]}」")
            print(f"  「{texts[j]}」")
            print(f"   相似度：{score:.4f}{tag}")
            print()


# ────────────────────────────────────────────────────────
# 第 4 步：语义搜索 —— 在一堆文档里找最相关的
# ────────────────────────────────────────────────────────

def semantic_search():
    """
    语义搜索的核心逻辑：
      1. 把所有文档变成向量（建索引）
      2. 把用户的问题也变成向量
      3. 算一下问题和每个文档的相似度
      4. 返回最相似的几个文档

    这就是 RAG（检索增强生成）的第一步——"检索"。
    """
    print("\n" + "=" * 50)
    print("第 4 步：语义搜索实战")
    print("=" * 50)

    # 模拟一个小型知识库
    documents = [
        "红烧肉需要五花肉、酱油、冰糖，先煎后炖",
        "Python 的列表推导式可以简洁地创建列表",
        "定期锻炼有助于保持身体健康和心情愉悦",
        "机器学习模型需要经过训练才能做出预测",
        "北京故宫是中国最大的古代宫殿建筑群",
        "向量数据库专门用来存储和搜索高维向量",
        "清蒸鱼要用新鲜的鱼，蒸的时间不宜过长",
        "Transformer 架构是现代大语言模型的基础",
    ]

    # 第 1 步：把所有文档变成向量
    print("\n正在把知识库文档转成向量...")
    doc_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=documents,
        dimensions=256,
    )
    doc_vectors = [item.embedding for item in doc_response.data]
    print(f"  已向量化 {len(doc_vectors)} 篇文档")

    # 第 2 步：用户提问
    query = "怎么做一道好吃的菜"
    print(f"\n用户提问：「{query}」\n")

    # 第 3 步：把问题也变成向量
    query_response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=256,
    )
    query_vector = query_response.data[0].embedding

    # 第 4 步：算相似度，排序，取最相关的
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(query_vector, doc_vec)
        scores.append((score, documents[i]))

    # 按相似度从高到低排序
    scores.sort(reverse=True)

    print("搜索结果（按相关度排序）：")
    for rank, (score, doc) in enumerate(scores[:3], 1):
        print(f"  Top {rank}  [{score:.4f}]  {doc}")

    print("\n可以看到，AI 理解了「做菜」的意思，")
    print("找到了和烹饪相关的文档，而不是简单地匹配关键词。")


# ────────────────────────────────────────────────────────
# 第 5 步：不同维度的对比
# ────────────────────────────────────────────────────────

def dimension_comparison():
    """
    千问 text-embedding-v3 支持自定义向量维度。

    维度就像照片的分辨率：
      - 维度越高 → 信息越丰富 → 效果越好 → 但占空间更大
      - 维度越低 → 信息压缩更多 → 搜索更快 → 但精度会降低

    常用维度：
      - 1024：高精度场景（推荐）
      - 512 ：平衡精度和性能
      - 256 ：快速检索、资源有限时
    """
    print("\n" + "=" * 50)
    print("第 5 步：不同维度对比")
    print("=" * 50)

    text_pair = [
        "今天天气真好",
        "外面阳光灿烂",
    ]

    for dim in [256, 512, 1024]:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text_pair,
            dimensions=dim,
        )
        vec_a = response.data[0].embedding
        vec_b = response.data[1].embedding
        score = cosine_similarity(vec_a, vec_b)
        print(f"\n  维度 {dim:>4d}：相似度 = {score:.4f}  （向量长度 {len(vec_a)} 个数字）")

    print("\n通常维度越高，相似度计算越精准，但差异不会特别大。")
    print("实际项目中根据需求选择合适的维度即可。")


# ────────────────────────────────────────────────────────
# 主入口：按顺序运行所有演示
# ────────────────────────────────────────────────────────

def main():
    print("\n" + "🚀 通义千问 Embedding 模型使用演示 ".center(50, "─"))
    print()

    # 第 1 步：基本调用
    basic_embedding()

    # 第 2 步：批量转换
    texts, vectors = batch_embedding()

    # 第 3 步：相似度计算
    similarity_demo(texts, vectors)

    # 第 4 步：语义搜索
    semantic_search()

    # 第 5 步：维度对比
    dimension_comparison()

    print("\n" + "─" * 50)
    print("  所有演示完成！")
    print("─" * 50)
    print("\n下一步你可以：")
    print("  1. 修改 documents 列表，换成你自己的内容试试")
    print("  2. 改变 query，看搜索结果的变化")
    print("  3. 结合 ChromaDB 等向量数据库，构建完整的 RAG 系统")
    print("     （参考 examples/python/21_embeddings_vectorstore.py）")


if __name__ == "__main__":
    main()


"""
─────────────🚀 通义千问 Embedding 模型使用演示 ─────────────

==================================================
第 1 步：把一句话变成向量
==================================================

原文：「人工智能正在改变世界」
向量维度：256 个数字
前 10 个数字：[-0.036815233528614044, 0.05874400585889816, -0.06885425001382828, 0.001362921902909875, -0.20011311769485474, -0.0642523467540741, -0.10507681965827942, 0.09901066869497299, 0.042776793241500854, 0.05626874044537544]

（完整向量太长了，这里只展示前 10 个数字作为示意）

==================================================
第 2 步：批量把多句话变成向量
==================================================

一共转换了 4 句话
  第 1 句：「今天天气真好，适合出去走走」→ 256 维向量
  第 2 句：「外面阳光明媚，去公园散步吧」→ 256 维向量
  第 3 句：「Python 是最流行的编程语言之一」→ 256 维向量
  第 4 句：「机器学习需要大量训练数据」→ 256 维向量

==================================================
第 3 步：计算语义相似度
==================================================

我们来两两对比每句话的相似度：

  「今天天气真好，适合出去走走」
  「外面阳光明媚，去公园散步吧」
   相似度：0.8147 ← 意思相近！

  「今天天气真好，适合出去走走」
  「Python 是最流行的编程语言之一」
   相似度：0.4438

  「今天天气真好，适合出去走走」
  「机器学习需要大量训练数据」
   相似度：0.3734

  「外面阳光明媚，去公园散步吧」
  「Python 是最流行的编程语言之一」
   相似度：0.3857

  「外面阳光明媚，去公园散步吧」
  「机器学习需要大量训练数据」
   相似度：0.3101

  「Python 是最流行的编程语言之一」
  「机器学习需要大量训练数据」
   相似度：0.5620


==================================================
第 4 步：语义搜索实战
==================================================

正在把知识库文档转成向量...
  已向量化 8 篇文档

用户提问：「怎么做一道好吃的菜」

搜索结果（按相关度排序）：
  Top 1  [0.5731]  红烧肉需要五花肉、酱油、冰糖，先煎后炖
  Top 2  [0.4612]  清蒸鱼要用新鲜的鱼，蒸的时间不宜过长
  Top 3  [0.4071]  定期锻炼有助于保持身体健康和心情愉悦

可以看到，AI 理解了「做菜」的意思，
找到了和烹饪相关的文档，而不是简单地匹配关键词。

==================================================
第 5 步：不同维度对比
==================================================

  维度  256：相似度 = 0.7986  （向量长度 256 个数字）

  维度  512：相似度 = 0.7820  （向量长度 512 个数字）

  维度 1024：相似度 = 0.7494  （向量长度 1024 个数字）

通常维度越高，相似度计算越精准，但差异不会特别大。
实际项目中根据需求选择合适的维度即可。
"""