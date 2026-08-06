# RAG与向量数据库 面试题库（100题）

> 涵盖检索增强生成全链路：文档处理、向量检索、混合检索、评测等核心主题。

---

## 目录

1. [RAG基本原理与适用场景 (Q1-Q7)](#1-rag基本原理与适用场景-q1-q7)
2. [文档预处理与清洗 (Q8-Q14)](#2-文档预处理与清洗-q8-q14)
3. [Chunking策略对比（固定/语义/递归/父子/滑动窗口） (Q15-Q22)](#3-chunking策略对比-q15-q22)
4. [Embedding模型选型 (Q23-Q29)](#4-embedding模型选型-q23-q29)
5. [向量数据库对比（Chroma/Pinecone/Weaviate/Milvus/pgvector） (Q30-Q37)](#5-向量数据库对比-q30-q37)
6. [相似度算法与HNSW索引 (Q38-Q44)](#6-相似度算法与hnsw索引-q38-q44)
7. [混合检索（向量+BM25+RRF） (Q45-Q51)](#7-混合检索-q45-q51)
8. [Re-ranking技术 (Q52-Q58)](#8-re-ranking技术-q52-q58)
9. [Advanced RAG（HyDE/Self-Query/Contextual Compression） (Q59-Q66)](#9-advanced-rag-q59-q66)
10. [Agentic RAG (Q67-Q73)](#10-agentic-rag-q67-q73)
11. [RAGAS评测框架 (Q74-Q80)](#11-ragas评测框架-q74-q80)
12. [生产级RAG设计 (Q81-Q87)](#12-生产级rag设计-q81-q87)
13. [RAG中的幻觉处理 (Q88-Q94)](#13-rag中的幻觉处理-q88-q94)
14. [RAG vs Fine-tuning决策 (Q95-Q100)](#14-rag-vs-fine-tuning决策-q95-q100)

---
## 1. RAG基本原理与适用场景 (Q1-Q7)

**Q1. 请描述RAG（检索增强生成）的完整工作流程，并解释它解决了LLM的哪些核心痛点？**
[难度：⭐] [类型：概念]
**答：**
RAG（Retrieval-Augmented Generation）工作流程分为两个阶段：

**索引阶段（Offline）：**
1. 文档加载 → 2. 文档分块（Chunking）→ 3. Embedding 向量化 → 4. 存入向量数据库

**推理阶段（Online）：**
1. 用户提问 → 2. 对 Query 进行 Embedding → 3. 向量相似度检索 Top-K 文档块 → 4. 将 Query + 检索结果拼接成 Prompt → 5. LLM 生成最终回答

**RAG 解决的核心痛点：**
- **知识截止问题**：LLM 训练数据有截止日期，RAG 可以实时注入最新知识
- **私域知识缺失**：企业内部文档、专有数据 LLM 无从得知，RAG 可按需注入
- **幻觉问题缓解**：有了具体上下文依据，LLM 生成更有据可查
- **成本可控**：无需重新训练或微调，更新知识库只需重新索引文档
- **可解释性增强**：可以溯源到具体文档片段，提升可信度

**考察点：** 考察候选人对 RAG 整体架构的理解深度，以及能否将其与 LLM 固有局限性联系起来分析。

---

**Q2. RAG的"幻觉"问题与纯LLM的"幻觉"有何本质区别？RAG是否能完全消除幻觉？**
[难度：⭐⭐] [类型：概念]
**答：**
**纯 LLM 幻觉：** 完全来自模型参数中的统计记忆，没有外部依据，模型"凭感觉"生成听起来合理但不正确的内容。

**RAG 中的幻觉类型：**
1. **检索幻觉**：检索到的文档本身不相关或错误，LLM 基于错误依据生成答案
2. **生成幻觉**：即便有正确的检索结果，LLM 仍可能忽视或曲解上下文内容
3. **不完整检索引发的幻觉**：问题需要多跳推理，单次检索覆盖不完整，LLM 用参数知识补全

**RAG 无法完全消除幻觉的原因：**
- 检索召回率有限，可能漏掉关键信息
- LLM 仍有将上下文与参数知识混合的倾向
- 用户问题本身可能含糊，导致检索偏离
- 文档本身可能包含错误信息

**缓解策略：** 提高检索精度 + 使用忠实度（Faithfulness）评估 + 在 Prompt 中明确要求"仅根据以下上下文回答"

**考察点：** 理解 RAG 幻觉的多种来源，不能简单认为"有了 RAG 就没有幻觉"。

---

**Q3. 什么场景适合使用RAG？什么场景下RAG不是最佳方案？**
[难度：⭐⭐] [类型：场景]
**答：**
**适合使用 RAG 的场景：**
- 知识库频繁更新（新闻、政策、产品文档）
- 私有/机密文档问答（企业内网知识库）
- 需要精确溯源的场景（法律、医疗合规问答）
- 文档规模庞大，超出上下文窗口限制
- 多语言文档检索

**RAG 不是最佳方案的场景：**
- **需要深度推理**：如复杂数学证明，RAG 提供的碎片化上下文反而干扰推理
- **格式转换任务**：如代码翻译、摘要生成，不需要外部知识
- **知识已在训练集中**：通用常识问题，直接用 LLM 更高效
- **知识图谱关系推理**：多跳关系查询用图数据库更合适
- **实时计算/聚合需求**：如"统计某字段均值"，应接 SQL/API，而非向量检索

**考察点：** 工程判断力——能识别 RAG 的边界，不盲目套用。

---

**Q4. 解释RAG中的"Top-K检索"参数，以及K值选择对系统的影响？**
[难度：⭐⭐] [类型：设计]
**答：**
Top-K 检索返回与 Query 向量相似度最高的 K 个文档块。

**K 值影响分析：**

| K 值 | 优点 | 缺点 |
|------|------|------|
| 小（K=3~5） | Prompt 短，节省 Token；噪音少 | 召回率低，可能遗漏关键信息 |
| 大（K=10~20） | 召回率高，覆盖更全面 | Prompt 长，成本增加；噪音多，LLM 易被干扰 |

**最佳实践：**
- 先检索较大的 K（如 20），再通过 Re-ranker 精选最终送入 LLM 的 Top-K（如 5）
- 动态 K：根据 Query 类型调整（简单事实查询 K=3，复杂分析 K=10）
- 使用 LLM 的上下文窗口大小反推合理的 K 值
- 结合 similarity threshold 过滤：相似度低于阈值的结果即使在 Top-K 内也丢弃

**考察点：** 对检索质量与成本权衡的理解，以及与 Re-ranking 结合的系统思维。

---

**Q5. 什么是"Lost in the Middle"问题？在RAG中如何应对？**
[难度：⭐⭐] [类型：概念]
**答：**
**Lost in the Middle 问题：** 研究发现（Liu et al. 2023），当 LLM 的上下文很长时，模型对头部和尾部的信息注意力强，而中间位置的信息容易被忽略，导致即便相关内容存在于上下文中，LLM 也无法有效利用。

**在 RAG 中的影响：**
- 检索到的多个文档块中，排在中间位置的可能是最相关的，但 LLM 却忽视了
- K 越大，问题越严重

**应对方法：**
1. **减小 K 值**：只传入最相关的少数 chunk
2. **结果重排**：将最相关的 chunk 放在首尾（前置最相关，末尾次相关）
3. **Re-ranking + Contextual Compression**：压缩每个 chunk 的内容，只保留与 Query 最相关的句子
4. **分步检索**：将问题拆解为子问题，多轮检索，每次只传少量上下文
5. **使用长上下文模型**：如 Gemini 1.5 Pro（1M context），但不能完全解决注意力偏差

**考察点：** 对 LLM 注意力机制局限性的认知，以及 RAG 系统优化思路。

---

**Q6. 如何评估一个RAG系统的检索质量？常用的指标有哪些？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**检索层指标：**

- **Recall@K（召回率）**：Ground Truth 文档被检索出的比例
  - `Recall@K = |相关文档 ∩ Top-K| / |相关文档|`
- **Precision@K（精确率）**：Top-K 中相关文档的比例
  - `Precision@K = |相关文档 ∩ Top-K| / K`
- **MRR（Mean Reciprocal Rank）**：第一个相关结果的排名倒数均值
- **NDCG（Normalized Discounted Cumulative Gain）**：考虑排名位置的相关性得分

**端到端指标（借助 RAGAS 框架）：**
- **Context Precision**：检索到的上下文中有多少是真正相关的
- **Context Recall**：Ground Truth 答案所需的信息有多少被检索到

**代码示例（计算 Recall@K）：**
```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)

# 示例
retrieved = ["doc3", "doc1", "doc5", "doc2"]
relevant = ["doc1", "doc3"]
print(recall_at_k(retrieved, relevant, k=3))  # 1.0
```

**考察点：** 对 RAG 评测体系的掌握，能分层（检索层、生成层）思考评估指标。

---

**Q7. 对比 Naive RAG、Advanced RAG 和 Modular RAG 三种架构模式？**
[难度：⭐⭐⭐] [类型：设计]
**答：**

| 架构 | 核心特点 | 优势 | 劣势 |
|------|----------|------|------|
| **Naive RAG** | 固定 Chunk → 向量检索 → LLM | 简单易实现 | 检索质量差、无优化 |
| **Advanced RAG** | 查询改写 + 混合检索 + Re-ranking + 后处理 | 检索质量高 | 链路长、延迟高 |
| **Modular RAG** | 模块可插拔，按需组合 | 灵活可扩展 | 设计复杂 |

**Naive RAG 典型问题：**
- 固定 chunk 切割破坏语义完整性
- 向量检索难以处理关键词精确匹配需求
- 无查询优化，一次检索结果差就全盘崩溃

**Advanced RAG 核心优化点：**
- Pre-retrieval：Query 扩展/改写、HyDE
- Retrieval：混合检索（Dense + Sparse）
- Post-retrieval：Re-ranking、Contextual Compression

**Modular RAG 代表框架：** LlamaIndex、LangChain，可以灵活组合检索器、重排器、生成器等模块。

**考察点：** RAG 演进路线的理解，以及能否针对具体业务选择合适的架构。

---

## 2. 文档预处理与清洗 (Q8-Q14)

**Q8. 在将文档送入RAG管道前，通常需要哪些预处理步骤？为什么这些步骤很重要？**
[难度：⭐] [类型：概念]
**答：**
文档预处理是 RAG 质量的基础，"垃圾进，垃圾出"。

**主要预处理步骤：**

1. **文档解析（Parsing）**
   - PDF → 提取文字（注意扫描件 PDF 需 OCR）
   - Word/HTML/Markdown → 结构化文本
   - 表格 → 转为自然语言描述或结构化格式

2. **噪声清洗**
   - 去除页眉页脚、水印、页码
   - 去除重复内容、广告链接
   - 修复乱码、特殊字符

3. **元数据提取**
   - 文件名、章节标题、作者、日期
   - 用于后续元数据过滤检索

4. **文本规范化**
   - 统一编码（UTF-8）
   - 繁简转换、全半角统一
   - 去除多余空白符

5. **语言检测**
   - 多语言文档需分别处理

**代码示例（使用 LangChain 加载并清洗 PDF）：**
```python
from langchain.document_loaders import PyPDFLoader
import re

loader = PyPDFLoader("document.pdf")
docs = loader.load()

def clean_text(text):
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除页码模式（如 "- 5 -"）
    text = re.sub(r'-\s*\d+\s*-', '', text)
    return text.strip()

for doc in docs:
    doc.page_content = clean_text(doc.page_content)
```

**考察点：** 工程实践意识——预处理的重要性以及常见的文档处理方法。

---

**Q9. 如何处理包含大量表格、图片的PDF文档？有哪些工具和策略？**
[难度：⭐⭐] [类型：场景]
**答：**
**表格处理策略：**

1. **表格转 Markdown**：Camelot、pdfplumber 等库可提取表格结构
2. **表格转自然语言描述**：用 LLM 将表格转换为"第1行第2列是XXX"的描述
3. **独立索引**：将表格内容单独作为一个 chunk，并附加表格位置元数据
4. **多模态处理**：GPT-4V 等模型直接理解表格图像

**图片处理策略：**

1. **OCR 提取文字**：Tesseract、PaddleOCR 提取图片中的文字
2. **图说生成（Caption Generation）**：使用视觉模型（BLIP-2、LLaVA）生成图片描述
3. **多模态 Embedding**：CLIP 等模型将图片和文字放入同一向量空间
4. **跳过策略**：对于装饰性图片，可标记并跳过

**工具推荐：**
- `unstructured`：处理多种格式文档，包括 PDF 表格和图片
- `pdfplumber`：精确提取 PDF 表格
- `llmsherpa`：保留 PDF 层级结构
- `marker`：将 PDF 高质量转为 Markdown

**代码示例（使用 unstructured 处理复杂 PDF）：**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="report.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
)

# 分类处理不同元素
for element in elements:
    if element.category == "Table":
        print("表格内容:", element.text)
    elif element.category == "Image":
        print("图片路径:", element.metadata.image_path)
```

**考察点：** 对真实文档复杂性的认知，能处理非结构化内容。

---

**Q10. 什么是文档的元数据（metadata）？如何利用元数据改进检索？**
[难度：⭐⭐] [类型：设计]
**答：**
**元数据定义：** 描述文档/chunk 的附加信息，不直接参与向量检索，但可用于过滤。

**常见元数据字段：**
```python
metadata = {
    "source": "annual_report_2024.pdf",
    "page": 12,
    "chapter": "财务分析",
    "author": "张三",
    "date": "2024-01-15",
    "doc_type": "financial_report",
    "department": "finance",
    "language": "zh",
    "chunk_index": 3
}
```

**元数据改进检索的方式：**

1. **元数据过滤（Pre-filtering）**：在向量检索前先过滤，缩小候选集
   ```python
   results = vectorstore.similarity_search(
       query="净利润增长",
       k=5,
       filter={"doc_type": "financial_report", "date": {"$gte": "2024-01-01"}}
   )
   ```

2. **Self-Query Retriever**：让 LLM 自动从用户问题中提取过滤条件
   ```python
   from langchain.retrievers.self_query.base import SelfQueryRetriever
   # LLM 自动将"2024年的财务报告"解析为 filter={"date": "2024"}
   ```

3. **混合排序**：向量相似度 × 元数据权重（如时间衰减）

4. **溯源展示**：回答时引用具体的文档来源和页码

**考察点：** 元数据设计能力和过滤检索的工程实践。

---

**Q11. 如何处理文档中的重复内容和近重复内容？**
[难度：⭐⭐] [类型：设计]
**答：**
**重复内容的危害：**
- 增加索引存储成本
- 检索时返回多个相同内容，浪费 Top-K 名额
- 降低 Precision 指标

**精确重复检测：**
```python
import hashlib

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# 去重
seen_hashes = set()
unique_chunks = []
for chunk in chunks:
    h = get_hash(chunk.page_content)
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_chunks.append(chunk)
```

**近重复内容（Near-duplicate）检测：**

1. **MinHash + LSH**：高效的近似相似度检测
   ```python
   from datasketch import MinHash, MinHashLSH

   lsh = MinHashLSH(threshold=0.85, num_perm=128)
   ```

2. **Cosine 相似度阈值**：两个 chunk 的 embedding 余弦相似度 > 0.95 则视为近重复

3. **SimHash**：文档指纹技术，O(1) 查询复杂度

**策略：**
- 精确重复：直接删除
- 近重复（相似度 > 0.95）：保留更完整的版本
- 版本差异（相似度 0.7~0.95）：保留所有版本，添加版本元数据

**考察点：** 数据质量意识和去重算法的工程实践。

---

**Q12. 文档分块（Chunking）之前，如何处理文档的层级结构（章节/标题/段落）？**
[难度：⭐⭐] [类型：设计]
**答：**
保留文档层级结构对检索质量有重要影响。

**层级结构处理策略：**

1. **标题感知分块（Title-aware Chunking）**
   - 识别 H1/H2/H3 标题
   - 每个 chunk 包含其所属的标题路径（面包屑）
   - 如：`"2. 财务分析 > 2.1 收入分析 > 内容..."`

2. **层级化元数据**
   ```python
   chunk.metadata = {
       "section": "第二章",
       "subsection": "2.1 收入分析",
       "hierarchy_path": "报告 > 第二章 > 2.1"
   }
   ```

3. **Markdown Header 解析**
   ```python
   from langchain.text_splitter import MarkdownHeaderTextSplitter

   headers_to_split_on = [
       ("#", "H1"),
       ("##", "H2"),
       ("###", "H3"),
   ]
   splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
   docs = splitter.split_text(markdown_content)
   ```

4. **上下文注入（Context Injection）**
   - 在每个 chunk 前自动注入其所属章节标题
   - 即使 chunk 被单独检索，也能携带上下文信息

**Anthropic 提出的 Contextual Retrieval：**
- 用 LLM 为每个 chunk 生成"这段话在整篇文档中的位置和含义"描述
- 将该描述前置到 chunk 内容中，大幅提升检索召回率

**考察点：** 对文档结构的理解，以及如何在分块时保留语义上下文。

---

**Q13. 在处理多语言文档时，RAG管道需要注意哪些问题？**
[难度：⭐⭐] [类型：场景]
**答：**
**核心挑战：**
1. 同一概念在不同语言中的 Embedding 向量不在同一空间
2. Tokenizer 对不同语言的处理效率不同
3. 中文/日文无空格分词，影响 BM25 检索

**解决方案：**

1. **使用多语言 Embedding 模型**
   - `multilingual-e5-large`（微软，支持100+语言）
   - `paraphrase-multilingual-mpnet-base-v2`
   - `text-embedding-3-large`（OpenAI，跨语言能力强）

2. **Query 翻译策略**
   - 将用户 Query 翻译为文档的语言后再检索
   - 使用 LLM 生成多语言 Query 变体

3. **分语言索引**
   - 中文文档和英文文档分开建索引
   - 根据 Query 语言路由到对应索引

4. **中文分词优化**
   - BM25 检索中文时需先分词（jieba/pkuseg）
   - 避免按字符切割的 BM25 效果差

**代码示例：**
```python
import jieba
from rank_bm25 import BM25Okapi

# 中文文档需先分词
def tokenize_chinese(text):
    return list(jieba.cut(text))

tokenized_corpus = [tokenize_chinese(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query_tokens = tokenize_chinese("检索增强生成技术")
scores = bm25.get_scores(query_tokens)
```

**考察点：** 跨语言检索的工程挑战和解决方案。

---

**Q14. 如何处理代码文档（源代码文件）的索引与检索？有哪些特殊考量？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
代码文档与普通文本有本质区别：语法结构严格，语义依赖上下文。

**特殊考量：**

1. **代码语义切割**
   - 不能在函数中间截断，应按函数/类/方法边界切割
   - 使用 AST（抽象语法树）解析代码结构

2. **代码专用 Embedding 模型**
   - `code-embedding-ada-002`（OpenAI）
   - `text-embedding-3-large`（对代码有较好理解）
   - `CodeBERT`（专门为代码设计）

3. **保留代码上下文**
   - 函数 chunk 需包含：函数签名 + docstring + 函数体
   - 添加函数所属类名/模块名作为元数据

4. **混合检索**：关键词精确匹配（函数名、变量名）+ 语义检索

5. **代码注释处理**：注释含有重要语义信息，需与代码一起保留

**代码示例（基于 AST 切割 Python 代码）：**
```python
import ast

def extract_functions(source_code, filename):
    tree = ast.parse(source_code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            func_code = source_code.split('\n')[start:end]
            functions.append({
                "content": '\n'.join(func_code),
                "metadata": {
                    "filename": filename,
                    "function_name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno
                }
            })
    return functions
```

**考察点：** 对代码检索特殊性的理解，以及结构化内容的处理能力。

---

## 3. Chunking策略对比（固定/语义/递归/父子/滑动窗口） (Q15-Q22)

**Q15. 对比"固定大小分块"和"语义分块"的优缺点，各自适用于什么场景？**
[难度：⭐⭐] [类型：对比]
**答：**

**固定大小分块（Fixed-size Chunking）：**
- **原理**：按固定字符数或 token 数截断，通常带有 overlap
- **优点**：简单、可预测、处理速度快
- **缺点**：可能在句子中间截断，破坏语义完整性
- **适用场景**：文本结构松散，或对速度要求高的原型系统

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
chunks = splitter.split_text(text)
```

**语义分块（Semantic Chunking）：**
- **原理**：计算相邻句子间的 embedding 相似度，在相似度骤降处切割（语义边界）
- **优点**：切割点更符合语义，chunk 内容语义连贯
- **缺点**：计算成本高（需要对每个句子做 embedding），chunk 大小不均匀
- **适用场景**：文档质量高、语义清晰的场景，如学术论文、法律文书

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

semantic_chunker = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
```

**考察点：** 对不同分块策略的权衡分析，能根据业务场景选择。

---

**Q16. 什么是"递归字符文本分割"（RecursiveCharacterTextSplitter）？为什么它通常是默认推荐的分块方法？**
[难度：⭐⭐] [类型：概念]
**答：**
RecursiveCharacterTextSplitter 按优先级尝试不同的分隔符来切割文本，只有当上一级分隔符无法将 chunk 控制在目标大小时，才使用下一级分隔符。

**默认分隔符优先级（中文文本）：**
1. `"\n\n"` （段落）
2. `"\n"` （换行）
3. `"。"` / `"！"` / `"？"` （句子）
4. `"，"` （逗号）
5. `" "` （空格）
6. `""` （字符级）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    length_function=len
)
chunks = splitter.split_text(text)
```

**为什么推荐它：**
- 尽量保持段落完整 → 语义连贯
- 当段落过长时自动降级到句子级切割 → 不会超出 chunk_size
- 实现简单但效果好，适用于大多数文本类型

**考察点：** 理解分割器的工作机制，而不仅仅是会调用 API。

---

**Q17. 解释"父子分块"（Parent-Child Chunking）策略，以及它如何同时兼顾检索精度和上下文完整性？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**核心思想：**
- **小 chunk（子）** 用于精确向量检索（更精准的语义匹配）
- **大 chunk（父）** 用于送入 LLM 生成答案（更完整的上下文）

**工作流程：**
1. 将文档切割为大的父 chunk（如 1500 tokens）
2. 将每个父 chunk 进一步切割为小的子 chunk（如 300 tokens）
3. 只对子 chunk 做 Embedding 并索引
4. 检索时：找到最相关的子 chunk → 查找其对应的父 chunk → 将父 chunk 送入 LLM

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)

vectorstore = Chroma(embedding_function=embeddings)
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
results = retriever.get_relevant_documents("RAG 的核心组件")
# 返回的是父 chunk，包含完整上下文
```

**适用场景：** 需要精确检索 + 完整上下文的场景，如长文档 QA、技术文档检索。

**考察点：** 检索与生成的解耦思想，以及 Parent-Child 架构的工程实现。

---

**Q18. 什么是"滑动窗口分块"？overlap 大小如何影响检索效果？**
[难度：⭐⭐] [类型：概念]
**答：**
**滑动窗口分块原理：**
相邻 chunk 之间有重叠（overlap）部分，确保跨 chunk 的信息不会因切割而丢失。

| Overlap 比例 | 效果 | 问题 |
|-------------|------|------|
| 0%（无重叠） | 存储最小 | 跨 chunk 信息丢失 |
| 10~20% | 均衡推荐 | — |
| 50%+ | 上下文连贯 | 存储翻倍，重复内容多 |

**最佳实践：**
- 一般建议 overlap = chunk_size 的 10~20%
- 过大的 overlap 会导致相邻 chunk 极度相似，检索时返回多个相似结果，浪费 K 名额

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # 20% overlap
)
```

**考察点：** overlap 参数的工程意义，不只是知道参数存在，要理解其影响机制。

---

**Q19. 什么是"命题分块"（Proposition Chunking）？它与普通分块有何不同？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**命题分块（Chen et al. 2023）：**
使用 LLM 将文档中的每个段落分解为最小的、自包含的、可以独立理解的原子事实陈述（命题），每个命题作为一个独立的检索单元。

**示例：**
- 原始文本：`"GPT-4 于2023年3月发布，是 OpenAI 的多模态大语言模型，支持图文输入。"`
- 命题 1：`"GPT-4 于2023年3月发布。"`
- 命题 2：`"GPT-4 是 OpenAI 开发的多模态大语言模型。"`
- 命题 3：`"GPT-4 支持图文输入。"`

**优点：** 极高的检索精度，一个命题对应一个具体事实，减少噪音。
**缺点：** 需要 LLM 处理所有文档，成本高；命题过于原子化，LLM 生成时上下文可能不足。

**适用场景：** 知识密集型 QA、需要高精度事实检索的场景。

**考察点：** 对 Proposition Retrieval 研究成果的了解，体现研究追踪能力。

---

**Q20. 如何为不同类型的文档（技术手册/法律合同/新闻文章）选择合适的 chunk_size？**
[难度：⭐⭐⭐] [类型：设计]
**答：**

| 文档类型 | 推荐 chunk_size | 原因 |
|---------|----------------|------|
| 技术手册 | 500~1000 tokens | 每段包含独立的操作步骤 |
| 法律合同 | 800~1500 tokens | 条款完整性重要，截断易失去法律含义 |
| 新闻文章 | 300~500 tokens | 段落短，信息密度高 |
| 学术论文 | 500~800 tokens | 按段落或小节切割 |
| 代码文件 | 函数/类边界 | 不应人为截断代码逻辑 |

**影响因素：**
1. **Embedding 模型最大长度**：超过最大长度会被截断，如 `all-MiniLM-L6-v2` 最大 512 tokens
2. **问题颗粒度**：事实型问题（小 chunk）vs 分析型问题（大 chunk）
3. **LLM 上下文窗口**：K × chunk_size 不能超过上下文限制

**调参建议：** 用 RAGAS 或手工标注的测试集，对不同 chunk_size 做 A/B 测试，用 Recall@K 和答案质量来选择最优值。

**考察点：** 系统性的参数选择思维，不是随便给一个数字。

---

**Q21. 在 RAG 中，chunk 的 overlap 区域会不会被重复索引？这会造成什么问题？如何解决？**
[难度：⭐⭐] [类型：场景]
**答：**
**是的，overlap 区域会被重复索引。** 两个相邻 chunk 都包含相同的 overlap 文本，会产生两个相近的 Embedding 向量。

**可能造成的问题：**
1. 检索结果冗余：Top-K 返回多个内容高度相似的 chunk，浪费名额
2. 存储浪费：overlap 越大，冗余存储越多

**解决方案：**

1. **MMR（Maximal Marginal Relevance）检索：** 在相关性和多样性之间取得平衡
```python
results = vectorstore.max_marginal_relevance_search(
    query, k=5, fetch_k=20, lambda_mult=0.7
)
```

2. **过滤高相似度结果：** 同一文档中相邻 chunk 若相似度 > 0.95，去掉重复的

3. **减少 overlap 比例：** 不要设置超过 20% 的 overlap

**考察点：** 对 overlap 机制的深层理解，以及工程中的去重意识。

---

**Q22. 实现一个自定义的 Chunking 函数，要求按中文段落分割，并保留章节标题信息。**
[难度：⭐⭐⭐] [类型：代码]
**答：**
```python
import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ChunkWithContext:
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    chunk_index: int = 0
    source: str = ""

def chinese_section_chunker(
    text: str,
    source: str = "",
    max_chunk_size: int = 800,
    overlap: int = 100
) -> List[ChunkWithContext]:
    chunks = []
    heading_pattern = re.compile(
        r'^(第[一二三四五六七八九十百]+[章节].*|'
        r'[一二三四五六七八九十]+、.*|'
        r'\d+\.\s+.{2,30}|'
        r'#{1,3}\s+.+)$',
        re.MULTILINE
    )
    current_chapter = None
    current_section = None
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if heading_pattern.match(para):
            if "章" in para or para.startswith("#"):
                current_chapter = para.lstrip("#").strip()
                current_section = None
            else:
                current_section = para.strip()
            current_chunk += para + "\n\n"
            continue

        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(ChunkWithContext(
                content=current_chunk.strip(),
                chapter=current_chapter,
                section=current_section,
                chunk_index=chunk_index,
                source=source
            ))
            current_chunk = current_chunk[-overlap:] + para + "\n\n"
            chunk_index += 1
        else:
            current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(ChunkWithContext(
            content=current_chunk.strip(),
            chapter=current_chapter,
            section=current_section,
            chunk_index=chunk_index,
            source=source
        ))
    return chunks
```

**考察点：** 编码能力 + 对中文文本结构的理解 + 工程抽象能力。

---

## 4. Embedding模型选型 (Q23-Q29)

**Q23. Embedding 模型的核心作用是什么？在 RAG 中如何选择合适的 Embedding 模型？**
[难度：⭐⭐] [类型：概念]
**答：**
**核心作用：** 将文本（chunk 或 Query）映射为高维向量空间中的点，使语义相似的文本在向量空间中距离更近。

**选型关键维度：**

| 维度 | 说明 |
|------|------|
| **语言支持** | 中文/英文/多语言 |
| **向量维度** | 768/1024/3072，维度越高通常效果越好但成本越高 |
| **最大 token 长度** | 超出长度会被截断，影响长 chunk 的质量 |
| **开源 vs 商业** | 本地部署 vs API 调用 |
| **MTEB 排行榜得分** | 综合检索任务的评测基准 |

**常用模型对比：**

| 模型 | 维度 | 最大长度 | 特点 |
|------|------|---------|------|
| `text-embedding-3-large`（OpenAI） | 3072 | 8192 | 效果好，商业 |
| `bge-large-zh-v1.5`（BAAI） | 1024 | 512 | 中文最优，开源 |
| `multilingual-e5-large`（微软） | 1024 | 512 | 多语言，开源 |
| `all-MiniLM-L6-v2` | 384 | 256 | 轻量快速 |
| `nomic-embed-text` | 768 | 8192 | 长文本友好 |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode(["RAG 技术原理", "检索增强生成"], normalize_embeddings=True)
```

**考察点：** 模型选型的多维权衡，不只是说"用 OpenAI 的"。

---

**Q24. 解释"对称检索"和"非对称检索"的区别，以及对应的 Embedding 模型选择有何不同？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**对称检索（Symmetric Retrieval）：**
- Query 和 Document 的长度相近，语义形式相似
- 例如：找与"苹果是一种水果"最相似的句子
- 适用模型：`all-MiniLM-L6-v2`，`paraphrase-*` 系列

**非对称检索（Asymmetric Retrieval）：**
- Query 是短问句，Document 是长段落（Query 和 Document 语义形式不同）
- 例如：Query="RAG 如何工作？" vs Document="RAG 通过先检索相关文档..."
- 这是 RAG 的典型场景

**非对称检索的挑战：**
- 短 Query 与长 Document 在向量空间中的距离较远，即使语义相关
- 解决方案 1：使用 bi-encoder 模型（如 BGE），训练时专门针对 Q-D 对
- 解决方案 2：Query 扩展，将短 Query 扩展为更完整的表达

**BGE 系列的非对称检索 prompt：**
```python
# BGE 模型在检索 Query 时需要加 instruction 前缀
queries = ["Represent this sentence for searching relevant passages: " + q
           for q in user_queries]
# Document 端不加前缀
passages = [doc.page_content for doc in docs]
```

**考察点：** 对 Embedding 任务类型的深度理解，能区分不同检索场景。

---

**Q25. 什么是 Embedding 模型的"维度折叠"（Matryoshka Representation Learning）？它有什么实际价值？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Matryoshka Representation Learning（MRL，套娃表示学习）：**
训练 Embedding 模型时，使得前 N 维向量本身就是一个有效的低维 Embedding，可以截取任意前缀维度使用。

**类比：** 就像俄罗斯套娃，打开大的里面有小的，小的仍然完整有效。

**实际价值：**
1. **存储成本可调**：同一个模型，可按需使用 768、512、256、128 维度
2. **速度与精度权衡**：低维度检索快，高维度精度高；可先用低维粗筛，再用高维精排
3. **代表模型**：OpenAI `text-embedding-3-*` 系列支持 MRL

```python
from openai import OpenAI
client = OpenAI()

# 使用 text-embedding-3-large 但只用前 256 维，节省存储和计算
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="RAG 技术原理",
    dimensions=256  # 折叠到 256 维
)
embedding = response.data[0].embedding  # 256 维向量
```

**考察点：** 对前沿 Embedding 技术的了解，展示技术广度。

---

**Q26. 如何评估和选择 Embedding 模型？MTEB 排行榜包含哪些评测任务？**
[难度：⭐⭐] [类型：设计]
**答：**
**MTEB（Massive Text Embedding Benchmark）：**
由 HuggingFace 维护，是目前最权威的 Embedding 模型评测基准，包含 56 个数据集，8 类任务。

**8 类评测任务：**
1. **Retrieval（检索）** — 最关键，直接对应 RAG 场景
2. **Clustering（聚类）**
3. **Classification（分类）**
4. **Pair Classification（对分类）**
5. **Reranking（重排序）**
6. **STS（语义文本相似度）**
7. **Summarization（摘要）**
8. **BitextMining（双语对齐）**

**实际评估方法：**
```python
# 在自己的数据集上评估检索性能
from sentence_transformers.evaluation import InformationRetrievalEvaluator

evaluator = InformationRetrievalEvaluator(
    queries=queries_dict,         # {qid: query_text}
    corpus=corpus_dict,           # {doc_id: doc_text}
    relevant_docs=relevant_dict,  # {qid: {relevant_doc_ids}}
    score_functions={"cos_sim": cos_sim}
)

# 评测模型在真实数据上的 NDCG@10、Recall@100 等
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
results = evaluator(model)
print(results)
```

**实践建议：** MTEB 排名仅供参考，最重要的是在自己的领域数据上做评估。

**考察点：** 严谨的模型评估方法论，而非盲目信任排行榜。

---

**Q27. 什么是 Embedding 的"遗忘问题"（Embedding Refresh）？如何处理知识库更新？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**遗忘问题：** 当 Embedding 模型升级或更换时，旧的向量与新模型不兼容，必须对整个知识库重新 Embedding，成本极高。

**知识库更新策略：**

1. **增量更新（推荐）**
   - 新增文档：直接用当前模型 Embedding 后插入向量库
   - 修改文档：删除旧向量，重新 Embedding 后插入
   - 删除文档：从向量库中删除对应 ID 的向量

2. **全量重建**
   - 当 Embedding 模型更换时必须进行
   - 建议：蓝绿部署，新旧向量库并存，新流量走新库，验证通过后切换

3. **混合版本兼容**（不推荐）
   - 新旧模型共存，不同文档用不同模型
   - 会导致检索不公平，不推荐

**工程实践：**
```python
# 版本化管理 Embedding 模型
metadata = {
    "embedding_model": "bge-large-zh-v1.5",
    "embedding_version": "1.5",
    "indexed_at": "2024-01-15"
}
# 查询时可以过滤特定版本的文档
results = vectorstore.similarity_search(
    query, filter={"embedding_version": "1.5"}
)
```

**考察点：** 生产系统的运维思维，知识库的全生命周期管理。

---

**Q28. 解释"双编码器"（Bi-encoder）和"交叉编码器"（Cross-encoder）的区别及其在 RAG 中的应用。**
[难度：⭐⭐⭐] [类型：概念]
**答：**

**双编码器（Bi-encoder）：**
- Query 和 Document 分别独立编码为向量
- 相似度 = cosine(v_query, v_doc)
- 优点：Document 向量可预计算存储，Query 推理快（毫秒级）
- 缺点：Query 和 Document 无法交互，精度有限
- 用途：**向量检索阶段**，对大规模候选集进行快速粗筛

**交叉编码器（Cross-encoder）：**
- Query 和 Document 拼接后一起送入模型，充分交互
- 输出是一个相关性分数（0~1）
- 优点：精度高，考虑了 Q-D 的完整交互信息
- 缺点：无法预计算，每次推理都需要完整前向传播，速度慢（秒级）
- 用途：**Re-ranking 阶段**，对 Top-K 候选进行精确重排

**RAG 中的两阶段检索：**
```
Query → Bi-encoder → Top-100 候选 → Cross-encoder Re-rank → Top-5 → LLM
```

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 重排候选文档
pairs = [(query, doc.page_content) for doc in candidates]
scores = cross_encoder.predict(pairs)

# 按得分降序排序
ranked = sorted(zip(scores, candidates), reverse=True)
top5 = [doc for _, doc in ranked[:5]]
```

**考察点：** 理解两阶段检索的原理，能解释为什么不直接用 Cross-encoder 做全量检索。

---

**Q29. 如何对特定领域（如医疗、法律、金融）进行 Embedding 模型微调？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
通用 Embedding 模型在专业领域效果可能不佳，原因是专业术语、表达方式与通用文本差异大。

**微调方法：**

**1. Contrastive Learning（对比学习，推荐）**
- 构造 (Query, Positive Doc, Negative Doc) 三元组
- 训练目标：拉近 Query 与 Positive Doc 的距离，推远与 Negative Doc 的距离

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 构造训练数据
train_examples = [
    InputExample(texts=["糖尿病诊断标准", "空腹血糖 >= 7.0 mmol/L 可诊断为糖尿病"], label=1.0),
    InputExample(texts=["糖尿病诊断标准", "血压超过 140/90 mmHg 为高血压"], label=0.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

**2. 领域数据预训练**：在领域语料上继续 MLM 预训练，获得领域知识注入

**3. 知识蒸馏**：用大模型（如 GPT-4）生成 Q-D 对标注数据，训练小模型

**数据来源：**
- 领域 FAQ 数据（问题-答案对）
- BM25 挖掘难负样本（硬负样本训练效果更好）
- LLM 生成合成 QA 对

**考察点：** 微调的完整方案，包括数据构造方法，不只是会调用 API。

---

## 5. 向量数据库对比（Chroma/Pinecone/Weaviate/Milvus/pgvector） (Q30-Q37)

**Q30. 对比主流向量数据库（Chroma/Pinecone/Weaviate/Milvus/pgvector），各自的核心特点和适用场景？**
[难度：⭐⭐] [类型：对比]
**答：**

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|---------|
| **Chroma** | 开源/本地 | 轻量、嵌入式、开发友好 | 原型开发、本地测试 |
| **FAISS** | 开源/本地库 | Facebook出品、纯内存、极快 | 批量检索研究 |
| **Pinecone** | 托管云服务 | 全托管、易扩展、SLA保障 | 企业生产、快速上线 |
| **Weaviate** | 开源/可托管 | GraphQL API、多模态、混合检索原生支持 | 复杂检索、知识图谱 |
| **Milvus** | 开源/分布式 | 高性能、可扩展到十亿向量 | 大规模生产部署 |
| **pgvector** | PostgreSQL扩展 | 基于 PG、SQL 操作、事务支持 | 已有 PG 基础设施 |
| **Qdrant** | 开源/Rust实现 | 高性能、向量 payload 过滤 | 生产部署、过滤检索 |

**考察点：** 不同场景的技术选型判断，能给出选择理由而非仅列举名字。

---

**Q31. FAISS 的工作原理是什么？它支持哪些索引类型，各有什么特点？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
FAISS（Facebook AI Similarity Search）是一个高效的向量相似性搜索库，不是数据库，只能在内存中运行，不支持持久化（需要手动保存/加载）。

**主要索引类型：**

| 索引类型 | 全称 | 原理 | 精度 | 速度 | 内存 |
|---------|------|------|------|------|------|
| `IndexFlatL2` | 精确搜索 | 暴力计算所有距离 | 100% | 最慢 | 大 |
| `IndexFlatIP` | 内积搜索 | 暴力计算，用于余弦相似度 | 100% | 慢 | 大 |
| `IndexIVFFlat` | 倒排索引 | 先量化分区，再在分区内搜索 | 高 | 中 | 中 |
| `IndexHNSW` | HNSW | 分层导航小世界图 | 高 | 快 | 大 |
| `IndexIVFPQ` | 乘积量化 | 向量压缩 + 倒排索引 | 中 | 很快 | 小 |

```python
import faiss
import numpy as np

dim = 1024  # 向量维度
n_vectors = 10000

# 精确搜索（适合小规模）
index_flat = faiss.IndexFlatL2(dim)
vectors = np.random.randn(n_vectors, dim).astype(np.float32)
index_flat.add(vectors)

# HNSW 索引（适合大规模）
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # M=32
index_hnsw.add(vectors)

# 搜索
query = np.random.randn(1, dim).astype(np.float32)
D, I = index_hnsw.search(query, k=5)  # D: 距离, I: 索引
print("Top-5 indices:", I[0])
```

**考察点：** 理解索引类型的取舍，能根据数据规模选择合适的索引。

---

**Q32. 解释向量数据库中的"命名空间"（Namespace）和"Collection"概念，如何组织多租户数据？**
[难度：⭐⭐] [类型：设计]
**答：**
**Collection（集合）：** 类似数据库中的"表"，一个 Collection 存储一批向量及其元数据，一个 Collection 中的所有向量通常使用相同维度和度量方式。

**Namespace（命名空间）：** Pinecone 特有概念，在同一 Index 内隔离不同租户的数据，查询时可指定 namespace，实现逻辑隔离。

**多租户设计策略：**

| 策略 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| 独立 Collection | 每个租户一个 Collection | 完全隔离 | Collection 数量爆炸 |
| Namespace | Pinecone namespace | 逻辑隔离，管理简单 | 仅 Pinecone 支持 |
| 元数据过滤 | 添加 tenant_id 元数据 | 实现简单 | 隔离性弱，性能略差 |

```python
# Chroma 多租户示例（按 Collection 隔离）
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 每个用户/租户一个独立的 Collection
tenant_collection = client.get_or_create_collection(
    name=f"tenant_{tenant_id}_documents",
    metadata={"hnsw:space": "cosine"}
)
tenant_collection.add(
    documents=["文档内容..."],
    embeddings=[[0.1, 0.2, ...]],
    ids=["doc_001"]
)
```

**考察点：** 多租户系统设计，理解隔离策略的权衡。

---

**Q33. 什么是向量数据库的"Metadata Filtering"？实现原理是什么？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Metadata Filtering：** 在向量检索时，对存储在向量旁边的元数据字段进行条件过滤，只在满足条件的向量子集中进行 ANN 搜索。

**两种实现方式：**

1. **Pre-filtering（前置过滤）：** 先过滤元数据，再在过滤后的候选集做向量检索
   - 优点：结果集完全满足过滤条件
   - 缺点：过滤后候选集太小可能导致 HNSW 精度下降

2. **Post-filtering（后置过滤）：** 先做向量检索取 Top-M，再过滤元数据
   - 优点：ANN 索引效率高
   - 缺点：最终结果数可能少于 K（有些被过滤掉了）

```python
# Chroma 中的元数据过滤
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "$and": [
            {"doc_type": {"$eq": "financial_report"}},
            {"year": {"$gte": 2023}}
        ]
    }
)

# Qdrant 的过滤
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = qdrant_client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="doc_type", match=MatchValue(value="report")),
            FieldCondition(key="year", range=Range(gte=2023))
        ]
    ),
    limit=5
)
```

**考察点：** 元数据过滤的实现原理和工程使用方式。

---

**Q34. pgvector 与专用向量数据库相比有哪些优劣势？何时应该选择 pgvector？**
[难度：⭐⭐] [类型：场景]
**答：**
**pgvector 优势：**
- 利用现有 PostgreSQL 基础设施，无需引入新系统
- 支持 ACID 事务，向量数据与关系数据可以在同一查询中处理
- 支持 SQL 查询，开发者学习成本低
- 复杂过滤条件可以直接用 SQL WHERE 子句

```sql
-- pgvector 示例：向量检索 + 关系过滤
SELECT id, content, embedding <=> query_embedding AS distance
FROM documents
WHERE
    doc_type = 'legal' AND
    created_at >= '2024-01-01'
ORDER BY embedding <=> $1
LIMIT 5;

-- 创建向量索引
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**pgvector 劣势：**
- 不支持 HNSW（较新版本开始支持，但性能仍弱于专用向量库）
- 大规模（亿级）向量时性能远不如 Milvus/Qdrant
- 向量检索性能受 PG 查询计划器影响

**选择 pgvector 的场景：**
- 数据量 < 1000 万向量
- 需要向量 + 关系数据的联合查询
- 团队已有 PostgreSQL 运维经验
- 希望减少系统复杂度

**考察点：** 能评估技术债务，不是所有场景都需要引入新系统。

---

**Q35. 在生产环境中，如何备份和恢复向量数据库？有哪些注意事项？**
[难度：⭐⭐] [类型：场景]
**答：**
**挑战：** 向量数据库的备份不同于传统数据库，需要同时备份向量数据、元数据和索引结构。

**Milvus 备份策略：**
```bash
# 使用 milvus-backup 工具
milvus-backup create --config backup.yaml --name my_backup

# 恢复
milvus-backup restore --config backup.yaml --backup_name my_backup
```

**Chroma 备份（本地持久化）：**
```python
import chromadb
import shutil

# Chroma 持久化目录直接备份
shutil.copytree("./chroma_db", "./chroma_db_backup")

# 恢复
shutil.copytree("./chroma_db_backup", "./chroma_db")
client = chromadb.PersistentClient(path="./chroma_db")
```

**注意事项：**
1. **索引重建成本**：恢复后可能需要重建 HNSW 索引，耗时较长
2. **增量备份**：记录 document_id 和 indexed_at，可以只备份新增部分
3. **原始数据保存**：建议同时保存原始文档，可以在任何时候重新 Embedding
4. **版本一致性**：备份时记录 Embedding 模型版本，恢复时需使用相同版本

**考察点：** 生产系统的运维意识，不只是会增删查改。

---

**Q36. 如何在 RAG 中实现"向量+元数据"的复合更新（增删改）？**
[难度：⭐⭐⭐] [类型：代码]
**答：**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import hashlib

class RAGVectorStore:
    def __init__(self, persist_dir: str):
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

    def _get_doc_id(self, source: str, chunk_index: int) -> str:
        return hashlib.md5(f"{source}_{chunk_index}".encode()).hexdigest()

    def upsert_document(self, chunks: list, source: str):
        """增量更新：先删除旧版本，再插入新版本"""
        # 1. 删除同一 source 的旧 chunk
        existing = self.store.get(where={"source": source})
        if existing["ids"]:
            self.store.delete(ids=existing["ids"])

        # 2. 插入新 chunk
        texts = [c["content"] for c in chunks]
        metadatas = [{"source": source, "chunk_index": i, **c.get("metadata", {})}
                     for i, c in enumerate(chunks)]
        ids = [self._get_doc_id(source, i) for i in range(len(chunks))]
        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    def delete_document(self, source: str):
        """删除指定文档的所有 chunk"""
        existing = self.store.get(where={"source": source})
        if existing["ids"]:
            self.store.delete(ids=existing["ids"])
        return len(existing["ids"])

    def search(self, query: str, k: int = 5, filter_dict: dict = None):
        return self.store.similarity_search(
            query, k=k, filter=filter_dict
        )

# 使用示例
rag_store = RAGVectorStore(persist_dir="./rag_db")
chunks = [{"content": "RAG技术原理...", "metadata": {"chapter": "第一章"}}]
rag_store.upsert_document(chunks, source="rag_guide.pdf")
```

**考察点：** 知识库的 CRUD 工程实现，不只是会添加文档。

---

**Q37. 解释向量数据库中的"ANN"（近似最近邻）搜索原理，为什么不用精确搜索？**
[难度：⭐⭐] [类型：概念]
**答：**
**精确最近邻搜索（Exact NN）：**
- 计算 Query 向量与所有 N 个向量的距离
- 时间复杂度：O(N × d)（d 为维度）
- 1亿个 1024 维向量，每次搜索需计算 1 × 10^11 次浮点运算，约几十秒

**近似最近邻搜索（ANN）：**
- 通过数据结构（HNSW、IVF 等）提前组织向量，牺牲少量精度换取极大速度提升
- 实际中 ANN 的召回率（recall@k）可以达到 95~99%
- 速度可以比精确搜索快 1000 倍以上

**主流 ANN 算法：**
1. **HNSW**（Hierarchical Navigable Small World）：图结构，最常用，精度高速度快
2. **IVF**（Inverted File Index）：向量聚类 + 倒排索引，内存占用小
3. **LSH**（Locality-Sensitive Hashing）：哈希碰撞，理论性能好但实际效果一般

**RAG 中的影响：**
ANN 偶尔会"漏掉"最相关的文档，这是检索质量损失的来源之一。可以通过：
- 扩大 `ef_search`（HNSW 搜索范围）提升召回率，但速度变慢
- 先取 Top-100（宽召回），再用 Re-ranker 精排到 Top-5

**考察点：** ANN 的本质是精度与速度的权衡，理解这一点对系统优化很重要。

---

## 6. 相似度算法与HNSW索引 (Q38-Q44)

**Q38. 在 RAG 向量检索中，余弦相似度、点积和欧氏距离有什么区别？如何选择？**
[难度：⭐⭐] [类型：概念]
**答：**

| 度量方式 | 公式 | 范围 | 考虑长度 | 适用场景 |
|---------|------|------|---------|---------|
| **余弦相似度** | cos(θ) = A·B / (||A||·||B||) | [-1, 1] | 否（归一化） | 文本语义相似度（最常用） |
| **点积（内积）** | A·B = Σ(a_i × b_i) | (-∞, +∞) | 是 | 归一化向量下等同余弦；搜索重要性 |
| **L2 欧氏距离** | ||A-B|| = √Σ(a_i-b_i)² | [0, +∞) | 是 | 图像特征、低维空间 |

**实践建议：**
- **文本 RAG**：使用**余弦相似度**，大多数 Embedding 模型的训练目标就是余弦相似度
- **归一化向量**：如果向量已 L2 归一化，点积和余弦相似度等价，计算更快
- **不要混用**：Embedding 模型用哪种度量训练，检索时就用哪种

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# BGE 模型输出已归一化，直接计算点积即等于余弦相似度
embeddings = model.encode(
    ["RAG 技术", "检索增强生成"],
    normalize_embeddings=True
)
similarity = np.dot(embeddings[0], embeddings[1])
print(f"相似度: {similarity:.4f}")  # 0.95+
```

**考察点：** 理解三种度量的本质区别，能根据模型特性选择正确的度量方式。

---

**Q39. HNSW（分层导航小世界图）索引的工作原理是什么？M 和 ef_construction 参数如何影响性能？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**HNSW 工作原理：**
HNSW 是一种多层图结构，受"小世界网络"理论启发：
- **底层（Layer 0）**：包含所有向量节点，每个节点有 M 个最近邻连接
- **高层（Layer 1, 2, ...）**：随机选取部分节点，形成稀疏长跳连接（类似高速公路）
- **搜索**：从最高层的入口节点开始，贪心地向目标查询方向移动，逐层下降

**关键参数：**

| 参数 | 含义 | 增大效果 | 权衡 |
|------|------|---------|------|
| **M** | 每个节点的双向连接数 | 精度提升 | 内存增加，构建变慢 |
| **ef_construction** | 构建时的搜索宽度 | 索引质量提升 | 构建时间增加 |
| **ef_search** | 搜索时的候选集大小 | 召回率提升 | 查询变慢 |

**参数推荐：**
- M = 16~64（文档数 < 100万：M=16；大规模：M=32~64）
- ef_construction = 100~200
- ef_search ≥ K（至少等于返回的 Top-K 数）

```python
import hnswlib
import numpy as np

dim = 1024
num_elements = 100000

# 创建 HNSW 索引
p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=32)

# 添加向量
data = np.random.randn(num_elements, dim).astype(np.float32)
p.add_items(data)

# 设置查询精度
p.set_ef(50)  # ef_search，需 >= k

# 搜索
query = np.random.randn(1, dim).astype(np.float32)
labels, distances = p.knn_query(query, k=5)
```

**考察点：** HNSW 的分层思想，以及参数对性能的精确影响。

---

**Q40. 什么是"向量量化"（Product Quantization）？它如何减少内存占用？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**乘积量化（Product Quantization, PQ）：**
1. 将 d 维向量切分为 M 个子空间（每个子空间 d/M 维）
2. 对每个子空间独立进行 K-means 聚类，得到 k 个代码本（codebook）
3. 每个子向量用最近聚类中心的索引（code）表示

**压缩效果：**
- 原始：每个 float32 向量 = 1024 × 4 = 4096 bytes
- PQ 压缩：64 个子空间，每个 256 个中心 = 64 bytes（压缩 64 倍）

**FAISS IVF + PQ 示例：**
```python
import faiss
import numpy as np

d = 1024      # 向量维度
nlist = 100   # IVF 分区数
m = 64        # PQ 子空间数（需整除 d）
bits = 8      # 每个子空间的编码位数 (2^8=256 个中心)

# 创建 IVFPQ 索引
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# 训练（需要训练集）
train_data = np.random.randn(50000, d).astype(np.float32)
index.train(train_data)

# 添加向量
vectors = np.random.randn(100000, d).astype(np.float32)
index.add(vectors)

# 搜索前设置探测分区数
index.nprobe = 10
query = np.random.randn(1, d).astype(np.float32)
D, I = index.search(query, k=5)
```

**PQ 的缺点：** 量化带来精度损失，通常与 HNSW 配合使用（HNSW+PQ）。

**考察点：** 大规模向量存储的工程优化，理解精度-存储的权衡。

---

**Q41. 解释"Recall@K"在向量检索中的含义，以及如何提高它？**
[难度：⭐⭐] [类型：设计]
**答：**
**Recall@K 定义：**
在检索的 Top-K 结果中，真正相关的文档占所有相关文档的比例。

```
Recall@K = |检索到的相关文档| / |所有相关文档|
```

**示例：**
- 真正相关文档共 5 篇
- Top-5 检索结果中有 4 篇相关 → Recall@5 = 4/5 = 80%

**提高 Recall@K 的方法：**

1. **增大 K 值**：Top-10 的召回率通常高于 Top-5

2. **优化 HNSW 参数**：
   ```python
   # 增大 ef_search（搜索候选集）
   index.hnsw.ef_search = 256  # 默认 50
   ```

3. **Query 扩展**：用多个 Query 变体检索，取并集

4. **改善 Embedding 质量**：使用更好的模型或领域微调

5. **混合检索**：BM25 + 向量检索，互补召回

6. **增加 M 值**：HNSW 连接数更多，图更稠密，召回率更高（内存代价）

**召回率与精确率的权衡：**
- 大 K 提高召回率，但降低精确率（噪声增多）
- 解决方案：大 K 粗筛 + Re-ranker 精排

**考察点：** 检索质量的定量分析，以及系统化的优化思路。

---

**Q42. 向量检索中的"维度灾难"是什么？Embedding 维度越高越好吗？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**维度灾难（Curse of Dimensionality）：**
在高维空间中，所有点之间的距离趋于相等，使得"最近邻"变得没有区分度，ANN 索引效率急剧下降。

**实验验证：**
```python
import numpy as np

for dim in [10, 100, 1000, 10000]:
    n = 1000
    vectors = np.random.randn(n, dim)
    query = np.random.randn(1, dim)

    dists = np.linalg.norm(vectors - query, axis=1)
    ratio = dists.max() / (dists.min() + 1e-9)
    print(f"dim={dim}: max/min 距离比 = {ratio:.2f}")
# 维度越高，ratio 越接近 1.0，区分度越差
```

**Embedding 维度越高越好吗？**
- 通常情况下，更高维度 = 更强的表达能力 = 更好的语义区分
- 但实际应用中需要权衡：
  - 维度从 768 升到 3072，检索精度提升有限（边际递减）
  - 存储成本是 768 维的 4 倍
  - ANN 搜索时间随维度增加

**实践结论：**
- 768 维：适合大多数 RAG 场景
- 1024 维：中文检索推荐（BGE-large）
- 3072 维：需要极高精度且不在乎成本

**考察点：** 理解维度与性能的非线性关系，做出有依据的工程决策。

---

**Q43. 如何为向量检索系统设置合理的相似度阈值（Similarity Threshold）？**
[难度：⭐⭐] [类型：设计]
**答：**
相似度阈值用于过滤掉与 Query 相关度过低的检索结果，避免将无关内容送入 LLM。

**确定阈值的方法：**

1. **统计分析法：**
   - 在测试集上分析 (Query, Relevant Doc) 和 (Query, Irrelevant Doc) 的相似度分布
   - 选取两个分布的分离点作为阈值

2. **业务驱动法：**
   - 若宁可不回答也不要错误回答：设置高阈值（0.8+）
   - 若要求覆盖率高：设置低阈值（0.6+）

3. **自适应阈值：**
   ```python
   def adaptive_filter(results, query, min_k=1, threshold=0.7):
       filtered = [r for r in results if r.score >= threshold]
       if len(filtered) < min_k:
           # 即使没有结果达到阈值，也返回最高分的 min_k 个（而不是不回答）
           return sorted(results, key=lambda x: x.score, reverse=True)[:min_k]
       return filtered
   ```

4. **余弦相似度参考值：**
   - >= 0.9：高度相关
   - 0.7~0.9：中度相关，通常有效
   - 0.5~0.7：弱相关，需要谨慎
   - < 0.5：几乎无关，建议过滤

**注意：** 不同 Embedding 模型的相似度分布不同，阈值不能跨模型复用。

**考察点：** 数据驱动的阈值设定方法，而非拍脑袋给一个数字。

---

**Q44. 在多向量检索（Multi-Vector Retrieval）中，如何处理一个文档对应多个 Embedding 的情况？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**多向量检索的场景：**
1. 一个文档被切成多个 chunk，每个 chunk 有独立 Embedding
2. 同一文档生成多个角度的摘要 Embedding
3. ColBERT：每个 token 都有独立向量

**文档-多向量的管理策略：**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class MultiVectorRetriever:
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        # 向量库：存 chunk 级别的向量
        self.vectorstore = Chroma(embedding_function=self.embedding)
        # 文档库：存完整文档，用于最终检索
        self.docstore = {}

    def index_document(self, doc_id: str, full_content: str, chunks: list):
        self.docstore[doc_id] = full_content
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.vectorstore.add_texts(
                texts=[chunk],
                ids=[chunk_id],
                metadatas=[{"doc_id": doc_id, "chunk_index": i}]
            )

    def retrieve(self, query: str, k: int = 3):
        # 检索相关 chunk
        chunk_results = self.vectorstore.similarity_search(query, k=k*3)
        # 根据 doc_id 聚合，返回完整文档
        seen_docs = set()
        retrieved_docs = []
        for chunk in chunk_results:
            doc_id = chunk.metadata["doc_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                retrieved_docs.append(self.docstore[doc_id])
            if len(retrieved_docs) >= k:
                break
        return retrieved_docs
```

**考察点：** 多向量索引的工程设计，以及 chunk 级检索与文档级输出的分离。

---

## 7. 混合检索（向量+BM25+RRF） (Q45-Q51)

**Q45. 解释 BM25 算法的工作原理，以及它与向量检索的互补性。**
[难度：⭐⭐] [类型：概念]
**答：**
**BM25（Best Match 25）：** 基于词频（TF）和逆文档频率（IDF）的统计检索模型，是关键词检索的黄金标准。

**BM25 公式：**
```
Score(D, Q) = Σ IDF(qi) × TF(qi, D) × (k1+1) / (TF(qi, D) + k1×(1 - b + b×|D|/avgdl))
```
- `k1`：词频饱和度（通常 1.2~2.0）
- `b`：文档长度归一化因子（0.75）
- `|D|/avgdl`：文档长度与平均长度的比值

**BM25 vs 向量检索：**

| 维度 | BM25 | 向量检索 |
|------|------|---------|
| 检索方式 | 精确关键词匹配 | 语义相似度 |
| 优势 | 术语/实体名精确匹配 | 语义理解、同义词 |
| 劣势 | 无法理解语义 | 关键词完全匹配时表现差 |
| 速度 | 极快 | 快（ANN） |

**互补性示例：**
- Query："Python 3.12 有哪些新特性" → BM25 擅长（精确版本号）
- Query："如何提高代码运行速度" → 向量检索擅长（语义相关的"优化"、"性能"等）

```python
from rank_bm25 import BM25Okapi
import jieba

corpus = ["RAG技术通过检索增强生成效果", "向量数据库存储文档embedding"]
tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query_tokens = list(jieba.cut("如何改善RAG检索质量"))
scores = bm25.get_scores(query_tokens)
```

**考察点：** 理解两种检索方式的本质差异，能分析它们的互补性。

---

**Q46. 什么是 RRF（Reciprocal Rank Fusion）？如何用它融合多路检索结果？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**RRF（Reciprocal Rank Fusion）：** 一种简单有效的排序融合算法，将多个检索系统的排名结果合并为一个统一排名。

**RRF 公式：**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```
- `k`：常数，通常取 60（防止最高排名的权重过大）
- `rank_i(d)`：文档 d 在第 i 个检索系统中的排名（从 1 开始）

**示例：**
```
向量检索结果：  doc_A(rank=1), doc_B(rank=2), doc_C(rank=3)
BM25 检索结果：  doc_B(rank=1), doc_D(rank=2), doc_A(rank=3)

RRF_score(doc_A) = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
RRF_score(doc_B) = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  ← 最高
RRF_score(doc_C) = 1/(60+3)              = 0.01563
RRF_score(doc_D) = 1/(60+2)              = 0.01613

最终排序：doc_B > doc_A > doc_D > doc_C
```

**代码实现：**
```python
from collections import defaultdict

def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    scores = defaultdict(float)
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.metadata.get("id") or doc.page_content[:50]
            scores[doc_id] += 1.0 / (k + rank)
    # 按 RRF 分数降序排列
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# 融合两路检索结果
vector_results = vectorstore.similarity_search(query, k=10)
bm25_results = bm25_retriever.get_relevant_documents(query)[:10]
fused = reciprocal_rank_fusion([vector_results, bm25_results])
```

**考察点：** RRF 算法的公式推导和代码实现，是混合检索的核心组件。

---

**Q47. 在 LangChain/LlamaIndex 中如何实现混合检索（Hybrid Search）？**
[难度：⭐⭐] [类型：代码]
**答：**
```python
# LangChain 混合检索实现
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. 准备文档
from langchain.schema import Document
docs = [
    Document(page_content="RAG 通过检索外部知识增强生成效果",
             metadata={"source": "rag_guide"}),
    Document(page_content="向量数据库用于存储和检索高维向量",
             metadata={"source": "db_guide"}),
    Document(page_content="BM25 是一种基于 TF-IDF 的检索算法",
             metadata={"source": "ir_guide"}),
]

# 2. 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 3. 创建向量检索器
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. EnsembleRetriever 融合（内置 RRF）
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25 权重 0.4，向量权重 0.6
)

# 5. 检索
results = hybrid_retriever.invoke("如何提升 RAG 检索质量")
for r in results:
    print(r.page_content)
```

**Weaviate 原生混合检索（BM25 + 向量融合）：**
```python
results = client.collections.get("Document").query.hybrid(
    query="RAG retrieval quality",
    alpha=0.5,  # 0: 纯BM25, 1: 纯向量, 0.5: 均衡
    limit=5
)
```

**考察点：** 混合检索的工程实现，不只是理论。

---

**Q48. 混合检索中，如何动态调整 BM25 和向量检索的权重？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
固定权重（如 BM25=0.4, 向量=0.6）适合通用场景，但不同 Query 类型需要不同权重。

**Query 类型分类与权重策略：**

| Query 类型 | 特征 | 建议权重 |
|-----------|------|---------|
| 实体查询 | 含专有名词、版本号、人名 | BM25 更高（0.7） |
| 语义查询 | 自然语言问句、概念理解 | 向量更高（0.8） |
| 混合查询 | 既有术语又有语义意图 | 均衡（0.5） |

**动态权重实现：**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def classify_query(query: str) -> float:
    """返回向量检索的权重（0~1）"""
    # 简单启发式：含引号或特殊符号 → 关键词检索为主
    import re
    has_exact_terms = bool(re.search(r'["'`]|v\d+\.\d+|\bAPI\b|\bSQL\b', query))
    if has_exact_terms:
        return 0.3  # 更依赖 BM25
    return 0.7      # 更依赖向量检索

def hybrid_search(query, vectorstore, bm25_retriever, k=5):
    vector_weight = classify_query(query)
    bm25_weight = 1 - vector_weight

    # 获取两路结果
    vec_results = vectorstore.similarity_search(query, k=k*2)
    bm25_results = bm25_retriever.get_relevant_documents(query)[:k*2]

    # 加权 RRF
    scores = {}
    for rank, doc in enumerate(vec_results, 1):
        doc_id = doc.page_content[:50]
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (60 + rank)
    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc.page_content[:50]
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (60 + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

**考察点：** 动态调整策略，对不同检索场景的深度理解。

---

**Q49. 稀疏向量（Sparse Vector）和稠密向量（Dense Vector）有什么区别？SPLADE 模型是什么？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**稠密向量（Dense Vector）：**
- 所有维度都有非零值
- 维度：768~3072
- 来源：Transformer Embedding 模型
- 优势：语义理解能力强

**稀疏向量（Sparse Vector）：**
- 大多数维度为 0，只有少数维度有值
- 维度：词汇表大小（3万~10万）
- 传统方法：TF-IDF、BM25
- 新方法：SPLADE 等基于 Transformer 的稀疏模型

**SPLADE（Sparse Lexical and Expansion）：**
- 用 Transformer 生成稀疏向量，结合了神经网络语义理解和稀疏检索的高效性
- 能自动扩展词汇（将"汽车"扩展为"vehicle"、"car"等），避免词汇不匹配

**混合稀疏+稠密检索（Pinecone/Qdrant 支持）：**
```python
# Qdrant 混合检索示例
from qdrant_client.models import SparseVector, NamedSparseVector

# 同时存储稠密和稀疏向量
qdrant_client.upsert(
    collection_name="hybrid_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": dense_embedding,      # 向量检索用
                "sparse": SparseVector(indices=[100, 200, 500], values=[0.5, 0.3, 0.8])  # BM25/SPLADE 用
            },
            payload={"content": "..."}
        )
    ]
)
```

**考察点：** 对稀疏/稠密向量的区分，以及 SPLADE 等新方法的了解。

---

**Q50. 在混合检索中，如何处理中英文混合的检索场景？**
[难度：⭐⭐] [类型：场景]
**答：**
中英文混合文档在 RAG 中非常常见（如技术文档包含 API 名称、代码片段等英文内容）。

**挑战：**
1. BM25 分词：中文需要分词，英文需要 tokenize，不能用同一种方法
2. 向量模型：需要支持双语或多语言的 Embedding 模型
3. 检索时语言路由：Query 可能是中文但关键词是英文

**解决方案：**
```python
import jieba
import re

def mixed_tokenizer(text: str) -> list:
    """中英文混合分词器"""
    # 先提取英文单词和数字（保护专有名词）
    english_tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*|\d+\.\d+|\d+', text)
    # 中文部分分词
    chinese_text = re.sub(r'[a-zA-Z0-9_\.]+', ' ', text)
    chinese_tokens = list(jieba.cut(chinese_text))
    # 合并，过滤空白
    all_tokens = [t.strip() for t in chinese_tokens + english_tokens if t.strip()]
    return all_tokens

# 构建支持中英文的 BM25
from rank_bm25 import BM25Okapi
corpus = ["RAG技术原理：Retrieval-Augmented Generation", "向量数据库：Chroma vs Pinecone"]
tokenized = [mixed_tokenizer(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized)

# 多语言 Embedding 模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")  # 支持100+语言，中英最优
```

**考察点：** 对多语言 RAG 系统的实际工程挑战的理解。

---

**Q51. 什么是"查询路由"（Query Routing）？如何根据 Query 类型自动选择最合适的检索器？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**查询路由：** 根据 Query 的特征（类型、意图、关键词特征），自动将 Query 发送给最合适的检索器或知识库。

**路由维度：**
1. **关键词 vs 语义**：选择 BM25 或向量检索
2. **文档类型**：财务报告、技术文档、FAQ 分别索引不同的 Collection
3. **检索粒度**：需要精确事实 → 小 chunk；需要深度分析 → 大 chunk
4. **领域路由**：医疗问题走医疗知识库，法律问题走法律知识库

**LLM 路由（最灵活）：**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from enum import Enum

class QueryType(str, Enum):
    FACTUAL = "factual"       # 事实查询 → BM25 + 小 chunk
    SEMANTIC = "semantic"     # 语义查询 → 向量 + 大 chunk
    ANALYTICAL = "analytical" # 分析查询 → 多步检索

def route_query(query: str, llm: ChatOpenAI) -> QueryType:
    prompt = ChatPromptTemplate.from_template(
        "以下查询的类型是什么？\n"
        "- factual: 需要精确信息，如人名、日期、版本号\n"
        "- semantic: 需要理解概念或解释\n"
        "- analytical: 需要综合分析多个文档\n"
        "查询：{query}\n只回答类型名称。"
    )
    response = llm.invoke(prompt.format_messages(query=query))
    return QueryType(response.content.strip().lower())

# 根据路由结果选择检索策略
query_type = route_query("Python 3.12 的 f-string 语法有什么变化？", llm)
if query_type == QueryType.FACTUAL:
    retriever = bm25_retriever
elif query_type == QueryType.SEMANTIC:
    retriever = vector_retriever
else:
    retriever = hybrid_retriever
```

**考察点：** 系统级思维，将检索作为一个可配置的模块而非固定流程。

---

## 8. Re-ranking技术 (Q52-Q58)

**Q52. 什么是 Re-ranking？为什么仅靠向量检索不够，还需要重排？**
[难度：⭐⭐] [类型：概念]
**答：**
**Re-ranking（重排序）：** 对向量检索返回的 Top-K 候选结果，使用更精确（但更慢）的模型重新评分排序，将最相关的文档排在前面。

**向量检索的局限性：**
1. **双塔结构局限**：Bi-encoder 中 Query 和 Document 独立编码，无法充分交互
2. **精度瓶颈**：ANN 近似索引可能错误排序相关文档
3. **语义漂移**：高维向量空间中细微的语义差别难以区分

**两阶段检索架构：**
```
Stage 1: 向量检索（Bi-encoder）→ Top-100 候选（毫秒级，牺牲精度）
Stage 2: Re-ranking（Cross-encoder）→ Top-5 精确结果（秒级，高精度）
```

**Re-ranking 的价值：**
实验数据显示，在 Top-100 候选上进行 Re-ranking，NDCG@5 通常提升 10~30%。

**考察点：** 理解为什么需要两阶段，以及 Bi-encoder/Cross-encoder 的互补关系。

---

**Q53. 主流的 Re-ranker 模型有哪些？如何在 Python 中使用它们？**
[难度：⭐⭐] [类型：代码]
**答：**

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 轻量、英文、快速 | 英文文档 |
| `BAAI/bge-reranker-large` | 中文优秀、开源 | 中文文档 |
| `Cohere Rerank API` | 托管服务、多语言 | 快速集成 |
| `mixedbread-ai/mxbai-rerank-large-v1` | SOTA 性能 | 高精度需求 |

```python
from sentence_transformers import CrossEncoder
from langchain.schema import Document

# 使用 BGE Re-ranker（中文推荐）
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank_documents(query: str, candidates: list[Document], top_k: int = 5) -> list[Document]:
    """对候选文档进行重排序"""
    # 构造 (query, document) 对
    pairs = [(query, doc.page_content) for doc in candidates]
    # 批量评分
    scores = reranker.predict(pairs, show_progress_bar=False)
    # 按得分降序排序
    scored_docs = list(zip(scores, candidates))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# 使用 Cohere Rerank（商业 API）
import cohere
co = cohere.Client("API_KEY")

def cohere_rerank(query: str, docs: list[str], top_k: int = 5):
    results = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return results.results
```

**考察点：** 会使用主流重排模型，理解开源 vs 商业 API 的选择。

---

**Q54. 解释 Re-ranking 中 Cross-encoder 的计算过程，为什么它比 Bi-encoder 更精确？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Bi-encoder（双塔）：**
```
Query → Encoder → q_vec
Doc   → Encoder → d_vec
Score = cosine(q_vec, d_vec)
```
Query 和 Doc 独立编码，没有交叉注意力，信息交互发生在最后的向量点积中。

**Cross-encoder（交叉编码）：**
```
[CLS] Query [SEP] Document [SEP] → BERT → Linear → Relevance Score
```
Query 和 Document 拼接后一起进入模型，每一层 Transformer 都有完整的双向注意力，Query 中的每个词都可以"看到" Document 中的所有词。

**为什么更精确：**
1. 完整的 Query-Document 交互
2. 注意力机制能捕捉细微的语义差别（如否定、比较级等）
3. 无需压缩到固定维度向量，保留更多语义信息

**速度对比（1000 个候选文档）：**
```python
import time

# Bi-encoder：向量已预计算，只需计算 1 次 Query
start = time.time()
query_vec = bi_encoder.encode(query)
scores = np.dot(doc_vecs, query_vec)  # 向量化操作
print(f"Bi-encoder: {(time.time()-start)*1000:.1f}ms")

# Cross-encoder：需要处理 1000 个 (query, doc) 对
start = time.time()
pairs = [(query, doc) for doc in docs]
scores = cross_encoder.predict(pairs)
print(f"Cross-encoder: {(time.time()-start)*1000:.1f}ms")
```

**考察点：** 对两种 Encoder 架构的深度理解，能从注意力机制角度解释差异。

---

**Q55. 什么是 "Contextual Compression" 重排策略？如何减少发送给 LLM 的冗余内容？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**Contextual Compression（上下文压缩）：** 不是对文档整体排名，而是从检索到的文档中提取与 Query 最相关的句子/段落，去掉不相关的部分，减少发送给 LLM 的 token 数量。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 方法一：使用 LLM 提取相关内容（精度高，成本高）
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)
results = compression_retriever.invoke("什么是 HNSW 索引？")
# 返回的每个文档只包含与 Query 相关的片段

# 方法二：使用 Embeddings 过滤（速度快，成本低）
embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76
)
compression_retriever2 = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)
```

**效果：**
- 平均可将每个文档 chunk 压缩 50~80%
- LLM 处理更少 token，速度更快，成本更低
- 减少 "Lost in the Middle" 效应

**考察点：** 理解 Contextual Compression 的原理，以及它与 Re-ranking 的区别。

---

**Q56. FlashRank、LLMListwiseRerank 等不同重排策略有何区别？**
[难度：⭐⭐⭐] [类型：对比]
**答：**

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Pointwise** | 对每个 (Q, D) 对独立打分 | 简单，并行计算 | 不考虑文档间相对关系 |
| **Pairwise** | 对 (D_i, D_j) 比较哪个更相关 | 更准确 | 复杂度 O(n²) |
| **Listwise** | 一次性评估整个候选列表 | 考虑全局关系 | 上下文长，成本高 |

**LLM Listwise Re-ranking（RankGPT）：**
```python
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_openai import ChatOpenAI

# 使用 LLM 对候选列表整体重排
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reranker = LLMListwiseRerank.from_llm(llm, top_n=5)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)
```

**FlashRank（超轻量级 Re-ranker）：**
```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
rerankrequest = RerankRequest(query="RAG 检索质量", passages=passages)
results = ranker.rerank(rerankrequest)
```

**考察点：** 了解重排策略的分类，能根据性能需求选择合适的方案。

---

**Q57. 如何评估 Re-ranker 的效果？用什么指标衡量重排前后的质量差异？**
[难度：⭐⭐] [类型：设计]
**答：**
**核心评估指标：**

1. **NDCG@K（归一化折损累积增益）：** 考虑排名位置的相关性评分
   ```python
   from sklearn.metrics import ndcg_score
   import numpy as np

   # 真实相关性得分（1=相关, 0=不相关）
   true_relevance = np.array([[1, 0, 1, 0, 1]])
   # 模型给出的得分
   scores_before = np.array([[0.9, 0.7, 0.5, 0.4, 0.3]])  # 重排前
   scores_after  = np.array([[0.95, 0.8, 0.6, 0.1, 0.05]]) # 重排后

   print("NDCG@5 before:", ndcg_score(true_relevance, scores_before))
   print("NDCG@5 after: ", ndcg_score(true_relevance, scores_after))
   ```

2. **MRR（Mean Reciprocal Rank）：** 第一个相关文档的排名倒数均值

3. **Hit Rate@K：** Top-K 中至少包含一个相关文档的查询比例

4. **Answer Quality（端到端）：** 将重排结果送入 LLM，评估最终答案质量（RAGAS 分数）

**评估流程：**
```python
# 构建评估集
test_queries = [
    {"query": "什么是 HNSW？", "relevant_ids": ["doc_12", "doc_34"]},
    {"query": "如何优化 RAG？", "relevant_ids": ["doc_56"]},
]

# 对比重排前后
for test in test_queries:
    before_results = vectorstore.similarity_search(test["query"], k=10)
    after_results = reranker_retriever.invoke(test["query"])
    # 计算 Recall@5
    before_recall = len(set([d.metadata["id"] for d in before_results[:5]]) & set(test["relevant_ids"])) / len(test["relevant_ids"])
    after_recall  = len(set([d.metadata["id"] for d in after_results[:5]])  & set(test["relevant_ids"])) / len(test["relevant_ids"])
    print(f"Recall@5: {before_recall:.2f} → {after_recall:.2f}")
```

**考察点：** 严格的评估方法论，能量化重排带来的收益。

---

**Q58. Re-ranking 会增加多少延迟？在实际生产中如何优化重排的性能？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**延迟分析：**
- 向量检索：10~50ms
- Cross-encoder Re-ranking（Top-20）：200~500ms（本地 GPU），1~2s（CPU）
- LLM Re-ranking（GPT-4o-mini, Top-20）：2~5s

**生产优化策略：**

1. **减少候选数量：** Re-ranking Top-20 而非 Top-100，速度提升 5x
   ```python
   # 分段召回：先取 50，再重排 20
   candidates = retriever.get_relevant_documents(query)[:50]
   reranked = reranker.rerank(query, candidates[:20])
   ```

2. **使用轻量模型：** FlashRank、ms-marco-MiniLM-L-6-v2（参数少、速度快）

3. **GPU 加速：** 将 Cross-encoder 部署在 GPU 上，速度提升 10x+

4. **批处理：** 将多个 Query 组成 batch 批量 Re-ranking

5. **异步管道：** 在生成阶段开始同时进行 Re-ranking（流式输出时）

6. **缓存：** 对相同/相似 Query 缓存 Re-ranking 结果（LRU 缓存）
   ```python
   import functools

   @functools.lru_cache(maxsize=1000)
   def cached_rerank(query: str, doc_ids_tuple: tuple) -> list:
       docs = [docstore.get(id) for id in doc_ids_tuple]
       return reranker.rerank(query, docs)
   ```

**延迟预算参考（总体 < 2s 的典型分配）：**
- 向量检索：50ms
- Re-ranking（Top-20）：300ms
- LLM 生成（流式）：500ms ~ ongoing

**考察点：** 工程权衡意识，理解延迟预算分配和优化手段。

---

## 9. Advanced RAG（HyDE/Self-Query/Contextual Compression） (Q59-Q66)

**Q59. 什么是 HyDE（Hypothetical Document Embeddings）？它如何提升检索效果？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**HyDE 核心思想：**
用 LLM 根据用户 Query 生成一个"假设性文档"（即 LLM 认为这个 Query 的答案长什么样），然后对这个假设性文档进行 Embedding，用生成文档的向量去检索知识库，而不是用原始 Query 的向量。

**为什么有效：**
- Query 通常是短问句，与知识库中的长文档在向量空间中距离较远
- 假设性文档与真实文档形式更接近（都是描述性文本），语义对齐更好
- 即使生成的文档内容有误，其向量仍能捕捉相关语义方向

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# 生成假设性文档的 Prompt
hyde_prompt = ChatPromptTemplate.from_template(
    "请根据以下问题，写一段详细的回答文档（约100字）：\n{question}"
)

# 创建 HyDE Embedder
embeddings = OpenAIEmbeddings()
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    embeddings=embeddings,
    prompt_template=hyde_prompt,
    n=1  # 生成1个假设文档
)

# 使用 HyDE Embedder 检索
query = "什么是 HNSW 索引？"
results = vectorstore.similarity_search_by_vector(
    embedding=hyde_embedder.embed_query(query),
    k=5
)
```

**适用场景：** 知识库文档是长段落，用户 Query 是简短问句的场景。

**考察点：** 理解 HyDE 的核心洞察——用"假设答案"对齐文档分布。

---

**Q60. 什么是 Self-Query Retriever？LLM 如何从 Query 中自动提取元数据过滤条件？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**Self-Query Retriever：** 使用 LLM 将自然语言查询解析为两部分：语义查询 + 元数据过滤条件，自动实现结构化检索。

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 定义文档的元数据字段（让 LLM 知道可以过滤哪些字段）
metadata_field_info = [
    AttributeInfo(name="doc_type", description="文档类型", type="string"),
    AttributeInfo(name="year", description="文档年份", type="integer"),
    AttributeInfo(name="department", description="部门名称", type="string"),
]

document_content_description = "公司内部文档，包含财务报告、技术文档、HR 政策等"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

# 用户自然语言查询
result = retriever.invoke("2024年财务部门的季报中有哪些关于成本的内容？")
# LLM 自动解析为：
# 语义查询: "成本"
# 过滤: year=2024 AND department="财务" AND doc_type="季报"
```

**Self-Query 的局限性：**
- 依赖 LLM 正确解析，偶尔会出错
- 元数据字段需要提前定义，无法处理动态字段
- 对复杂条件（如 OR 逻辑）支持有限

**考察点：** 理解结构化过滤与语义检索的结合，以及 LLM 在其中的角色。

---

**Q61. 什么是"多查询检索"（Multi-Query Retrieval）？如何用它提升召回率？**
[难度：⭐⭐] [类型：设计]
**答：**
**核心思想：** 用 LLM 将原始 Query 扩展为多个不同表达方式的子查询，分别检索后合并结果，提升召回率。

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# MultiQueryRetriever 内置了生成子查询的 Prompt
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# 原始 Query: "RAG 的性能优化方法"
# LLM 生成多个变体：
# 1. "如何提升检索增强生成系统的速度？"
# 2. "RAG pipeline 有哪些优化技巧？"
# 3. "减少 RAG 延迟的最佳实践是什么？"

results = retriever.invoke("RAG 的性能优化方法")
# 返回 3 × 5 = 15 个候选（去重后约 8~12 个独特结果）
```

**与 HyDE 的区别：**
- HyDE：生成假设性答案文档，改变 Embedding 分布
- Multi-Query：生成多个同义 Query，增加检索覆盖面

**考察点：** Query 扩展的工程实现，以及与单次检索的召回率对比。

---

**Q62. 解释 RAG-Fusion 技术，以及它与普通 Multi-Query 的区别。**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**RAG-Fusion：** 结合 Multi-Query 和 RRF（Reciprocal Rank Fusion）的增强检索方法：
1. LLM 生成多个 Query 变体
2. 对每个 Query 分别检索
3. 用 RRF 融合多路检索结果

**vs 普通 Multi-Query：**
- 普通 Multi-Query：简单取并集，所有查询结果平等对待
- RAG-Fusion：用 RRF 加权融合，在多个查询中都出现的文档排名更高

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def generate_query_variants(query: str, llm: ChatOpenAI, n: int = 4) -> list[str]:
    prompt = ChatPromptTemplate.from_template(
        "请为以下用户查询生成 {n} 个不同表达方式的查询变体，每行一个：\n{query}"
    )
    response = llm.invoke(prompt.format_messages(query=query, n=n))
    variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return [query] + variants[:n]  # 包含原始查询

def rag_fusion_search(query: str, vectorstore, llm, k=5, n_variants=4):
    # Step 1: 生成多个查询变体
    queries = generate_query_variants(query, llm, n=n_variants)

    # Step 2: 多路检索
    all_results = {}  # doc_id -> (doc, [rank1, rank2, ...])
    for q in queries:
        results = vectorstore.similarity_search(q, k=k*2)
        for rank, doc in enumerate(results, 1):
            doc_id = doc.page_content[:50]
            if doc_id not in all_results:
                all_results[doc_id] = {"doc": doc, "ranks": []}
            all_results[doc_id]["ranks"].append(rank)

    # Step 3: RRF 融合
    for doc_id, data in all_results.items():
        data["rrf_score"] = sum(1.0 / (60 + r) for r in data["ranks"])

    sorted_docs = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:k]]
```

**考察点：** RAG-Fusion 是 Multi-Query + RRF 的组合，理解其中的每个环节。

---

**Q63. 什么是"Step-Back Prompting"在 RAG 中的应用？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Step-Back Prompting：** 在检索之前，让 LLM 将具体问题"退一步"转化为更抽象、更通用的问题，用通用问题检索背景知识，再结合原始具体问题生成答案。

**示例：**
```
原始 Query: "Python 3.12 中 f-string 的新变化是什么？"

Step-Back Query: "Python f-string 语法的历史演变和设计原则是什么？"
```

**检索策略：**
```python
from langchain.prompts import ChatPromptTemplate

step_back_prompt = ChatPromptTemplate.from_template(
    "以下是一个具体问题，请将其转化为一个更通用、更基础的背景性问题，"
    "以便检索相关知识背景：\n问题：{question}\n通用问题："
)

def step_back_rag(query: str, llm, vectorstore):
    # Step 1: 生成通用背景问题
    response = llm.invoke(step_back_prompt.format_messages(question=query))
    abstract_query = response.content.strip()

    # Step 2: 用原始问题和通用问题分别检索
    specific_docs = vectorstore.similarity_search(query, k=3)
    abstract_docs = vectorstore.similarity_search(abstract_query, k=3)

    # Step 3: 合并两路检索结果
    all_docs = list({doc.page_content: doc for doc in specific_docs + abstract_docs}.values())
    return all_docs
```

**适用场景：** 高度专业的细节问题，需要先建立背景知识才能准确回答。

**考察点：** 对查询改写技术的多样性理解，不只是 HyDE 和 Multi-Query。

---

**Q64. 什么是"CRAG"（Corrective RAG）？它如何处理检索质量不佳的情况？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**CRAG（Corrective RAG，纠正式 RAG）：** 在 RAG 流程中加入自我评估机制，当检索结果质量不佳时，自动采取纠正措施。

**工作流程：**
1. 检索 Top-K 文档
2. **评分阶段**：用 LLM 或专用评估模型对每个文档评分（相关/不相关/不确定）
3. **纠正阶段**：
   - 全部相关 → 直接使用
   - 全部不相关 → 触发 Web Search 获取新知识
   - 部分相关 → 提炼相关部分 + 补充 Web Search

```python
from enum import Enum

class DocumentGrade(str, Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"

def grade_document(query: str, doc_content: str, llm) -> DocumentGrade:
    prompt = f"""评估以下文档是否与用户问题相关。
用户问题：{query}
文档内容：{doc_content[:500]}
请回答 'relevant' 或 'not_relevant'，只输出这两个词之一。"""
    response = llm.invoke(prompt)
    grade = response.content.strip().lower()
    return DocumentGrade.RELEVANT if "relevant" in grade else DocumentGrade.NOT_RELEVANT

def corrective_rag(query, vectorstore, llm, web_search_tool):
    docs = vectorstore.similarity_search(query, k=3)
    grades = [grade_document(query, doc.page_content, llm) for doc in docs]
    relevant_docs = [doc for doc, grade in zip(docs, grades) if grade == DocumentGrade.RELEVANT]

    if not relevant_docs:
        # 全部不相关，使用 Web Search 获取信息
        web_results = web_search_tool.run(query)
        return [web_results]
    return relevant_docs
```

**考察点：** 理解 CRAG 的自我纠错机制，体现对 RAG 鲁棒性的思考。

---

**Q65. 什么是"Adaptive RAG"？它如何根据问题复杂度动态选择检索策略？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**Adaptive RAG：** 根据 Query 的复杂度，动态选择以下三种策略之一：
1. **无检索（No retrieval）**：简单常识问题，直接 LLM 回答
2. **单次检索**：中等复杂度，标准 RAG
3. **迭代检索**：复杂多跳推理，多轮检索

```python
from enum import Enum

class QueryComplexity(str, Enum):
    SIMPLE = "simple"         # 常识 → 无需检索
    MODERATE = "moderate"     # 标准 → 单次检索
    COMPLEX = "complex"       # 多跳 → 迭代检索

def classify_complexity(query: str, llm) -> QueryComplexity:
    prompt = f"""分析以下查询的复杂度：
- simple：通用常识，无需外部文档
- moderate：需要检索一次文档即可回答
- complex：需要多步推理或多个文档组合
查询：{query}
只回答: simple/moderate/complex"""
    result = llm.invoke(prompt).content.strip().lower()
    if "simple" in result:
        return QueryComplexity.SIMPLE
    elif "complex" in result:
        return QueryComplexity.COMPLEX
    return QueryComplexity.MODERATE

def adaptive_rag(query, llm, vectorstore, max_iterations=3):
    complexity = classify_complexity(query, llm)

    if complexity == QueryComplexity.SIMPLE:
        return llm.invoke(query).content  # 直接回答

    elif complexity == QueryComplexity.MODERATE:
        docs = vectorstore.similarity_search(query, k=5)
        return generate_answer(query, docs, llm)  # 标准 RAG

    else:  # COMPLEX
        # 迭代检索
        context = []
        current_query = query
        for i in range(max_iterations):
            docs = vectorstore.similarity_search(current_query, k=3)
            context.extend(docs)
            # 检查是否已有足够信息
            assessment = llm.invoke(f"基于以下信息，能否回答'{query}'？\n{[d.page_content for d in context]}")
            if "sufficient" in assessment.content.lower() or "可以" in assessment.content:
                break
            # 生成后续子查询
            current_query = llm.invoke(f"为了回答'{query}'，基于已有信息，还需要查询什么？").content
        return generate_answer(query, context, llm)
```

**考察点：** Adaptive RAG 的策略路由思想，体现系统设计的灵活性。

---

**Q66. 什么是 GraphRAG？它与传统 RAG 相比有何优势？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**GraphRAG（Microsoft, 2024）：** 在知识图谱上进行 RAG，而不是在平面文档块上检索。

**构建过程：**
1. 使用 LLM 从文档中抽取实体和关系，构建知识图谱
2. 对实体/关系进行社区检测（Leiden 算法），生成社区摘要
3. 建立多级索引：实体级、社区级、文档级

**两种查询模式：**
- **Local Search**：针对具体实体的问题，在知识图谱中追踪关系
- **Global Search**：针对全局性问题（"文档的主要主题是什么"），利用社区摘要

**GraphRAG 优势：**
- 可以回答传统 RAG 无法处理的"关系型"问题（A 和 B 的关系是什么？）
- 全局问题（整个知识库的主题）回答更好
- 可以多跳推理（A → B → C）

**劣势：**
- 索引成本极高（需要大量 LLM 调用）
- 实时更新困难
- 对结构化知识效果好，对随机文本效果有限

```python
# 使用 Microsoft GraphRAG 库
# pip install graphrag
# graphrag init --root ./project
# graphrag index --root ./project
# graphrag query --root ./project --method global "文档的核心主题是什么？"
```

**考察点：** 了解 GraphRAG 的架构，能判断何时应该使用它而非普通 RAG。

---

## 10. Agentic RAG (Q67-Q73)

**Q67. 什么是 Agentic RAG？它与传统 RAG 有何本质区别？**
[难度：⭐⭐] [类型：概念]
**答：**
**传统 RAG：** 线性管道，固定流程：Query → 检索 → 生成，一次性完成。

**Agentic RAG：** 将 LLM 作为智能代理，能够自主决策何时检索、检索什么、是否需要多次检索、如何组合多个工具。

**核心区别：**

| 维度 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| 流程 | 固定线性 | 动态、循环 |
| 检索次数 | 固定1次 | 自适应（0~N次） |
| 工具使用 | 只有向量检索 | 多工具（搜索/计算/数据库） |
| 自我评估 | 无 | 有（检查答案完整性） |
| 适用场景 | 简单QA | 复杂推理、多步任务 |

**Agentic RAG 的关键能力：**
1. **反思（Reflection）**：评估检索结果是否足够，不够则再次检索
2. **规划（Planning）**：将复杂问题分解为子任务
3. **工具调用（Tool Use）**：选择调用搜索、计算、数据库等不同工具

**考察点：** 理解 Agentic RAG 是将 RAG 从"管道"升级为"代理"的根本转变。

---

**Q68. 如何用 LangGraph 实现一个带有自我反思能力的 Agentic RAG 系统？**
[难度：⭐⭐⭐] [类型：代码]
**答：**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    query: str
    documents: list
    generation: str
    attempts: int
    sufficient: bool

# 节点1：检索文档
def retrieve(state: AgentState) -> AgentState:
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {"documents": docs, "attempts": state.get("attempts", 0) + 1}

# 节点2：生成答案
def generate(state: AgentState) -> AgentState:
    context = "\n".join([d.page_content for d in state["documents"]])
    response = llm.invoke(f"基于以下上下文回答问题：\n{context}\n\n问题：{state['query']}")
    return {"generation": response.content}

# 节点3：评估答案质量
def grade_generation(state: AgentState) -> AgentState:
    grade_prompt = f"""评估以下答案是否充分回答了问题：
问题：{state['query']}
答案：{state['generation']}
如果答案完整，回复 'sufficient'；否则回复 'insufficient'。"""
    result = llm.invoke(grade_prompt).content.lower()
    return {"sufficient": "sufficient" in result}

# 路由函数
def should_continue(state: AgentState) -> str:
    if state.get("sufficient") or state.get("attempts", 0) >= 3:
        return "end"
    return "retrieve"  # 重新检索

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade", grade_generation)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")
workflow.add_conditional_edges("grade", should_continue, {"end": END, "retrieve": "retrieve"})

app = workflow.compile()
result = app.invoke({"query": "RAG 如何处理知识更新问题？", "attempts": 0})
print(result["generation"])
```

**考察点：** LangGraph 的状态机模型，以及 Agentic RAG 的循环检索逻辑。

---

**Q69. 在 Agentic RAG 中，如何设计多工具路由（Tool Routing）？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
Agentic RAG 不只有向量检索工具，还可以接入 Web 搜索、SQL 查询、计算器等。

```python
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 工具定义
tools = [
    Tool(
        name="vector_search",
        func=lambda q: vectorstore.similarity_search(q, k=5),
        description="搜索内部知识库，适合查找产品文档、技术规范等内部资料"
    ),
    Tool(
        name="web_search",
        func=lambda q: web_search_api.run(q),
        description="搜索互联网获取最新信息，适合时效性强的问题"
    ),
    StructuredTool.from_function(
        func=lambda table, conditions: sql_db.run(f"SELECT * FROM {table} WHERE {conditions}"),
        name="sql_query",
        description="查询关系型数据库，适合结构化数据统计、聚合分析"
    ),
]

# 创建带工具的 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能问答助手，根据问题选择合适的工具检索信息。"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Agent 自动决策使用哪个工具
result = agent_executor.invoke({"input": "最新版本的 Python 有哪些新特性，以及我们内部代码库用的是哪个版本？"})
```

**考察点：** 多工具 Agent 的架构设计，以及工具描述的重要性（决定 LLM 的工具选择）。

---

**Q70. 解释 "Iterative Retrieval"（迭代检索）和 "Recursive Retrieval"（递归检索）的区别。**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**迭代检索（Iterative Retrieval）：**
- 多轮检索，每轮检索的 Query 基于上一轮的结果动态生成
- 用于补充缺失信息
```
轮1: 检索"RAG架构" → 发现缺少"向量数据库"细节
轮2: 检索"向量数据库选型" → 信息充足，停止
```

**递归检索（Recursive Retrieval）：**
- 先检索高层摘要，然后根据摘要递归检索更具体的细节
- 从宏观到微观，层层深入
```
检索文档摘要 → 发现第3章相关 → 检索第3章内容 → 发现3.2节最相关 → 检索3.2节
```

**实现对比：**
```python
# 迭代检索
def iterative_retrieval(initial_query: str, max_rounds: int = 3):
    context = []
    current_query = initial_query
    for round_num in range(max_rounds):
        docs = vectorstore.similarity_search(current_query, k=3)
        context.extend(docs)
        # 评估是否需要继续
        follow_up = llm.invoke(
            f"基于问题'{initial_query}'和已有信息，还需要进一步了解什么？"
            f"如果信息已足够，回复'DONE'，否则回复下一个子问题。"
            f"已有信息：{[d.page_content[:200] for d in context]}"
        ).content
        if "DONE" in follow_up.upper():
            break
        current_query = follow_up
    return context

# 递归检索（分层索引）
def recursive_retrieval(query: str):
    # Step1: 检索文档级摘要
    summaries = summary_store.similarity_search(query, k=3)
    # Step2: 根据摘要找到最相关文档
    relevant_doc_ids = [s.metadata["doc_id"] for s in summaries]
    # Step3: 在具体文档中精细检索
    detailed_results = []
    for doc_id in relevant_doc_ids:
        chunk_results = vectorstore.similarity_search(
            query, k=2, filter={"doc_id": doc_id}
        )
        detailed_results.extend(chunk_results)
    return detailed_results
```

**考察点：** 理解不同迭代/递归检索策略适合的场景。

---

**Q71. 什么是 RAG 中的"查询分解"（Query Decomposition）？如何处理多跳推理问题？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**问题背景：** 某些问题需要多步推理，如"A公司的CEO和B公司的CTO哪个工资更高？"
这需要：1）找A公司CEO是谁 → 2）查CEO工资 → 3）找B公司CTO → 4）查CTO工资 → 5）比较

**查询分解（Query Decomposition）：**
将复杂问题拆解为多个可独立回答的子问题，逐一检索并综合答案。

```python
def decompose_query(query: str, llm) -> list[str]:
    prompt = f"""将以下复杂问题分解为多个可以独立检索回答的子问题：
复杂问题：{query}

请输出子问题列表，每行一个，格式如：
1. 子问题一
2. 子问题二"""
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    return [line.split(". ", 1)[1] for line in lines if ". " in line]

def multi_hop_rag(complex_query: str, llm, vectorstore):
    # Step 1: 分解问题
    sub_queries = decompose_query(complex_query, llm)
    print(f"分解为 {len(sub_queries)} 个子问题")

    # Step 2: 逐一检索并回答子问题
    sub_answers = []
    for sub_q in sub_queries:
        docs = vectorstore.similarity_search(sub_q, k=3)
        context = "\n".join([d.page_content for d in docs])
        answer = llm.invoke(f"基于以下信息回答：{context}\n\n问题：{sub_q}").content
        sub_answers.append({"question": sub_q, "answer": answer})

    # Step 3: 综合子答案回答原始问题
    synthesis_prompt = f"""基于以下子问题和答案，综合回答原始问题：
原始问题：{complex_query}
子问题答案：{sub_answers}"""
    final_answer = llm.invoke(synthesis_prompt).content
    return final_answer
```

**考察点：** 对多跳推理的工程实现能力，以及 Agentic RAG 的实际应用。

---

**Q72. 什么是 Self-RAG？它如何用特殊 Token 控制检索和生成过程？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Self-RAG（Asai et al. 2023）：** 训练一个特殊的 LLM，它能在生成过程中自主决定是否需要检索，并对自己的输出进行评估。

**四种特殊 Token：**
1. `[Retrieve]`：判断是否需要检索（Yes/No）
2. `[IsREL]`：评估检索文档是否相关（Relevant/Irrelevant）
3. `[IsSUP]`：评估生成内容是否有文档支撑（Fully/Partially/Not Supported）
4. `[IsUSE]`：评估输出是否对用户有用（1~5分）

**工作流程：**
```
用户问题
  ↓
LLM 决定：需要检索[Retrieve=Yes]
  ↓
检索 Top-K 文档
  ↓
对每个文档评估相关性[IsREL]
  ↓
生成答案片段
  ↓
评估支撑度[IsSUP]和有用性[IsUSE]
  ↓
选择得分最高的答案片段组合成最终答案
```

**与普通 RAG 的区别：**
- 普通 RAG：固定检索，无自我评估
- Self-RAG：按需检索 + 反思评估，只在必要时检索

**考察点：** Self-RAG 是将检索决策内化到模型本身的前沿方向，体现对最新研究的跟进。

---

**Q73. 如何防止 Agentic RAG 系统陷入无限循环？有哪些终止条件设计策略？**
[难度：⭐⭐] [类型：设计]
**答：**
Agentic RAG 中，如果检索-评估-再检索的循环没有合理的终止条件，可能导致无限循环。

**终止条件设计：**

1. **最大迭代次数**（最简单）
   ```python
   MAX_ITERATIONS = 3
   if state["iterations"] >= MAX_ITERATIONS:
       return "end"
   ```

2. **置信度阈值**
   ```python
   if state["confidence_score"] >= 0.85:
       return "end"
   ```

3. **信息重复检测**（避免重复检索）
   ```python
   def is_new_information(new_docs, existing_context):
       new_content = " ".join([d.page_content for d in new_docs])
       existing_content = " ".join(existing_context)
       overlap = len(set(new_content.split()) & set(existing_content.split()))
       overlap_ratio = overlap / max(len(new_content.split()), 1)
       return overlap_ratio < 0.7  # 新信息 > 30% 才继续
   ```

4. **LLM 自我评估**
   ```python
   def check_sufficiency(query, context, llm):
       result = llm.invoke(f"基于已有信息能否充分回答'{query}'？只回答 yes/no。\n信息：{context[:1000]}")
       return "yes" in result.content.lower()
   ```

5. **超时机制**
   ```python
   import time
   MAX_TIME = 30  # 30秒
   start_time = time.time()
   if time.time() - start_time > MAX_TIME:
       return "timeout_end"
   ```

**最佳实践：** 结合多种终止条件，优先级：超时 > 最大迭代次数 > 信息重复 > 置信度达标

**考察点：** 生产系统的稳定性设计，避免代价高昂的无限循环。

---

## 11. RAGAS评测框架 (Q74-Q80)

**Q74. 介绍 RAGAS 框架，它评测哪几个核心维度？**
[难度：⭐⭐] [类型：概念]
**答：**
**RAGAS（RAG Assessment，2023）：** 专为 RAG 系统设计的自动化评测框架，使用 LLM 作为评判官，评测无需人工标注。

**核心评测维度：**

| 指标 | 英文名 | 含义 | 输入 |
|------|--------|------|------|
| **忠实度** | Faithfulness | 答案是否完全基于检索到的上下文（防幻觉） | 答案 + 上下文 |
| **答案相关性** | Answer Relevancy | 答案是否切实回答了用户问题 | 问题 + 答案 |
| **上下文精确率** | Context Precision | 检索到的上下文中有多少是真正相关的 | 问题 + 上下文 + 真实答案 |
| **上下文召回率** | Context Recall | 真实答案所需信息有多少被检索到了 | 上下文 + 真实答案 |

**快速上手：**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 构建评测数据集
data = {
    "question": ["什么是 RAG？", "HNSW 的 M 参数有什么作用？"],
    "answer": ["RAG 是检索增强生成技术...", "M 参数控制每个节点的连接数..."],
    "contexts": [["RAG 通过检索外部文档增强 LLM..."], ["HNSW 的 M 参数决定了图的稠密程度..."]],
    "ground_truth": ["RAG 全称 Retrieval-Augmented Generation...", "M 越大图越稠密，召回率越高..."]
}

dataset = Dataset.from_dict(data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
# {'faithfulness': 0.89, 'answer_relevancy': 0.92, 'context_precision': 0.85, 'context_recall': 0.88}
```

**考察点：** 熟悉 RAGAS 的核心指标，能解释每个指标评测的具体方面。

---

**Q75. 深入解释 RAGAS 的 Faithfulness（忠实度）指标，它是如何计算的？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Faithfulness 衡量：** 答案中的每个陈述是否都可以从检索到的上下文中推断出来（是否有文档依据）。

**计算步骤：**
1. 将答案分解为原子陈述（S1, S2, S3...）
2. 对每个陈述，判断是否可以从上下文推断（0 或 1）
3. `Faithfulness = 可从上下文推断的陈述数 / 总陈述数`

**示例：**
```
答案："RAG 由检索和生成两部分组成。它可以实时获取最新信息，并且成本比微调低。"

分解为原子陈述：
S1: "RAG 由检索和生成两部分组成" → 上下文支持 ✓
S2: "它可以实时获取最新信息"    → 上下文支持 ✓
S3: "成本比微调低"              → 上下文中未提及 ✗（幻觉！）

Faithfulness = 2/3 = 0.67
```

**使用 LLM 计算 Faithfulness：**
```python
from ragas.metrics import faithfulness
from ragas.metrics._faithfulness import _statements_from_answer

# 内部流程（简化版）
def compute_faithfulness(answer, contexts, llm):
    # Step 1: 将答案分解为原子陈述
    statements = llm.invoke(f"将以下答案分解为原子陈述：{answer}").content

    # Step 2: 逐一验证是否在上下文中有依据
    verified_count = 0
    for stmt in statements.split("\n"):
        verdict = llm.invoke(
            f"以下陈述是否可以从上下文中推断出来？\n"
            f"陈述：{stmt}\n上下文：{contexts}\n只回答 yes 或 no。"
        ).content
        if "yes" in verdict.lower():
            verified_count += 1

    return verified_count / max(len(statements.split("\n")), 1)
```

**考察点：** 深入理解 Faithfulness 的计算方式，能解释为什么它是幻觉检测的核心指标。

---

**Q76. RAGAS 的 Context Recall 和 Context Precision 有何区别？**
[难度：⭐⭐] [类型：对比]
**答：**
这两个指标都评测检索质量，但角度不同：

**Context Precision（上下文精确率）：**
- 问：检索到的文档中，有多少是真正有用的（有信号的）？
- 公式：`有用的上下文块 / 所有检索到的上下文块`
- 反映：Noise 有多少，检索是否精确

**Context Recall（上下文召回率）：**
- 问：回答问题所需的所有信息，有多少被检索到了？
- 公式：基于 Ground Truth 答案，评估其中的陈述有多少可以在检索上下文中找到
- 反映：有没有遗漏关键信息

**示例对比：**
```
查询："RAG 有哪些核心组件？"

检索到的文档：
Doc1: "RAG 包含检索器和生成器" ← 相关
Doc2: "向量数据库是常见的存储方式" ← 相关
Doc3: "LangChain 是一个开发框架" ← 不相关
Doc4: "检索器会返回Top-K文档" ← 相关

Context Precision = 3/4 = 0.75（3个相关，1个不相关）

如果 Ground Truth 答案还提到了"Embedding模型"和"Chunking"，
但这两个都没有被检索到：
Context Recall = 0.5（检索到了2个关键点，漏掉了2个）
```

**考察点：** 区分精确率（噪声）和召回率（遗漏）的视角差异。

---

**Q77. 如何构建一个 RAG 系统的自动化评测流水线？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
```python
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI

class RAGEvaluationPipeline:
    def __init__(self, rag_chain, test_dataset_path: str):
        self.rag_chain = rag_chain
        self.test_data = pd.read_json(test_dataset_path)

    def generate_answers(self) -> pd.DataFrame:
        """对所有测试问题生成答案"""
        results = []
        for _, row in self.test_data.iterrows():
            query = row["question"]
            # 获取答案和检索上下文
            response = self.rag_chain.invoke({"input": query})
            results.append({
                "question": query,
                "answer": response["answer"],
                "contexts": [doc.page_content for doc in response["source_documents"]],
                "ground_truth": row.get("ground_truth", "")
            })
        return pd.DataFrame(results)

    def evaluate(self) -> dict:
        """运行 RAGAS 评测"""
        df = self.generate_answers()
        dataset = Dataset.from_pandas(df)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model="gpt-4o-mini"),
            raise_exceptions=False  # 遇到单个错误不中断
        )
        return result

    def regression_check(self, current_scores: dict, baseline_scores: dict,
                          threshold: float = 0.05) -> bool:
        """检测是否有性能退化（CI/CD 中使用）"""
        for metric, score in current_scores.items():
            baseline = baseline_scores.get(metric, 0)
            if score < baseline - threshold:
                print(f"REGRESSION: {metric} 下降 {baseline-score:.3f} (from {baseline:.3f} to {score:.3f})")
                return False
        return True

# 使用
pipeline = RAGEvaluationPipeline(rag_chain, "test_data.json")
scores = pipeline.evaluate()
print(f"Faithfulness: {scores['faithfulness']:.3f}")
print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
```

**考察点：** 自动化评测流水线的工程设计，以及 CI/CD 中的回归测试思维。

---

**Q78. 除了 RAGAS，还有哪些 RAG 评测框架和方法？**
[难度：⭐⭐] [类型：概念]
**答：**

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **RAGAS** | 无参考答案可运行，LLM 评判 | 通用 RAG 评测 |
| **TruLens** | 监控+评测一体，可视化好 | 生产监控 |
| **ARES** | 训练专用评估模型，成本低 | 大规模低成本评测 |
| **LlamaIndex Evaluators** | 集成在 LlamaIndex 框架内 | 使用 LlamaIndex 的项目 |
| **Deepeval** | 覆盖面广，支持多种指标 | 全栈 LLM 测试 |

**非 LLM-as-Judge 的传统指标：**
```python
# ROUGE（词汇重叠）
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_answer, generated_answer)

# BERTScore（语义相似度）
from bert_score import score
P, R, F1 = score([generated], [reference], lang="zh")
```

**考察点：** 对评测生态的全面了解，不只局限于 RAGAS。

---

**Q79. 如何设计 RAG 系统的"黄金测试集"？收集标注数据有哪些方法？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**黄金测试集（Golden Test Set）：** 包含问题、真实相关文档 ID 和参考答案的高质量评测数据集。

**数据收集方法：**

1. **人工标注（最高质量）**
   - 领域专家编写问题和答案
   - 成本高，适合关键场景

2. **用户行为挖掘**
   - 从历史日志中提取真实用户问题
   - 标注其中的高质量 QA 对

3. **LLM 合成（成本最低）**
   ```python
   def generate_test_qa(document_chunks, llm, n_per_chunk=3):
       qa_pairs = []
       for chunk in document_chunks[:50]:  # 随机抽样
           prompt = f"""基于以下文档片段，生成 {n_per_chunk} 对问题和答案：
文档：{chunk.page_content}
格式：
Q: 问题
A: 答案（50字以内）"""
           response = llm.invoke(prompt)
           # 解析 Q-A 对
           for line in response.content.split("\n"):
               if line.startswith("Q:"):
                   qa_pairs.append({"question": line[2:].strip(), "source_chunk": chunk.metadata["id"]})
       return qa_pairs
   ```

4. **对抗性测试集**：专门包含边界案例
   - 知识库中不存在答案的问题（测试"不知道"能力）
   - 具有迷惑性的近似相关文档

**测试集质量评估：**
- 覆盖性：覆盖知识库的主要主题
- 多样性：不同难度、不同问题类型
- 准确性：由领域专家验证

**考察点：** 测试集设计的完整方法论，不只是说"用 LLM 生成"。

---

**Q80. 如何用 RAGAS 进行 A/B 测试，比较不同 RAG 配置的效果？**
[难度：⭐⭐] [类型：设计]
**答：**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
import pandas as pd

def ragas_ab_test(
    test_questions: list,
    ground_truths: list,
    rag_config_a,
    rag_config_b,
    config_names: tuple = ("Config A", "Config B")
) -> pd.DataFrame:
    results = {}
    for name, config in zip(config_names, [rag_config_a, rag_config_b]):
        # 运行每种配置
        answers, contexts = [], []
        for q in test_questions:
            resp = config.invoke(q)
            answers.append(resp["answer"])
            contexts.append([d.page_content for d in resp["source_docs"]])

        dataset = Dataset.from_dict({
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        results[name] = dict(scores)

    # 对比展示
    df = pd.DataFrame(results).T
    print("\nA/B Test Results:")
    print(df.to_string())
    winner = df.mean(axis=1).idxmax()
    print(f"\nWinner: {winner}")
    return df

# 示例：比较 chunk_size=500 vs chunk_size=1000
ab_results = ragas_ab_test(
    test_questions=test_qs,
    ground_truths=ground_truths,
    rag_config_a=rag_chain_chunk500,
    rag_config_b=rag_chain_chunk1000,
    config_names=("chunk_500", "chunk_1000")
)
```

**考察点：** 科学的 A/B 测试方法，能用数据驱动决策而非直觉。

---

## 12. 生产级RAG设计 (Q81-Q87)

**Q81. 生产级 RAG 系统需要哪些关键的可观测性（Observability）组件？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**可观测性三支柱在 RAG 中的应用：**

**1. 日志（Logging）：**
```python
import logging
from datetime import datetime

class RAGLogger:
    def log_retrieval(self, query, retrieved_docs, latency_ms):
        logging.info({
            "event": "retrieval",
            "query": query,
            "doc_count": len(retrieved_docs),
            "top_score": retrieved_docs[0].score if retrieved_docs else None,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat()
        })

    def log_generation(self, query, answer, faithfulness_score, latency_ms):
        logging.info({
            "event": "generation",
            "query": query[:100],
            "answer_length": len(answer),
            "faithfulness": faithfulness_score,
            "latency_ms": latency_ms
        })
```

**2. 指标（Metrics）：**
- 检索延迟（p50/p95/p99）
- 答案生成延迟
- 检索相似度分数分布
- 空召回率（无结果的 Query 比例）
- 实时 Faithfulness 评分

**3. 追踪（Tracing）：**
```python
# 使用 LangSmith 追踪整个 RAG 链路
from langchain_core.tracers.langchain import wait_for_all_tracers
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
```

**核心监控告警：**
- 检索空结果率 > 10% → 告警（知识库可能未更新）
- Faithfulness < 0.7 → 告警（幻觉风险）
- 端到端延迟 > 5s → 告警
- 用户负反馈率突增 → 告警

**考察点：** 生产系统的运维意识，从代码到监控的完整思路。

---

**Q82. 在高并发场景下，如何对 RAG 系统进行性能优化？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**RAG 性能优化分层策略：**

**1. 缓存层**
```python
import redis
import hashlib
import json

redis_client = redis.Redis(host="localhost", port=6379)

def cached_rag(query: str, rag_chain, cache_ttl=3600):
    # 对相同 Query 缓存结果
    cache_key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    result = rag_chain.invoke({"input": query})
    redis_client.setex(cache_key, cache_ttl, json.dumps(result))
    return result
```

**2. Embedding 批处理**
```python
# 批量计算 Embedding 而非逐条处理
def batch_embed(texts: list, model, batch_size=64) -> list:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        all_embeddings.extend(embeddings)
    return all_embeddings
```

**3. 异步处理**
```python
import asyncio
from langchain_openai import ChatOpenAI

async def async_rag(queries: list[str], rag_chain) -> list:
    tasks = [rag_chain.ainvoke({"input": q}) for q in queries]
    return await asyncio.gather(*tasks)
```

**4. 向量检索优化**
- 使用 HNSW（比精确搜索快 100x+）
- 调整 ef_search 参数（速度/精度权衡）
- GPU 加速 Embedding 计算

**5. LLM 优化**
- 流式输出（Streaming）：首字节时间减少
- 使用更小的模型（gpt-4o-mini 而非 gpt-4）
- Prompt 缓存（Anthropic Claude API 支持 Prompt Caching）

**考察点：** 多层次的性能优化思维，从缓存到并发的全链路优化。

---

**Q83. 如何设计 RAG 系统的"知识更新"（Incremental Indexing）流水线？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**增量更新的挑战：**
- 如何检测文档是否已更新？
- 如何只更新变化的部分，而不是全量重建？
- 如何处理文档删除？

**增量索引方案：**
```python
import hashlib
from datetime import datetime

class IncrementalIndexer:
    def __init__(self, vectorstore, doc_registry_db):
        self.vectorstore = vectorstore
        self.registry = doc_registry_db  # 记录文档版本信息

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def upsert_document(self, doc_id: str, content: str, metadata: dict):
        """增量更新单个文档"""
        new_hash = self._compute_hash(content)
        existing = self.registry.get(doc_id)

        if existing and existing["hash"] == new_hash:
            return "no_change"  # 内容未变，跳过

        if existing:
            # 删除旧 chunk 的向量
            old_chunk_ids = existing["chunk_ids"]
            self.vectorstore.delete(ids=old_chunk_ids)

        # 重新分块和索引
        chunks = self.chunker.split(content)
        new_chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.vectorstore.add_texts(
            texts=[c.page_content for c in chunks],
            ids=new_chunk_ids,
            metadatas=[{**metadata, "doc_id": doc_id} for _ in chunks]
        )

        # 更新注册表
        self.registry.set(doc_id, {
            "hash": new_hash,
            "chunk_ids": new_chunk_ids,
            "updated_at": datetime.utcnow().isoformat()
        })
        return "updated"

    def delete_document(self, doc_id: str):
        existing = self.registry.get(doc_id)
        if existing:
            self.vectorstore.delete(ids=existing["chunk_ids"])
            self.registry.delete(doc_id)
```

**触发机制：**
- Webhook：文档系统变更时主动推送
- 定时轮询：每天检查文档更新
- 事件驱动：通过消息队列（Kafka/RabbitMQ）异步处理

**考察点：** 生产知识库的全生命周期管理，不只是一次性建索引。

---

**Q84. RAG 系统的"安全性"（Security）需要考虑哪些方面？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**RAG 特有的安全威胁：**

1. **Prompt Injection（提示词注入）**
   - 攻击者在文档中嵌入恶意指令，当文档被检索后注入到 Prompt 中
   - 防御：输入净化、将检索内容放在严格分隔的位置

2. **数据泄露（Data Exfiltration）**
   - 通过精心设计的 Query，诱导检索到敏感文档
   - 防御：文档密级标注 + 用户权限过滤

3. **知识库投毒（Data Poisoning）**
   - 攻击者向知识库注入错误/恶意信息
   - 防御：文档来源验证、内容审核流水线

**安全设计原则：**
```python
# 1. 基于角色的文档访问控制
def secure_retrieve(query, user_roles, vectorstore):
    # 只检索用户有权限访问的文档
    results = vectorstore.similarity_search(
        query,
        filter={"allowed_roles": {"$in": user_roles}}
    )
    return results

# 2. 防止 Prompt Injection
def sanitize_context(retrieved_docs):
    sanitized = []
    for doc in retrieved_docs:
        # 去除可能的指令字符
        content = doc.page_content
        content = content.replace("Ignore previous instructions", "")
        content = content.replace("System:", "")
        sanitized.append(content)
    return sanitized

# 3. 敏感信息脱敏（PII）
import re
def mask_pii(text: str) -> str:
    # 手机号脱敏
    text = re.sub(r'1[3-9]\d{9}', '[手机号]', text)
    # 身份证脱敏
    text = re.sub(r'\d{17}[\dX]', '[身份证]', text)
    return text
```

**考察点：** RAG 安全是生产落地的重要考量，体现对攻击向量的认知。

---

**Q85. 如何设计 RAG 系统的"回退策略"（Fallback Strategy）？**
[难度：⭐⭐] [类型：设计]
**答：**
当 RAG 系统无法提供高质量答案时，需要合理的回退机制。

**回退触发条件：**
1. 检索结果相似度低于阈值（无相关文档）
2. LLM 生成答案的置信度低
3. Faithfulness 评分过低（幻觉风险）
4. 检索服务超时

**分层回退策略：**
```python
class RAGWithFallback:
    def __init__(self, vectorstore, web_search_tool, llm):
        self.vectorstore = vectorstore
        self.web_search = web_search_tool
        self.llm = llm

    def answer(self, query: str) -> dict:
        # Level 1: 尝试本地知识库
        docs = self.vectorstore.similarity_search_with_score(query, k=5)
        top_score = docs[0][1] if docs else 0

        if top_score >= 0.75:
            # 本地知识库质量足够
            return {"answer": self._generate(query, [d[0] for d in docs]), "source": "local_kb"}

        # Level 2: 本地质量不足，尝试 Web 搜索补充
        web_results = self.web_search.run(query)
        if web_results:
            all_context = [d[0] for d in docs] + [web_results]
            return {"answer": self._generate(query, all_context), "source": "hybrid"}

        # Level 3: 完全回退，使用 LLM 直接回答，并标注不确定性
        direct_answer = self.llm.invoke(
            f"请尽可能回答以下问题，如果不确定请明确说明：{query}"
        ).content
        return {"answer": direct_answer, "source": "llm_direct", "warning": "可能不准确"}

    def _generate(self, query, docs):
        context = "\n".join([d.page_content for d in docs])
        return self.llm.invoke(f"基于上下文：{context}\n\n回答：{query}").content
```

**考察点：** 对系统鲁棒性的设计，理解优雅降级的重要性。

---

**Q86. RAG 系统的延迟预算如何分配？各组件的典型延迟是多少？**
[难度：⭐⭐] [类型：设计]
**答：**
**典型 RAG 系统延迟分解（目标总体 < 3s）：**

| 组件 | 典型延迟 | 优化目标 |
|------|---------|---------|
| Query Embedding | 20~50ms | 缓存/批处理 |
| 向量检索（HNSW） | 10~50ms | 减小 ef_search |
| BM25 检索 | 5~20ms | - |
| Re-ranking（CPU） | 200~500ms | GPU 加速 / 减少候选数 |
| LLM 生成（GPT-4o-mini） | 500~1500ms | 流式输出 |
| **总计** | **800~2200ms** | |

**延迟优化的优先级：**
1. LLM 生成通常占 50~70% 的时间，用流式输出改善用户感知
2. Re-ranking 可以用轻量模型（FlashRank）替代
3. Embedding 计算可以批处理或缓存

```python
import time

class TimedRAGChain:
    def invoke(self, query):
        timings = {}

        t0 = time.perf_counter()
        query_embedding = self.embed_query(query)
        timings["embedding"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        candidates = self.vectorstore.search(query_embedding, k=20)
        timings["retrieval"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates[:20])
        timings["reranking"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        answer = self.llm.invoke(self.build_prompt(query, reranked[:5]))
        timings["generation"] = (time.perf_counter() - t3) * 1000

        timings["total"] = sum(timings.values())
        return {"answer": answer, "timings": timings}
```

**考察点：** 端到端延迟的定量分析能力，以及有针对性的优化思路。

---

**Q87. 如何在 RAG 系统中实现多租户隔离和数据权限控制？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**多租户隔离策略：**

**方案一：Collection 隔离（强隔离）**
```python
def get_tenant_collection(tenant_id: str, client):
    return client.get_or_create_collection(
        name=f"tenant_{tenant_id}",
        metadata={"tenant_id": tenant_id}
    )

# 查询时只访问本租户的 Collection
def tenant_search(query, tenant_id, client, k=5):
    collection = get_tenant_collection(tenant_id, client)
    return collection.query(query_texts=[query], n_results=k)
```

**方案二：元数据过滤（软隔离，适合共享索引）**
```python
def permission_aware_search(query, user_id, user_roles, vectorstore, k=5):
    # 过滤条件：用户 ID 匹配 OR 角色匹配
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={
            "$or": [
                {"owner_id": {"$eq": user_id}},
                {"visible_roles": {"$in": user_roles}},
                {"visibility": {"$eq": "public"}}
            ]
        }
    )
    return results
```

**文档权限模型：**
```python
@dataclass
class DocumentPermission:
    doc_id: str
    owner_id: str
    visibility: str  # "public", "team", "private"
    visible_roles: list  # ["admin", "finance", "hr"]
    visible_users: list  # 具体用户白名单
```

**审计日志：**
每次检索操作记录：用户 ID、时间戳、查询内容、检索到的文档 ID，用于安全审计。

**考察点：** 企业级 RAG 的权限设计，体现对数据安全的系统性思考。

---

## 13. RAG中的幻觉处理 (Q88-Q94)

**Q88. 在 RAG 中，如何通过 Prompt Engineering 减少幻觉？**
[难度：⭐⭐] [类型：设计]
**答：**
**核心 Prompt 设计原则：**

1. **明确指示只使用上下文回答：**
```python
RAG_PROMPT = """你是一个专业的问答助手。
请严格根据以下提供的上下文信息回答用户问题。
如果上下文中没有足够的信息回答问题，请直接说"根据现有资料，我无法回答该问题"，不要推测或使用外部知识。

上下文：
{context}

用户问题：{question}

回答要求：
1. 只使用上下文中明确提供的信息
2. 如果信息不足，明确说明缺少什么信息
3. 回答后标注信息来源（如：根据文档第X段）"""
```

2. **引导模型自我检查：**
```python
SELF_CHECK_PROMPT = """{answer}

请验证以上答案中的每个事实陈述是否都能在以下上下文中找到依据：
{context}
对于不确定的陈述，用[待验证]标注。"""
```

3. **Chain-of-Thought 引导准确引用：**
```python
COT_PROMPT = """根据以下上下文回答问题：
{context}

问题：{question}

请按以下步骤回答：
1. 首先找出上下文中与问题相关的关键信息
2. 列出这些关键信息（逐条引用）
3. 基于这些信息综合给出答案"""
```

**考察点：** Prompt Engineering 对幻觉控制的具体方法，有代码示例。

---

**Q89. 如何实时检测 RAG 生成答案中的幻觉？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**在线幻觉检测方案：**

**方案一：NLI（自然语言推理）检测**
```python
from transformers import pipeline

# 用 NLI 模型检测答案是否能从上下文推断
nli_model = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def check_faithfulness(answer: str, context: str) -> float:
    # 将上下文作为前提，答案的每个句子作为假设
    import re
    sentences = re.split(r'[。！？.!?]', answer)
    supported_count = 0

    for sent in sentences:
        if not sent.strip():
            continue
        result = nli_model(f"{context[:512]} [SEP] {sent}")
        if result[0]["label"] == "ENTAILMENT" and result[0]["score"] > 0.8:
            supported_count += 1

    return supported_count / max(len([s for s in sentences if s.strip()]), 1)
```

**方案二：LLM 自我核查（SelfCheckGPT 思想）**
```python
def self_check_hallucination(query, answer, context, llm, n_samples=3):
    """生成多个答案变体，如果不一致则可能存在幻觉"""
    variants = []
    for _ in range(n_samples):
        variant = llm.invoke(
            f"基于以下上下文回答问题（每次可以用不同表达）：\n{context}\n问题：{query}"
        ).content
        variants.append(variant)

    # 检查原答案与变体的一致性
    consistency_prompt = f"""以下是对同一问题的多个回答，判断原始答案中哪些陈述在所有变体中都一致（=可信），哪些不一致（=可能幻觉）：
原始答案：{answer}
变体答案：{variants}"""
    return llm.invoke(consistency_prompt).content
```

**方案三：RAG 专用 Faithfulness 评分**
```python
from ragas.metrics import faithfulness
# 实时计算 Faithfulness（需要几次 LLM 调用，约 300~500ms）
```

**考察点：** 在线幻觉检测的多种方法，及各自的速度/精度权衡。

---

**Q90. 当 RAG 系统检测到问题不在知识库范围内时，应该如何处理？**
[难度：⭐⭐] [类型：场景]
**答：**
**"不知道"是一种重要能力。** 错误地回答比承认不知道更有害。

**处理策略：**

```python
class RobustRAGChain:
    CONFIDENCE_THRESHOLD = 0.7

    def answer(self, query: str) -> dict:
        # 1. 检索
        results = self.vectorstore.similarity_search_with_score(query, k=5)

        if not results:
            return self._no_info_response(query)

        top_score = results[0][1]  # 相似度分数

        # 2. 低相似度：直接拒绝
        if top_score < 0.5:
            return self._no_info_response(query)

        # 3. 中等相似度：谨慎回答
        docs = [r[0] for r in results]
        answer = self.llm.invoke(self.build_prompt(query, docs)).content

        # 4. 生成后验证（Faithfulness 检查）
        faithfulness_score = self._check_faithfulness(answer, docs)
        if faithfulness_score < self.CONFIDENCE_THRESHOLD:
            return {
                "answer": answer,
                "warning": "此回答可能包含不确定信息，请谨慎参考",
                "confidence": faithfulness_score,
                "sources": [d.metadata.get("source") for d in docs]
            }

        return {
            "answer": answer,
            "confidence": faithfulness_score,
            "sources": [d.metadata.get("source") for d in docs]
        }

    def _no_info_response(self, query: str) -> dict:
        return {
            "answer": f"很抱歉，关于'{query[:30]}...'，我在现有知识库中没有找到相关信息。建议您查阅官方文档或联系专业人员。",
            "confidence": 0.0,
            "sources": []
        }
```

**考察点：** 构建"诚实"的 RAG 系统，处理 Out-of-Scope 问题的工程实践。

---

**Q91. 如何处理 RAG 中的"时间敏感性"问题（outdated information）？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
知识库中的信息可能已过时，LLM 需要能够识别并提醒用户。

**解决方案：**

1. **文档时间戳元数据 + 提示**
```python
def build_time_aware_prompt(query, docs, current_date):
    context_with_dates = []
    for doc in docs:
        doc_date = doc.metadata.get("updated_at", "未知日期")
        context_with_dates.append(f"[文档日期: {doc_date}]\n{doc.page_content}")

    return f"""当前日期：{current_date}

请注意文档的发布日期，对于过期信息要明确提醒用户。

上下文：
{chr(10).join(context_with_dates)}

问题：{query}

如果文档信息可能已过时（超过1年），请在答案末尾注明"注：此信息来自[日期]，请验证是否仍然有效"。"""
```

2. **时间敏感性分类器**
```python
def is_time_sensitive(query: str, llm) -> bool:
    result = llm.invoke(
        f"以下问题是否对时效性敏感（如价格、政策、版本等）？只回答 yes/no：\n{query}"
    ).content.lower()
    return "yes" in result

# 对时效性问题，优先使用 Web Search 而非本地知识库
if is_time_sensitive(query, llm):
    results = web_search.run(query)
else:
    results = vectorstore.similarity_search(query, k=5)
```

3. **知识截止日期声明**
```python
SYSTEM_PROMPT = """你是一个 RAG 系统，知识库最后更新于 {knowledge_cutoff}。
对于可能发生变化的信息（价格、政策、规范等），请主动提醒用户验证最新信息。"""
```

**考察点：** 对 RAG 时效性局限的认知和解决方案。

---

**Q92. 什么是 RAG 中的"幻觉级联"（Hallucination Cascade）问题？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**幻觉级联：** 在多轮对话或多步推理的 RAG 中，第一步的幻觉会被当作"事实"用于后续步骤，导致错误不断累积、放大。

**示例：**
```
用户：A公司2024年的营收是多少？
RAG第一步：检索到旧文档，生成"A公司营收30亿"（实为过时数据）

用户：这比竞争对手B公司多多少？
RAG第二步：基于"A公司营收30亿"（幻觉数据）+ 检索到B公司数据，给出错误比较

用户：请分析A公司的利润率...
RAG第三步：继续基于错误数据，分析完全失真
```

**防止幻觉级联的策略：**

1. **每轮验证：** 每次生成后都进行 Faithfulness 检查
2. **信息追踪：** 维护"已确认事实"列表，标注每个事实的来源
3. **不信任历史答案：** 重新检索而不是依赖对话历史中的数字
4. **引用标注：** 强制每个关键数字都带来源标注

```python
def multi_turn_rag_with_verification(conversation_history, new_query, vectorstore, llm):
    # 不信任历史答案中的具体数字，重新检索验证
    if any(char.isdigit() for char in new_query):
        fresh_docs = vectorstore.similarity_search(new_query, k=5)
        # 从新文档生成，而不是依赖历史答案
    else:
        # 一般问题可以利用对话历史
        pass
```

**考察点：** 对多轮 RAG 系统中错误传播机制的理解，体现对系统性风险的认知。

---

**Q93. 在 RAG 中，如何让模型正确处理矛盾信息（检索到的文档观点相互矛盾）？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
知识库中可能存在来自不同时间、不同来源的矛盾信息。

**矛盾检测与处理策略：**

```python
def handle_contradictory_context(query, docs, llm):
    # Step 1: 检测矛盾
    context_texts = [d.page_content for d in docs]
    contradiction_check = llm.invoke(f"""以下文档片段中是否存在相互矛盾的信息？
文档：{context_texts}
如果有矛盾，请指出哪些陈述相互矛盾。只回答 yes/no 和矛盾点。""").content

    if "yes" in contradiction_check.lower():
        # Step 2: 处理矛盾 - 按时间排序，优先最新
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("updated_at", ""), reverse=True)
        contradiction_prompt = f"""以下信息中存在矛盾（{contradiction_check}）。
请按照以下原则回答问题：
1. 优先使用更新的信息（文档日期：{[d.metadata.get('updated_at') for d in docs]}）
2. 明确指出存在矛盾
3. 列出不同来源的不同说法

问题：{query}
信息：{[d.page_content for d in docs_sorted]}"""
        return llm.invoke(contradiction_prompt).content
    else:
        # 无矛盾，正常回答
        return standard_rag_answer(query, docs, llm)
```

**用户提示示例：**
```
答案："关于X的说法，不同来源存在分歧：
- 2023年版文档指出：...
- 2024年最新文档指出：...
建议以最新版本为准。"
```

**考察点：** 对知识库中矛盾信息的处理能力，体现对数据质量问题的认识。

---

**Q94. 如何评估 RAG 系统的"拒绝回答"（Abstention）行为是否合理？**
[难度：⭐⭐] [类型：设计]
**答：**
**拒绝回答的两类错误：**
1. **误拒**（False Negative）：本应能回答的问题被拒绝 → 降低用户体验
2. **误答**（False Positive）：本应拒绝的问题给出了幻觉回答 → 传播错误信息

**评估指标：**
```python
def evaluate_abstention(test_set, rag_system):
    """
    test_set 包含：
    - answerable_questions: 知识库中有答案的问题
    - unanswerable_questions: 知识库中没有答案的问题
    """
    tp = fp = tn = fn = 0

    for q in test_set["answerable_questions"]:
        result = rag_system.answer(q)
        if result["confidence"] > 0 and len(result["answer"]) > 20:
            tp += 1  # 正确回答
        else:
            fn += 1  # 误拒

    for q in test_set["unanswerable_questions"]:
        result = rag_system.answer(q)
        if "无法回答" in result["answer"] or result["confidence"] < 0.3:
            tn += 1  # 正确拒绝
        else:
            fp += 1  # 误答（幻觉）

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    abstention_precision = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"回答精确率: {precision:.2f}")
    print(f"回答召回率: {recall:.2f}")
    print(f"拒绝精确率: {abstention_precision:.2f}")
```

**调整策略：**
- 误拒太多：降低相似度阈值 / 放宽 Faithfulness 要求
- 误答太多：提高阈值 / 加强 Prompt 中的"不确定则拒绝"指引

**考察点：** 对 RAG 系统"诚实性"的定量评估，以及阈值调优思路。

---

## 14. RAG vs Fine-tuning决策 (Q95-Q100)

**Q95. RAG 和 Fine-tuning（微调）有什么本质区别？各自适合什么场景？**
[难度：⭐⭐] [类型：概念]
**答：**

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识注入方式** | 动态检索，推理时注入 | 静态训练，参数化记忆 |
| **更新成本** | 低（重新索引文档） | 高（重新训练）|
| **知识边界** | 清晰（来源可追溯） | 模糊（参数中分散存储）|
| **幻觉风险** | 较低（有上下文约束） | 较高（参数记忆可能失真）|
| **推理能力** | 受上下文窗口限制 | 可以习得新的推理模式 |
| **成本** | 推理成本高（检索+LLM） | 训练成本高，推理成本低 |
| **隐私** | 数据不进入模型 | 数据参数化（可能隐私泄露）|

**RAG 适合场景：**
- 知识需要频繁更新（新闻、产品文档、政策）
- 需要追溯信息来源（合规、审计）
- 知识量大，超出 token 限制
- 私有/敏感数据

**Fine-tuning 适合场景：**
- 学习特定格式/风格（公司公文格式）
- 习得专业领域的推理模式（医学诊断逻辑）
- 知识相对稳定不变
- 需要低延迟（无检索步骤）

**考察点：** 理解两者在知识注入机制上的根本差异，能给出有依据的选型建议。

---

**Q96. RAG 和 Fine-tuning 可以结合使用吗？如何设计 RAG + Fine-tuning 的混合方案？**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**RAG + Fine-tuning 混合方案（互补优化）：**

**策略1：Fine-tune 基础能力，RAG 注入知识**
- 先用领域数据 Fine-tune 模型，让其学会领域的"语言风格"和"推理模式"
- 再用 RAG 动态注入最新的事实知识
- 例：医疗问答 = 医学语言微调 + 最新治疗指南 RAG

**策略2：Fine-tune Embedding 模型提升检索质量**
- 对 Embedding 模型做领域微调（contrastive learning）
- 提升 RAG 检索阶段的质量

**策略3：Fine-tune 生成指令遵循**
- 微调模型学会 "只根据上下文回答，不确定时拒绝回答"
- 减少 Prompt Engineering 的工作量，提升系统鲁棒性

```python
# 结合 RAG + Fine-tuned 模型的架构示例
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载领域微调模型
model = AutoModelForCausalLM.from_pretrained("my_company/domain_finetuned_llm")
tokenizer = AutoTokenizer.from_pretrained("my_company/domain_finetuned_llm")

# 在微调模型上应用 RAG
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# 基于微调 LLM 构建 RAG 链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

**考察点：** 理解 RAG 和 Fine-tuning 的互补性，能设计混合方案。

---

**Q97. 如何量化评估 RAG 是否优于 Fine-tuning？设计一个对比实验框架。**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**对比实验设计：**

```python
class RAGvsFinetuneExperiment:
    def __init__(self, test_dataset, rag_chain, finetuned_chain, base_llm_chain):
        self.test_data = test_dataset
        self.systems = {
            "RAG": rag_chain,
            "Fine-tuned": finetuned_chain,
            "Base LLM": base_llm_chain
        }

    def run_experiment(self) -> dict:
        results = {}
        for system_name, chain in self.systems.items():
            scores = {"accuracy": [], "faithfulness": [], "latency_ms": [], "cost": []}
            for sample in self.test_data:
                import time
                t0 = time.time()
                answer = chain.invoke(sample["question"])
                latency = (time.time() - t0) * 1000

                # 评估
                acc = self._evaluate_accuracy(answer, sample["ground_truth"])
                faith = self._evaluate_faithfulness(answer, sample.get("context", ""))
                cost = self._estimate_cost(system_name, sample["question"], answer)

                scores["accuracy"].append(acc)
                scores["faithfulness"].append(faith)
                scores["latency_ms"].append(latency)
                scores["cost"].append(cost)

            results[system_name] = {k: sum(v)/len(v) for k, v in scores.items()}
        return results

    def _evaluate_accuracy(self, answer, ground_truth):
        # 用 LLM 评估答案准确性 (0~1)
        pass

    def _estimate_cost(self, system_name, query, answer):
        if system_name == "RAG":
            # Embedding + 检索 + LLM 生成
            return len(query.split()) * 0.00001 + len(answer.split()) * 0.00002
        elif system_name == "Fine-tuned":
            # 只有推理成本
            return len(answer.split()) * 0.00001
        return len(answer.split()) * 0.00002
```

**关键评估维度：**
1. **回答准确率**（Accuracy）：与 Ground Truth 的一致性
2. **幻觉率**（Faithfulness）：生成内容的可信度
3. **延迟**（Latency）：端到端响应时间
4. **成本**（Cost）：每次查询的 API 费用
5. **知识新鲜度**（Freshness）：对最新信息的处理能力

**考察点：** 科学的实验设计能力，能用数据支撑技术决策。

---

**Q98. 当知识库只有几百个文档时，RAG 还适用吗？还是 Fine-tuning 更好？**
[难度：⭐⭐] [类型：场景]
**答：**
**小规模知识库（< 1000 文档）的决策矩阵：**

**仍然推荐 RAG 的条件：**
- 文档内容经常更新（每周/每月）
- 需要精确引用来源
- LLM 的上下文窗口足够大时，可以考虑"Full-Context RAG"（把所有文档都塞入上下文）
- 涉及敏感/机密信息，不能放入训练集

**考虑 Fine-tuning 的条件：**
- 文档内容稳定（6个月以上不变）
- 对回答风格/格式有严格要求（需要学会特定输出格式）
- 延迟敏感，无法接受检索步骤

**全文档上下文（Full-Context）方案：**
```python
# 当文档量小时，直接把全部内容放入 Prompt（无需向量检索）
from langchain_anthropic import ChatAnthropic

all_docs_content = "\n\n---\n\n".join([d.page_content for d in all_documents])

# Claude 等支持 200K token 的模型可以处理大量文档
llm = ChatAnthropic(model="claude-opus-4-5")
response = llm.invoke(f"""知识库内容：
{all_docs_content}

用户问题：{query}

请基于以上知识库内容回答。""")
```

**工程建议：** 先用 RAG 快速实现，上线后通过 RAGAS 评测发现不足，再决定是否需要 Fine-tuning。

**考察点：** 小规模数据集场景的实际工程判断，避免过度工程化。

---

**Q99. 解释 "Long-Context LLM" 的出现如何改变了 RAG 的必要性？**
[难度：⭐⭐⭐] [类型：概念]
**答：**
**Long-Context LLM 的现状（2024-2025）：**
- GPT-4 Turbo：128K tokens
- Claude 3.5+：200K tokens
- Gemini 1.5 Pro：1M tokens

**对 RAG 的冲击：**
理论上，如果上下文窗口足够大，可以把整个知识库放入 Prompt，无需向量检索。

**RAG 仍然必要的原因：**

| 问题 | Long-Context 的局限 | RAG 的优势 |
|------|---------------------|-----------|
| 成本 | 1M tokens 的 API 调用费用极高 | 只检索 Top-K，成本可控 |
| 速度 | 处理 1M tokens 需要数秒 | 检索+短 Prompt，响应快 |
| Lost in the Middle | 即使是 1M context，中间内容仍易被忽略 | 精准检索，只送最相关内容 |
| 知识库超出窗口 | 企业知识库可能有数十万文档 | 向量检索没有上限 |
| 动态更新 | 每次更新都需要重新构建 Prompt | 增量更新更高效 |

**Long-Context + RAG 的最优组合：**
```python
def adaptive_context_strategy(query, knowledge_base, llm_with_large_context):
    # 对于简短知识库（< 50K tokens），直接全文上下文
    if knowledge_base.total_tokens < 50000:
        return full_context_response(query, knowledge_base, llm_with_large_context)
    # 对于大型知识库，使用 RAG
    else:
        return rag_response(query, knowledge_base)
```

**考察点：** 对 RAG 在 Long-Context LLM 时代的价值重新评估，展示技术视野。

---

**Q100. 综合题：为一家需要部署"智能客服 RAG 系统"的中型企业设计完整的技术方案。**
[难度：⭐⭐⭐] [类型：设计]
**答：**
**业务背景：** 10000+ 产品文档，日均 5000 次查询，中文为主，要求延迟 < 2s，支持文档定期更新。

**完整技术方案：**

**1. 文档处理层**
```
PDF/Word/HTML → unstructured 解析 → 清洗（去页眉页脚、去重）
→ RecursiveCharacterTextSplitter（chunk_size=800, overlap=100）
→ 元数据提取（文档名、章节、更新日期、产品类别）
```

**2. 索引层**
```python
# Embedding 模型：BAAI/bge-large-zh-v1.5（中文最优）
# 向量数据库：Qdrant（高性能、支持过滤、可私有化部署）
# 索引类型：HNSW（M=32, ef_construction=200）
# 增量更新：基于文档 hash 检测变化，只更新修改的文档
```

**3. 检索层**
```python
# 混合检索：向量检索（权重0.6）+ BM25（权重0.4）
# Re-ranking：BAAI/bge-reranker-large
# 元数据过滤：按产品线、文档类型过滤
# 两阶段：Top-30 粗筛 → Re-rank → Top-5
```

**4. 生成层**
```python
# LLM：gpt-4o-mini（速度快、成本低）
# Prompt：严格上下文约束 + 来源引用
# 流式输出：减少用户等待感知
# 回退策略：无结果时返回"请联系人工客服"
```

**5. 评测层**
```python
# RAGAS 离线评测：每周运行，监控 Faithfulness/Recall/Precision
# 在线监控：LangSmith 追踪，关注相似度分布、空结果率
# A/B 测试：新配置先在 10% 流量上验证
```

**6. 安全层**
- 基于用户角色的文档访问控制
- Prompt Injection 检测
- PII（敏感信息）脱敏

**估算指标：**
- 延迟分布：Embedding 30ms + 检索 30ms + Rerank 200ms + LLM 500ms = ~760ms（p50）
- 月成本（5000次/天）：Embedding + LLM ≈ 约$500~$1000/月
- 召回率目标：Context Recall > 0.85，Faithfulness > 0.90

**考察点：** 综合运用全部 RAG 知识，体现系统性设计能力和实际工程经验。

---

> **结语：** RAG 技术仍在快速演进，从 Naive RAG → Advanced RAG → Agentic RAG → GraphRAG，每一步都是对"如何让 LLM 更可靠地访问外部知识"这一问题的深化探索。掌握本题库的 100 个问题，将帮助你在面试和实际项目中从容应对 RAG 相关挑战。
