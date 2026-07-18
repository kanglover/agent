// 运行: node 06_rag_advanced.js
// 依赖: npm install @anthropic-ai/sdk @langchain/community @langchain/core langchain express multer
//
// 进阶 RAG：使用 LangChain MemoryVectorStore（内存向量库）
//
// 和 05_rag_simple.js 的区别：
//   简单版：手写余弦相似度 + 词频向量，适合学习和小规模
//   进阶版：MemoryVectorStore（真正的向量相似度搜索） + Express API，适合实际项目
//
// MemoryVectorStore 说明：
//   - 数据存在内存里（重启就没了），不需要安装 ChromaDB / Pinecone 等外部数据库
//   - 适合开发调试和中小规模文档（几千篇以内）
//   - 想持久化可以序列化到文件
//
// 架构：
//   POST /rag/upload → 上传文档 → 分块 → 生成 Embedding → 存入 MemoryVectorStore
//   POST /rag/query  → 接收问题 → 向量检索 → Claude 生成回答 → 返回 JSON

import Anthropic from '@anthropic-ai/sdk';
import express from 'express';
import { MemoryVectorStore } from '@langchain/community/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from '@langchain/core/documents';

const client = new Anthropic();
const app = express();
app.use(express.json({ limit: '10mb' })); // 解析 JSON 请求体，允许最大 10MB

// ═══════════════════════════════════════════════════════════════
// 自定义 Embedding：用 Claude 生成文本嵌入向量
// 说明：Anthropic 目前不提供专用 Embedding API，
//       这里用 Claude 的文本补全来模拟向量（简化演示）
//       生产环境推荐用 OpenAI text-embedding-3-small 或 Cohere Embed
// ═══════════════════════════════════════════════════════════════

/**
 * 简单的确定性哈希向量（用于演示，不依赖外部 embedding API）
 * 把文本转为 128 维的伪向量，相似文本的向量会有一定相关性
 * @param {string} text
 * @returns {number[]}
 */
function hashEmbedding(text, dims = 128) {
  const normalized = text.toLowerCase().replace(/\s+/g, ' ').trim();
  const vector = new Array(dims).fill(0);

  // 基于字符和位置生成向量值
  for (let i = 0; i < normalized.length; i++) {
    const charCode = normalized.charCodeAt(i);
    const pos = i % dims;
    vector[pos] += Math.sin(charCode * (pos + 1) * 0.01);
    vector[(pos + 1) % dims] += Math.cos(charCode * (i + 1) * 0.01);
  }

  // 归一化到 -1 ~ 1
  const magnitude = Math.sqrt(vector.reduce((s, v) => s + v * v, 0)) || 1;
  return vector.map(v => v / magnitude);
}

/**
 * LangChain Embeddings 接口的简单实现
 * 让 MemoryVectorStore 能调用我们的 hashEmbedding
 */
class SimpleEmbeddings {
  /**
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embedDocuments(texts) {
    return texts.map(t => hashEmbedding(t));
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedQuery(text) {
    return hashEmbedding(text);
  }
}

// ═══════════════════════════════════════════════════════════════
// RAG 服务核心逻辑
// ═══════════════════════════════════════════════════════════════

/** 全局向量库实例（内存存储，重启清空） */
let vectorStore = null;
const embeddings = new SimpleEmbeddings();

/** 文档统计 */
const docStats = {
  totalDocuments: 0,      // 上传的原始文档数
  totalChunks: 0,         // 分块后的块数
  sources: [],            // 文档来源列表
};

/**
 * 文本分块器（使用 LangChain 的 RecursiveCharacterTextSplitter）
 * 比简单按字符数切割更智能：优先在段落边界切割
 */
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,       // 每块最大字符数
  chunkOverlap: 80,     // 相邻块重叠字符数（防止丢失跨块的上下文）
  separators: ['\n\n', '\n', '。', '！', '？', '.', '!', '?', ' ', ''], // 优先级从高到低
});

/**
 * 上传文档到向量库
 * @param {string} text      - 文档内容
 * @param {string} source    - 来源标识
 * @param {object} metadata  - 额外元数据
 * @returns {Promise<{chunks: number, source: string}>}
 */
async function uploadDocument(text, source, metadata = {}) {
  // Step 1: 分块
  const rawChunks = await textSplitter.splitText(text);

  // Step 2: 包装为 LangChain Document 对象（带 metadata）
  const documents = rawChunks.map((chunk, index) =>
    new Document({
      pageContent: chunk,
      metadata: {
        source,
        chunkIndex: index,
        totalChunks: rawChunks.length,
        uploadedAt: new Date().toISOString(),
        ...metadata,
      },
    })
  );

  // Step 3: 存入向量库（会自动调用 embeddings.embedDocuments）
  if (!vectorStore) {
    // 第一次上传，创建向量库
    vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);
  } else {
    // 追加到现有向量库
    await vectorStore.addDocuments(documents);
  }

  // 更新统计
  docStats.totalDocuments++;
  docStats.totalChunks += rawChunks.length;
  if (!docStats.sources.includes(source)) {
    docStats.sources.push(source);
  }

  console.log(`📄 文档上传完成: "${source}" → ${rawChunks.length} 个块`);
  return { chunks: rawChunks.length, source };
}

/**
 * 检索 + 生成（核心 RAG 流程）
 * @param {string} question  - 用户问题
 * @param {number} topK      - 检索最相关的 K 个块
 * @returns {Promise<object>} - 结构化的 JSON 结果
 */
async function queryRAG(question, topK = 3) {
  if (!vectorStore) {
    return {
      answer: '知识库为空，请先上传文档。',
      sources: [],
      chunks: [],
      confidence: 0,
    };
  }

  // Step 1: 向量相似度检索
  // similaritySearchWithScore 返回 [Document, score] 数组
  const results = await vectorStore.similaritySearchWithScore(question, topK);

  if (results.length === 0) {
    return {
      answer: '未找到相关内容。',
      sources: [],
      chunks: [],
      confidence: 0,
    };
  }

  // 整理检索结果
  const retrievedChunks = results.map(([doc, score]) => ({
    text: doc.pageContent,
    source: doc.metadata.source,
    score: Number(score.toFixed(4)),
    metadata: doc.metadata,
  }));

  console.log(`🔍 检索到 ${retrievedChunks.length} 个相关块`);
  retrievedChunks.forEach((c, i) => {
    console.log(`  ${i + 1}. [${c.source}] score=${c.score}: ${c.text.slice(0, 60)}...`);
  });

  // Step 2: 构建上下文 prompt
  const contextText = retrievedChunks
    .map((c, i) => `【参考资料 ${i + 1}】来源：${c.source}\n${c.text}`)
    .join('\n\n---\n\n');

  const systemPrompt = `你是一个专业的知识问答助手。
请严格根据提供的参考资料回答用户问题。
如果参考资料中没有足够信息，请明确说明"参考资料中未找到相关信息"。
回答要准确、简洁，并指出信息来源。`;

  const userPrompt = `参考资料：
${contextText}

用户问题：${question}`;

  // Step 3: Claude 生成回答
  const response = await client.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: 'user', content: userPrompt }],
  });

  const answer = response.content[0].text;
  const sources = [...new Set(retrievedChunks.map(c => c.source))];
  const avgScore = retrievedChunks.reduce((s, c) => s + c.score, 0) / retrievedChunks.length;

  return {
    answer,
    sources,
    chunks: retrievedChunks,
    confidence: Number(avgScore.toFixed(4)),
    tokenUsage: response.usage,
  };
}

// ═══════════════════════════════════════════════════════════════
// Express API 路由
// ═══════════════════════════════════════════════════════════════

/**
 * POST /rag/upload
 * 上传文档到知识库
 *
 * 请求体：
 * {
 *   "text": "文档内容...",
 *   "source": "文档名称（可选）",
 *   "metadata": { "author": "..." }   可选
 * }
 */
app.post('/rag/upload', async (req, res) => {
  try {
    const { text, source = 'unknown', metadata = {} } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: '请提供 text 字段（文档内容）' });
    }

    if (text.length < 10) {
      return res.status(400).json({ error: '文档内容太短，至少需要 10 个字符' });
    }

    const result = await uploadDocument(text, source, metadata);

    res.json({
      success: true,
      message: `文档上传成功`,
      ...result,
      stats: docStats,
    });
  } catch (error) {
    console.error('上传失败：', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /rag/query
 * 向知识库提问
 *
 * 请求体：
 * {
 *   "question": "你的问题",
 *   "topK": 3   可选，默认 3
 * }
 *
 * 返回：
 * {
 *   "answer": "Claude 的回答",
 *   "sources": ["文档A", "文档B"],
 *   "chunks": [...],
 *   "confidence": 0.85,
 *   "tokenUsage": {...}
 * }
 */
app.post('/rag/query', async (req, res) => {
  try {
    const { question, topK = 3 } = req.body;

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: '请提供 question 字段' });
    }

    const result = await queryRAG(question, Math.min(topK, 10));
    res.json({ success: true, ...result });
  } catch (error) {
    console.error('查询失败：', error);
    res.status(500).json({ error: error.message });
  }
});

/** GET /rag/stats - 查看知识库统计 */
app.get('/rag/stats', (req, res) => {
  res.json({
    ...docStats,
    vectorStoreReady: vectorStore !== null,
  });
});

/** GET /rag/clear - 清空知识库 */
app.post('/rag/clear', (req, res) => {
  vectorStore = null;
  docStats.totalDocuments = 0;
  docStats.totalChunks = 0;
  docStats.sources = [];
  res.json({ success: true, message: '知识库已清空' });
});

// ═══════════════════════════════════════════════════════════════
// 启动服务 + 自动加载示例文档
// ═══════════════════════════════════════════════════════════════

const PORT = 3006;

app.listen(PORT, async () => {
  console.log(`🚀 RAG 服务启动：http://localhost:${PORT}`);
  console.log('\nAPI 端点：');
  console.log(`  POST /rag/upload  - 上传文档`);
  console.log(`  POST /rag/query   - 提问`);
  console.log(`  GET  /rag/stats   - 统计信息`);
  console.log(`  POST /rag/clear   - 清空知识库\n`);

  // 自动预加载示例文档
  console.log('正在加载示例文档...\n');

  await uploadDocument(`
Node.js 是基于 Chrome V8 引擎的 JavaScript 运行时，由 Ryan Dahl 于 2009 年创建。
它采用事件驱动、非阻塞 I/O 的编程模型，特别适合构建高并发的网络应用。

核心优势：
- 单线程事件循环，轻松处理数万并发连接
- npm 生态系统拥有超过 200 万个开源包
- 前后端可共用 JavaScript 代码
- 启动速度快，内存占用小

适用场景：RESTful API、实时聊天、流媒体处理、CLI 工具
  `, 'nodejs-overview');

  await uploadDocument(`
Claude API 快速入门

1. 安装 SDK：npm install @anthropic-ai/sdk

2. 设置 API 密钥：
   export ANTHROPIC_API_KEY="your-key-here"

3. 基础调用示例：
   import Anthropic from '@anthropic-ai/sdk';
   const client = new Anthropic();
   const msg = await client.messages.create({
     model: 'claude-opus-4-5',
     max_tokens: 1024,
     messages: [{ role: 'user', content: 'Hello!' }],
   });

4. 流式输出：使用 client.messages.stream() 方法

5. 工具调用：在 tools 数组中定义工具，Claude 会自动调用
  `, 'claude-quickstart');

  console.log('\n✅ 服务就绪！可以用以下命令测试：');
  console.log(`curl -X POST http://localhost:${PORT}/rag/query \\`);
  console.log(`  -H "Content-Type: application/json" \\`);
  console.log(`  -d '{"question": "Node.js 有什么优势？"}'`);
});
