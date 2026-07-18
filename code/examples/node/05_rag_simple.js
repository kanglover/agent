// 运行: node 05_rag_simple.js
// 依赖: npm install @anthropic-ai/sdk
//
// RAG = Retrieval-Augmented Generation（检索增强生成）
//
// 问题：Claude 只知道训练数据里的内容，不知道你自己的文档。
// 解法：先从你的文档里"检索"相关片段，再把片段 + 问题一起发给 Claude 生成回答。
//
// 类比：
//   普通 AI：考试时全凭记忆（Claude 的训练数据）
//   RAG：考试时可以翻书（你的文档库）→ 找到相关段落 → 结合书上内容回答
//
// 本文件用纯 JS 实现，不依赖向量数据库，适合小规模场景（<1000 文档）：
//   - 余弦相似度（手写）
//   - 文档分块（chunk）
//   - TF-IDF 风格的简单向量化
//   - 完整检索 → 生成流程

import Anthropic from '@anthropic-ai/sdk';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const client = new Anthropic();

// ═══════════════════════════════════════════════════════════════
// 第一节：文本向量化（把文字变成数字数组）
// 这是 RAG 的核心：数字可以计算相似度，文字不行
// ═══════════════════════════════════════════════════════════════

/**
 * 简单的词频向量化（类似 Bag of Words）
 * 把一段文字变成"词 → 出现次数"的向量
 *
 * 局限：无法理解语义（"汽车" 和 "轿车" 被认为不相关）
 * 实际生产中使用嵌入模型（Embedding Model）生成语义向量
 *
 * @param {string} text - 输入文本
 * @returns {Map<string, number>} - 词频 Map
 */
function textToVector(text) {
  // 分词：按空格和标点分割，转小写，过滤短词
  const words = text
    .toLowerCase()
    .replace(/[^\w一-龥\s]/g, ' ') // 保留英文、中文、空格
    .split(/\s+/)
    .filter(w => w.length > 1);            // 过滤单字符

  // 统计词频
  const vector = new Map();
  for (const word of words) {
    vector.set(word, (vector.get(word) ?? 0) + 1);
  }
  return vector;
}

/**
 * 余弦相似度（Cosine Similarity）
 * 衡量两段文字的"方向相似度"（0 = 完全不同，1 = 完全相同）
 *
 * 类比：两个人走路方向的相似程度。
 *   方向完全一样 → 相似度 1
 *   方向完全垂直 → 相似度 0
 *   方向相反   → 相似度 -1
 *
 * @param {Map<string, number>} vecA
 * @param {Map<string, number>} vecB
 * @returns {number} - 0 到 1 之间
 */
function cosineSimilarity(vecA, vecB) {
  // 点积：两个向量对应位置相乘再求和
  let dotProduct = 0;
  for (const [word, countA] of vecA) {
    if (vecB.has(word)) {
      dotProduct += countA * vecB.get(word);
    }
  }

  // 模长：向量自身的"长度"
  const magnitudeA = Math.sqrt([...vecA.values()].reduce((sum, v) => sum + v * v, 0));
  const magnitudeB = Math.sqrt([...vecB.values()].reduce((sum, v) => sum + v * v, 0));

  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dotProduct / (magnitudeA * magnitudeB);
}

// ═══════════════════════════════════════════════════════════════
// 第二节：文档分块（Chunking）
// 把长文档切成小块，每块独立检索
// 为什么要分块？
//   - 整篇文档太长，把无关内容也发给 Claude 浪费 token
//   - 分块后可以精准找到最相关的段落
// ═══════════════════════════════════════════════════════════════

/**
 * 把长文本分割成固定大小的块
 * @param {string} text       - 原始文本
 * @param {number} chunkSize  - 每块的字符数（约 200-500 字符为宜）
 * @param {number} overlap    - 相邻块重叠的字符数（防止在边界切断句子含义）
 * @returns {string[]}
 */
function splitIntoChunks(text, chunkSize = 300, overlap = 50) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = start + chunkSize;

    // 尽量在句号/换行处切断，而不是切断单词中间
    if (end < text.length) {
      const breakPoints = ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? '];
      for (const bp of breakPoints) {
        const idx = text.lastIndexOf(bp, end);
        if (idx > start + chunkSize * 0.5) { // 切割点在后半段才有效
          end = idx + bp.length;
          break;
        }
      }
    }

    chunks.push(text.slice(start, Math.min(end, text.length)).trim());
    start = end - overlap; // 回退 overlap 字符，制造重叠
  }

  return chunks.filter(c => c.length > 20); // 过滤太短的碎片
}

// ═══════════════════════════════════════════════════════════════
// 第三节：SimpleRAG 类（核心逻辑）
// ═══════════════════════════════════════════════════════════════

class SimpleRAG {
  constructor({ chunkSize = 300, overlap = 50, topK = 3 } = {}) {
    this.chunkSize = chunkSize;
    this.overlap = overlap;
    this.topK = topK;      // 检索时返回最相关的 K 个块
    this.documents = [];   // 存储所有文档块及其向量
  }

  /**
   * 添加文档（字符串）到知识库
   * @param {string} text     - 文档内容
   * @param {string} source   - 文档来源标识（文件名等）
   */
  addDocument(text, source = 'unknown') {
    const chunks = splitIntoChunks(text, this.chunkSize, this.overlap);

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      this.documents.push({
        text: chunk,
        source,
        chunkIndex: i,
        vector: textToVector(chunk), // 预计算向量，检索时直接用
      });
    }

    console.log(`📄 添加文档 "${source}"：${chunks.length} 个块`);
  }

  /**
   * 从文件加载文档
   * @param {string} filePath
   */
  async addFile(filePath) {
    const content = await fs.readFile(filePath, 'utf-8');
    const filename = path.basename(filePath);
    this.addDocument(content, filename);
  }

  /**
   * 检索最相关的文档块
   * @param {string} query    - 用户问题
   * @param {number} topK     - 返回 K 个最相关的块
   * @returns {Array<{text, source, similarity}>}
   */
  retrieve(query, topK = this.topK) {
    const queryVector = textToVector(query);

    // 计算 query 与每个文档块的相似度，并排序
    const scored = this.documents
      .map(doc => ({
        text: doc.text,
        source: doc.source,
        chunkIndex: doc.chunkIndex,
        similarity: cosineSimilarity(queryVector, doc.vector),
      }))
      .sort((a, b) => b.similarity - a.similarity) // 从高到低排序
      .slice(0, topK);                              // 取前 K 个

    return scored;
  }

  /**
   * 完整的 RAG 流程：检索 → 构建 prompt → 生成回答
   * @param {string} question - 用户问题
   * @returns {Promise<{answer: string, sources: Array, retrievedChunks: Array}>}
   */
  async query(question) {
    console.log(`\n❓ 问题: ${question}`);

    // Step 1: 检索相关文档块
    const relevantChunks = this.retrieve(question);

    if (relevantChunks.length === 0) {
      return { answer: '知识库为空，无法回答。', sources: [], retrievedChunks: [] };
    }

    console.log(`🔍 检索到 ${relevantChunks.length} 个相关块：`);
    relevantChunks.forEach((c, i) => {
      console.log(`  ${i + 1}. [${c.source}] 相似度=${c.similarity.toFixed(3)}: ${c.text.slice(0, 60)}...`);
    });

    // Step 2: 构建包含上下文的 prompt
    const contextText = relevantChunks
      .map((c, i) => `[来源 ${i + 1}: ${c.source}]\n${c.text}`)
      .join('\n\n---\n\n');

    const prompt = `请根据以下参考资料回答用户的问题。

参考资料：
${contextText}

用户问题：${question}

要求：
1. 只根据参考资料中的信息回答
2. 如果参考资料中没有相关信息，明确说明
3. 回答要简洁清晰
4. 用中文回答`;

    // Step 3: 让 Claude 生成回答
    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 512,
      messages: [{ role: 'user', content: prompt }],
    });

    const answer = response.content[0].text;
    const sources = [...new Set(relevantChunks.map(c => c.source))];

    console.log(`\n💬 回答: ${answer}`);
    console.log(`📚 参考来源: ${sources.join(', ')}`);

    return { answer, sources, retrievedChunks: relevantChunks };
  }

  /** 获取知识库统计 */
  getStats() {
    return {
      totalChunks: this.documents.length,
      sources: [...new Set(this.documents.map(d => d.source))],
    };
  }
}

// ═══════════════════════════════════════════════════════════════
// 主函数演示
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('📚 简单 RAG 示例（纯 JS，不需要向量数据库）\n');

  const rag = new SimpleRAG({ chunkSize: 300, topK: 2 });

  // 添加示例文档（实际使用时可以加载真实文件）
  rag.addDocument(`
Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行时环境。
它使用事件驱动、非阻塞 I/O 模型，使其轻量且高效，适用于构建数据密集型实时应用程序。

Node.js 的主要特点：
1. 异步和事件驱动：所有 API 都是异步的，Node.js 基于服务器不等待任何 API 返回数据。
2. 非常快速：基于 Google Chrome 的 V8 JavaScript 引擎，Node.js 代码执行速度非常快。
3. 单线程但高度可扩展：Node.js 使用单线程模型和事件循环机制。
4. 无缓冲：Node.js 应用程序从不缓冲任何数据，这些应用程序只是以块的形式输出数据。

Node.js 的常见应用场景：
- RESTful API 和后端服务
- 实时通信应用（聊天、游戏）
- 流式数据处理
- 命令行工具
- 微服务架构
  `, 'nodejs-intro.txt');

  rag.addDocument(`
Claude API 使用指南

Anthropic 的 Claude API 提供了强大的语言模型能力。

基础调用：
使用 @anthropic-ai/sdk 包可以轻松调用 Claude API。
需要设置 ANTHROPIC_API_KEY 环境变量。

主要功能：
1. 文本对话：支持多轮对话，通过 messages 数组管理历史。
2. 工具调用（Tool Use）：让 Claude 能够调用外部函数。
3. 流式输出：使用 stream() 方法实现打字机效果。
4. Vision（图像理解）：可以传入图片进行分析。

定价策略：
Claude API 按 token 计费，输入和输出 token 分开计费。
使用 Prompt Caching 可以对重复的前缀缓存，降低成本。

模型选择：
- Claude 3.5 Sonnet：智能与速度的平衡，推荐日常使用
- Claude 3 Opus：最强推理能力，适合复杂任务
- Claude 3 Haiku：最快最便宜，适合简单任务
  `, 'claude-api-guide.txt');

  console.log(`\n📊 知识库统计: ${JSON.stringify(rag.getStats())}\n`);
  console.log('─'.repeat(60));

  // 测试查询
  await rag.query('Node.js 适合做什么类型的应用？');
  await rag.query('如何调用 Claude API 的流式输出功能？');
  await rag.query('Claude 不同模型之间有什么区别？');
  await rag.query('Python 和 JavaScript 哪个更好？'); // 知识库中没有这个信息

  console.log('\n✅ RAG 示例完成！');
  console.log('💡 下一步：查看 06_rag_advanced.js 了解带向量库的进阶 RAG');
}

main().catch(err => {
  console.error('❌ 错误：', err.message);
  process.exit(1);
});
