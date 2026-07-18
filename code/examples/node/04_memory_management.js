// 运行: node 04_memory_management.js
// 依赖: npm install @anthropic-ai/sdk
//
// 对话记忆管理（Memory Management）
//
// 问题：Claude 没有原生记忆，每次请求都要发完整历史。
//       但历史越来越长，token 费用越来越高，甚至超出上下文窗口限制。
//
// 解法三板斧：
//   1. 滑动窗口：只保留最近 N 条消息（丢弃旧的）
//   2. AI 摘要压缩：把旧对话让 AI 总结成摘要，塞进 system prompt
//   3. JSON 持久化：把历史写入文件（类比浏览器的 localStorage）
//
// 多用户支持：用 Map<userId, messages> 隔离不同用户的会话

import Anthropic from '@anthropic-ai/sdk';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const client = new Anthropic();

// ═══════════════════════════════════════════════════════════════
// Token 估算工具
// 精确 token 数需要 tokenizer，这里用字符数 / 4 做粗略估算
// 英文约 1 token = 4 字符，中文约 1 token = 1-2 字符
// ═══════════════════════════════════════════════════════════════

/**
 * 粗略估算文本的 token 数
 * @param {string} text
 * @returns {number}
 */
function estimateTokens(text) {
  if (!text) return 0;
  // 中文字符单独计算（每个约 1.5 token），其余按 4 字符/token
  const chineseChars = (text.match(/[一-龥]/g) ?? []).length;
  const otherChars = text.length - chineseChars;
  return Math.ceil(chineseChars * 1.5 + otherChars / 4);
}

/**
 * 计算消息数组的总 token 估算
 * @param {Array<{role: string, content: string}>} messages
 * @returns {number}
 */
function estimateTotalTokens(messages) {
  return messages.reduce((sum, msg) => {
    const contentText = typeof msg.content === 'string'
      ? msg.content
      : JSON.stringify(msg.content);
    return sum + estimateTokens(contentText) + 4; // +4 是每条消息的结构开销
  }, 0);
}

// ═══════════════════════════════════════════════════════════════
// ConversationMemory 类：核心记忆管理器
// ═══════════════════════════════════════════════════════════════

class ConversationMemory {
  /**
   * @param {object} options
   * @param {number} options.maxTokens         - 最大 token 数，超出则触发压缩
   * @param {number} options.windowSize        - 滑动窗口保留的消息数
   * @param {boolean} options.enableSummary    - 是否启用 AI 摘要压缩
   * @param {string|null} options.persistPath  - 持久化文件路径（null = 不持久化）
   */
  constructor({
    maxTokens = 4000,
    windowSize = 20,
    enableSummary = true,
    persistPath = null,
  } = {}) {
    this.maxTokens = maxTokens;
    this.windowSize = windowSize;
    this.enableSummary = enableSummary;
    this.persistPath = persistPath;

    // 消息历史
    this.messages = [];

    // 摘要（对旧对话的压缩结果）
    // 这个摘要会被注入到 system prompt 中，让 Claude 知道之前聊了什么
    this.summary = '';

    // 压缩次数统计
    this.compressionCount = 0;
  }

  /**
   * 添加一条消息
   * @param {'user'|'assistant'} role
   * @param {string} content
   */
  async addMessage(role, content) {
    this.messages.push({ role, content, timestamp: Date.now() });

    // 检查是否需要压缩
    const currentTokens = estimateTotalTokens(this.messages);
    if (currentTokens > this.maxTokens) {
      console.log(`\n⚠️  token 估算 ${currentTokens} 超过限制 ${this.maxTokens}，触发压缩...`);
      await this.compress();
    }

    // 如果开启持久化，自动保存
    if (this.persistPath) {
      await this.saveToDisk();
    }
  }

  /**
   * 压缩策略：先尝试 AI 摘要，降级到滑动窗口
   */
  async compress() {
    if (this.enableSummary && this.messages.length >= 4) {
      await this.summarizeOldMessages();
    } else {
      this.applySlideWindow();
    }
    this.compressionCount++;
  }

  /**
   * AI 摘要压缩：
   * 把前一半旧消息让 Claude 总结成摘要，只保留后一半新消息
   * 新摘要 = 旧摘要 + 旧消息的总结
   */
  async summarizeOldMessages() {
    // 取前一半作为"旧消息"来总结
    const halfPoint = Math.floor(this.messages.length / 2);
    const oldMessages = this.messages.slice(0, halfPoint);
    const recentMessages = this.messages.slice(halfPoint);

    console.log(`📝 AI 摘要压缩：总结前 ${oldMessages.length} 条，保留后 ${recentMessages.length} 条`);

    try {
      // 构建要总结的文本
      const conversationText = oldMessages
        .map(m => `${m.role === 'user' ? '用户' : 'AI'}: ${m.content}`)
        .join('\n');

      const prevSummarySection = this.summary
        ? `之前的摘要：\n${this.summary}\n\n新增对话：\n`
        : '对话内容：\n';

      // 让 Claude 生成摘要
      const summaryResponse = await client.messages.create({
        model: 'claude-opus-4-5',
        max_tokens: 512,
        messages: [
          {
            role: 'user',
            content: `请将以下对话摘要成 3-5 句话，保留关键信息（姓名、重要决定、上下文）：

${prevSummarySection}${conversationText}

摘要：`,
          },
        ],
      });

      this.summary = summaryResponse.content[0].text;
      this.messages = recentMessages; // 只保留新消息

      console.log(`✅ 摘要生成完毕（${estimateTokens(this.summary)} tokens）`);
      console.log(`摘要内容: ${this.summary.slice(0, 100)}...`);
    } catch (error) {
      console.warn(`摘要生成失败，降级到滑动窗口: ${error.message}`);
      this.applySlideWindow();
    }
  }

  /**
   * 滑动窗口：直接丢弃最旧的消息，只保留最近 windowSize 条
   * 简单粗暴但有效（丢失上下文，适合无状态对话）
   */
  applySlideWindow() {
    if (this.messages.length > this.windowSize) {
      const removed = this.messages.length - this.windowSize;
      this.messages = this.messages.slice(-this.windowSize);
      console.log(`🪟 滑动窗口：移除 ${removed} 条旧消息，保留最近 ${this.windowSize} 条`);
    }
  }

  /**
   * 获取当前状态（用于传给 Claude）
   * @returns {{ system: string, messages: Array }}
   */
  getContext() {
    // 如果有摘要，注入到 system prompt
    const systemAddition = this.summary
      ? `\n\n【对话历史摘要】\n${this.summary}`
      : '';

    return {
      systemAddition,
      messages: this.messages.map(({ role, content }) => ({ role, content })),
    };
  }

  /**
   * 获取内存使用统计
   * @returns {object}
   */
  getStats() {
    return {
      messageCount: this.messages.length,
      estimatedTokens: estimateTotalTokens(this.messages),
      summaryTokens: estimateTokens(this.summary),
      compressionCount: this.compressionCount,
      hasSummary: !!this.summary,
    };
  }

  // ─── 持久化（类比 localStorage）─────────────────────────────

  /**
   * 保存到磁盘（JSON 文件）
   * 类比：localStorage.setItem('chat_history', JSON.stringify(data))
   */
  async saveToDisk() {
    const data = {
      messages: this.messages,
      summary: this.summary,
      savedAt: new Date().toISOString(),
      stats: this.getStats(),
    };
    await fs.writeFile(this.persistPath, JSON.stringify(data, null, 2), 'utf-8');
  }

  /**
   * 从磁盘加载（JSON 文件）
   * 类比：JSON.parse(localStorage.getItem('chat_history'))
   * @returns {Promise<boolean>} - 是否成功加载
   */
  async loadFromDisk() {
    try {
      const raw = await fs.readFile(this.persistPath, 'utf-8');
      const data = JSON.parse(raw);
      this.messages = data.messages ?? [];
      this.summary = data.summary ?? '';
      console.log(`📂 从磁盘加载对话历史: ${this.messages.length} 条消息`);
      return true;
    } catch {
      return false; // 文件不存在或解析失败，从零开始
    }
  }

  /** 清除所有记忆 */
  clear() {
    this.messages = [];
    this.summary = '';
    this.compressionCount = 0;
  }
}

// ═══════════════════════════════════════════════════════════════
// 多用户会话管理器
// 用 Map<userId, ConversationMemory> 隔离不同用户
// ═══════════════════════════════════════════════════════════════

class MultiUserSessionManager {
  constructor() {
    // Map: userId → ConversationMemory 实例
    // 就像每个用户有自己独立的对话框
    this.sessions = new Map();
    this.persistDir = path.join(__dirname, '.sessions');
  }

  /**
   * 获取或创建用户的 Memory 实例
   * @param {string} userId
   * @returns {Promise<ConversationMemory>}
   */
  async getSession(userId) {
    if (!this.sessions.has(userId)) {
      // 为新用户创建 Memory，持久化到单独的文件
      await fs.mkdir(this.persistDir, { recursive: true });
      const persistPath = path.join(this.persistDir, `${userId}.json`);

      const memory = new ConversationMemory({
        maxTokens: 3000,
        windowSize: 15,
        enableSummary: true,
        persistPath,
      });

      // 尝试从磁盘恢复之前的会话
      await memory.loadFromDisk();
      this.sessions.set(userId, memory);
    }

    return this.sessions.get(userId);
  }

  /**
   * 向指定用户发送消息并获取回复
   * @param {string} userId
   * @param {string} userMessage
   * @returns {Promise<string>}
   */
  async chat(userId, userMessage) {
    const memory = await this.getSession(userId);

    // 把用户消息加入记忆
    await memory.addMessage('user', userMessage);

    const { systemAddition, messages } = memory.getContext();

    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 512,
      system: `你是一个友好的 AI 助理，请记住用户告诉你的信息。${systemAddition}`,
      messages,
    });

    const reply = response.content[0].text;

    // 把 AI 回复也加入记忆
    await memory.addMessage('assistant', reply);

    return reply;
  }

  /** 打印所有用户的会话统计 */
  printStats() {
    console.log('\n📊 多用户会话统计：');
    for (const [userId, memory] of this.sessions) {
      const stats = memory.getStats();
      console.log(`  用户 ${userId}: ${stats.messageCount} 条消息，~${stats.estimatedTokens} tokens，压缩 ${stats.compressionCount} 次`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// 演示
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('🧠 对话记忆管理示例\n');

  // ── 演示1：基础记忆管理 ─────────────────────────────────────
  console.log('=== 演示1：基础多轮对话记忆 ===\n');

  const persistPath = path.join(__dirname, '.demo_memory.json');
  const memory = new ConversationMemory({
    maxTokens: 2000,
    windowSize: 10,
    enableSummary: true,
    persistPath,
  });

  async function chat(userMessage) {
    await memory.addMessage('user', userMessage);
    const { systemAddition, messages } = memory.getContext();

    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 256,
      system: `你是一个记忆力很好的助理。${systemAddition}`,
      messages,
    });

    const reply = response.content[0].text;
    await memory.addMessage('assistant', reply);

    console.log(`用户: ${userMessage}`);
    console.log(`AI: ${reply}\n`);
    console.log(`[内存状态] ${JSON.stringify(memory.getStats())}\n`);
    return reply;
  }

  await chat('我叫小明，我是一名前端开发者。');
  await chat('我最近在学 React，遇到了 hooks 的问题。');
  await chat('你还记得我叫什么名字，我是做什么的吗？');

  // ── 演示2：多用户隔离 ───────────────────────────────────────
  console.log('\n=== 演示2：多用户会话隔离 ===\n');

  const manager = new MultiUserSessionManager();

  // 用户A 的对话
  console.log('--- 用户A ---');
  await manager.chat('user_alice', '我是 Alice，我在学 Python。');
  await manager.chat('user_alice', '你记得我叫什么吗？');

  // 用户B 的对话（完全独立，不互相干扰）
  console.log('--- 用户B ---');
  await manager.chat('user_bob', '我是 Bob，我在学 Java。');
  await manager.chat('user_bob', '你知道我在学什么吗？');

  manager.printStats();

  // 清理 demo 文件
  await fs.unlink(persistPath).catch(() => {});

  console.log('\n✅ 示例完成！');
  console.log('💡 下一步：查看 05_rag_simple.js 学习 RAG（检索增强生成）');
}

main().catch(err => {
  console.error('❌ 错误：', err.message);
  process.exit(1);
});
