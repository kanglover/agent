/**
 * 20_production_patterns.js
 * =========================
 * 生产级 Agent 常用模式合集
 *
 * 包含以下 7 个核心模式，每个都有完整实现和使用示例：
 * 1. RequestQueue      —— 请求队列，控制并发为 5，防止打爆 API 速率
 * 2. MultiSessionManager —— 多用户会话管理，LRU 淘汰策略
 * 3. ResponseCache     —— 内存缓存，TTL = 5 分钟，避免重复调用
 * 4. TokenTracker      —— 按用户统计 token 消耗
 * 5. ABTestRouter      —— 50/50 分流不同 Prompt
 * 6. HealthMonitor     —— 跟踪成功率和平均延迟
 * 7. GracefulDegradation —— 主模型失败时自动切换备用模型
 *
 * 安装：npm install @anthropic-ai/sdk
 * 运行：node 20_production_patterns.js
 */

import Anthropic from '@anthropic-ai/sdk';
import { createHash } from 'crypto';

const anthropic = new Anthropic(); // 读取 ANTHROPIC_API_KEY 环境变量

// ═══════════════════════════════════════════════════════════════
// 1. RequestQueue —— 请求队列（控制并发 + 限速）
//
// 问题背景：
//   如果 100 个用户同时请求，你的代码会同时发出 100 个 API 请求，
//   触发 Anthropic 的速率限制（Rate Limit），导致大量 429 错误。
//
// 解决方案：
//   把所有请求放入队列，最多同时执行 5 个，其余排队等待。
//   类比：银行只开 5 个窗口，多余的客户在候区等待，不会混乱。
// ═══════════════════════════════════════════════════════════════

class RequestQueue {
  /**
   * @param {number} concurrency  - 最大并发数（同时处理的请求数）
   */
  constructor(concurrency = 5) {
    this.concurrency = concurrency;
    this.running = 0;        // 当前正在执行的请求数
    this.queue = [];         // 等待执行的请求（每项是个函数）
    this.totalProcessed = 0; // 总处理数（监控用）
  }

  /**
   * 将一个异步任务加入队列
   * @param {Function} task  - 返回 Promise 的异步函数
   * @returns {Promise}      - 任务执行结果的 Promise
   */
  async add(task) {
    return new Promise((resolve, reject) => {
      // 把 {task, resolve, reject} 放入队列
      this.queue.push({ task, resolve, reject });
      // 尝试立即执行（如果还有空闲槽位）
      this._processNext();
    });
  }

  _processNext() {
    // 如果已达并发上限，或队列为空，不处理
    while (this.running < this.concurrency && this.queue.length > 0) {
      const { task, resolve, reject } = this.queue.shift(); // 取出队列第一项
      this.running++;

      task()
        .then(result => {
          this.running--;
          this.totalProcessed++;
          resolve(result);
          this._processNext(); // 一个任务完成，尝试处理下一个
        })
        .catch(err => {
          this.running--;
          reject(err);
          this._processNext();
        });
    }
  }

  /** 获取队列状态 */
  getStatus() {
    return {
      running: this.running,         // 当前并发数
      queued: this.queue.length,     // 排队等待数
      totalProcessed: this.totalProcessed,
    };
  }
}

// 使用示例
async function demo_RequestQueue() {
  console.log('\n=== 1. RequestQueue 请求队列 ===');

  const queue = new RequestQueue(5); // 最大并发 5

  // 模拟 15 个并发请求
  const requests = Array.from({ length: 15 }, (_, i) => `用户${i + 1}`);

  console.log(`  同时发出 ${requests.length} 个请求，最大并发 5...`);

  const results = await Promise.all(
    requests.map(userId =>
      queue.add(async () => {
        // 模拟 API 调用（实际会调用 anthropic.messages.create）
        await new Promise(r => setTimeout(r, Math.random() * 200 + 100));
        return `${userId} 的回复`;
      })
    )
  );

  console.log(`  全部完成！处理了 ${results.length} 个请求`);
  console.log('  队列状态：', queue.getStatus());
}

// ═══════════════════════════════════════════════════════════════
// 2. MultiSessionManager —— 多用户会话管理（LRU 淘汰）
//
// 问题背景：
//   多用户聊天时，每个用户的对话历史必须隔离，不能串台。
//   同时会话数量可能很多，需要设置上限防止内存溢出。
//
// LRU（Least Recently Used）淘汰策略：
//   当会话数量超过上限时，删除最久没有使用的会话。
//   类比：书架放满了，把最久没翻的书移走。
// ═══════════════════════════════════════════════════════════════

class MultiSessionManager {
  /**
   * @param {number} maxSessions  - 最大会话数（超过时 LRU 淘汰）
   * @param {number} maxTurns     - 每个会话保留的最大对话轮数（滑动窗口）
   */
  constructor(maxSessions = 1000, maxTurns = 20) {
    this.maxSessions = maxSessions;
    this.maxTurns = maxTurns;
    // Map 天然保持插入顺序，配合 delete+set 实现 LRU
    this.sessions = new Map(); // sessionId → { messages, lastActive }
  }

  /** 获取或创建会话 */
  getSession(sessionId) {
    if (!this.sessions.has(sessionId)) {
      // 超过上限时，删除最久未使用的会话（Map 第一项）
      if (this.sessions.size >= this.maxSessions) {
        const oldestKey = this.sessions.keys().next().value;
        this.sessions.delete(oldestKey);
        console.log(`  [LRU] 淘汰会话：${oldestKey}`);
      }
      this.sessions.set(sessionId, { messages: [], lastActive: Date.now() });
    }

    // 更新最后活跃时间（将此项移到 Map 末尾，表示最近使用）
    const session = this.sessions.get(sessionId);
    session.lastActive = Date.now();
    this.sessions.delete(sessionId);
    this.sessions.set(sessionId, session); // 重新插入到末尾

    return session;
  }

  /** 向会话添加消息 */
  addMessage(sessionId, role, content) {
    const session = this.getSession(sessionId);
    session.messages.push({ role, content });

    // 滑动窗口：超过 maxTurns 时截断最早的消息
    if (session.messages.length > this.maxTurns) {
      session.messages.splice(0, session.messages.length - this.maxTurns);
    }
  }

  /** 获取会话的消息历史 */
  getMessages(sessionId) {
    return this.getSession(sessionId).messages;
  }

  /** 清除指定会话 */
  clearSession(sessionId) {
    this.sessions.delete(sessionId);
  }

  /** 获取管理器状态 */
  getStats() {
    return {
      activeSessions: this.sessions.size,
      maxSessions: this.maxSessions,
    };
  }
}

// 使用示例
async function demo_MultiSessionManager() {
  console.log('\n=== 2. MultiSessionManager 多用户会话管理 ===');

  const manager = new MultiSessionManager(100, 20); // 最多 100 个会话，每个最多 20 轮

  // 模拟 3 个用户各自聊天
  const users = ['alice', 'bob', 'carol'];

  for (const userId of users) {
    manager.addMessage(userId, 'user', '你好');
    manager.addMessage(userId, 'assistant', '你好！有什么可以帮你的？');
    manager.addMessage(userId, 'user', '今天天气怎么样？');

    const msgs = manager.getMessages(userId);
    console.log(`  ${userId} 的会话：${msgs.length} 条消息，最后一条：${msgs[msgs.length - 1].content}`);
  }

  console.log('  会话状态：', manager.getStats());
}

// ═══════════════════════════════════════════════════════════════
// 3. ResponseCache —— 内存缓存（TTL = 5 分钟）
//
// 适用场景：FAQ 问答、文档摘要等"相同输入产生相同输出"的任务。
// 不适用：需要实时数据（天气、股价）或个性化响应的场景。
//
// TTL（Time To Live）：缓存有效期，超过后自动失效。
// 类比：便利店的存货单——超过保质期的扔掉，重新进货。
// ═══════════════════════════════════════════════════════════════

class ResponseCache {
  /**
   * @param {number} ttlMs  - 缓存有效期（毫秒），默认 5 分钟
   */
  constructor(ttlMs = 5 * 60 * 1000) {
    this.ttlMs = ttlMs;
    this.cache = new Map(); // key → { value, expiresAt }
    this.hitCount = 0;      // 缓存命中次数
    this.missCount = 0;     // 缓存未命中次数

    // 每分钟清理一次过期缓存（防止内存泄漏）
    this._cleanupTimer = setInterval(() => this._cleanup(), 60 * 1000);
  }

  /** 生成缓存 key（对输入做哈希，相同输入得到相同 key） */
  _buildKey(prompt, model) {
    return createHash('sha256')
      .update(`${model}:${prompt}`)
      .digest('hex')
      .slice(0, 16);
  }

  /** 查询缓存 */
  get(prompt, model) {
    const key = this._buildKey(prompt, model);
    const entry = this.cache.get(key);

    if (!entry) {
      this.missCount++;
      return null; // 未命中
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key); // 已过期，删除
      this.missCount++;
      return null;
    }

    this.hitCount++;
    return entry.value; // 命中，返回缓存值
  }

  /** 写入缓存 */
  set(prompt, model, value) {
    const key = this._buildKey(prompt, model);
    this.cache.set(key, {
      value,
      expiresAt: Date.now() + this.ttlMs,
    });
  }

  /** 清理过期条目 */
  _cleanup() {
    const now = Date.now();
    for (const [key, entry] of this.cache) {
      if (now > entry.expiresAt) this.cache.delete(key);
    }
  }

  /** 停止自动清理（程序退出前调用） */
  destroy() {
    clearInterval(this._cleanupTimer);
  }

  /** 获取缓存统计 */
  getStats() {
    const total = this.hitCount + this.missCount;
    return {
      size: this.cache.size,
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate: total > 0 ? `${((this.hitCount / total) * 100).toFixed(1)}%` : '0%',
    };
  }
}

// 使用示例
async function demo_ResponseCache() {
  console.log('\n=== 3. ResponseCache 内存缓存（TTL=5分钟）===');

  const cache = new ResponseCache(5 * 60 * 1000); // 5 分钟 TTL

  // 模拟带缓存的 API 调用
  async function cachedComplete(prompt, model = 'claude-opus-4-5') {
    const cached = cache.get(prompt, model);
    if (cached) {
      console.log('  [缓存命中]', prompt.slice(0, 30));
      return cached;
    }

    console.log('  [缓存未命中，调用 API]', prompt.slice(0, 30));
    // 模拟 API 调用
    const response = `[模拟回复] ${prompt} 的答案是...`;
    cache.set(prompt, model, response);
    return response;
  }

  // 同样的问题调用两次，第二次命中缓存
  await cachedComplete('什么是 TypeScript？');
  await cachedComplete('什么是 TypeScript？'); // 命中缓存
  await cachedComplete('什么是 React？');
  await cachedComplete('什么是 TypeScript？'); // 再次命中缓存

  console.log('  缓存统计：', cache.getStats());
  cache.destroy();
}

// ═══════════════════════════════════════════════════════════════
// 4. TokenTracker —— 按用户统计 token 消耗
//
// 用途：
// - 按用户计费（SaaS 产品中常见）
// - 监控哪些用户消耗 token 最多（异常检测）
// - 统计产品整体 API 成本
// ═══════════════════════════════════════════════════════════════

class TokenTracker {
  constructor() {
    // userId → { inputTokens, outputTokens, requestCount, firstSeen, lastSeen }
    this.userStats = new Map();
    this.globalStats = { inputTokens: 0, outputTokens: 0, requestCount: 0 };
  }

  /**
   * 记录一次 API 调用的 token 使用
   * @param {string} userId   - 用户标识
   * @param {object} usage    - Anthropic API 返回的 usage 对象
   *                           { input_tokens: number, output_tokens: number }
   */
  record(userId, usage) {
    if (!usage) return;

    const input = usage.input_tokens ?? 0;
    const output = usage.output_tokens ?? 0;
    const now = Date.now();

    // 更新用户统计
    const user = this.userStats.get(userId) ?? {
      inputTokens: 0, outputTokens: 0, requestCount: 0,
      firstSeen: now, lastSeen: now,
    };

    user.inputTokens += input;
    user.outputTokens += output;
    user.requestCount += 1;
    user.lastSeen = now;
    this.userStats.set(userId, user);

    // 更新全局统计
    this.globalStats.inputTokens += input;
    this.globalStats.outputTokens += output;
    this.globalStats.requestCount += 1;
  }

  /** 获取指定用户的统计 */
  getUserStats(userId) {
    return this.userStats.get(userId) ?? { inputTokens: 0, outputTokens: 0, requestCount: 0 };
  }

  /** 获取全局统计 */
  getGlobalStats() {
    return this.globalStats;
  }

  /** 按 token 消耗排名（取前 N 名） */
  getTopUsers(n = 10) {
    return [...this.userStats.entries()]
      .sort((a, b) => (b[1].inputTokens + b[1].outputTokens) - (a[1].inputTokens + a[1].outputTokens))
      .slice(0, n)
      .map(([userId, stats]) => ({ userId, ...stats }));
  }

  /** 估算费用（以 Claude Opus 定价为例） */
  estimateCost(userId) {
    const stats = this.getUserStats(userId);
    // Claude Opus 价格（近似）：输入 $15/1M tokens，输出 $75/1M tokens
    const inputCost = (stats.inputTokens / 1_000_000) * 15;
    const outputCost = (stats.outputTokens / 1_000_000) * 75;
    return {
      inputCostUSD: inputCost.toFixed(4),
      outputCostUSD: outputCost.toFixed(4),
      totalCostUSD: (inputCost + outputCost).toFixed(4),
    };
  }
}

// 使用示例
async function demo_TokenTracker() {
  console.log('\n=== 4. TokenTracker Token 统计 ===');

  const tracker = new TokenTracker();

  // 模拟几次 API 调用的 token 记录
  const mockUsages = [
    { userId: 'alice', usage: { input_tokens: 150, output_tokens: 80 } },
    { userId: 'bob',   usage: { input_tokens: 300, output_tokens: 150 } },
    { userId: 'alice', usage: { input_tokens: 200, output_tokens: 120 } },
    { userId: 'carol', usage: { input_tokens: 50,  output_tokens: 30 } },
    { userId: 'bob',   usage: { input_tokens: 400, output_tokens: 200 } },
  ];

  for (const { userId, usage } of mockUsages) {
    tracker.record(userId, usage);
  }

  console.log('  各用户统计：');
  for (const { userId } of [{ userId: 'alice' }, { userId: 'bob' }, { userId: 'carol' }]) {
    const stats = tracker.getUserStats(userId);
    const cost = tracker.estimateCost(userId);
    console.log(`  ${userId}：输入 ${stats.inputTokens} / 输出 ${stats.outputTokens} tokens，调用 ${stats.requestCount} 次，估算费用 $${cost.totalCostUSD}`);
  }

  console.log('\n  全局统计：', tracker.getGlobalStats());
  console.log('\n  Top 用户（按总 token）：', tracker.getTopUsers(3).map(u => `${u.userId}(${u.inputTokens + u.outputTokens})`).join(', '));
}

// ═══════════════════════════════════════════════════════════════
// 5. ABTestRouter —— A/B 测试分流（50/50）
//
// 目的：对比两种不同的 Prompt 策略，看哪个效果更好。
// 关键要求：同一个用户每次都落到同一组（稳定性），
//            且两组用户数量大致相等（公平性）。
//
// 实现方式：对 userId 做哈希，哈希值奇偶决定分组。
// ═══════════════════════════════════════════════════════════════

class ABTestRouter {
  /**
   * @param {object} variants  - A/B 变体配置，key 为变体名，value 为 Prompt 配置
   * @param {number} splitRatio - A 组的比例（0~1），默认 0.5（50/50）
   */
  constructor(variants, splitRatio = 0.5) {
    this.variants = variants;
    this.splitRatio = splitRatio;
    // 记录每个变体被分配的用户数（用于验证分流是否均匀）
    this.variantCounts = Object.fromEntries(Object.keys(variants).map(k => [k, 0]));
  }

  /**
   * 根据 userId 稳定地分配变体
   * 同一个 userId 每次调用都返回同一个变体（确定性）
   */
  getVariant(userId) {
    // MD5 哈希 → 取前 8 位十六进制 → 转为 0~1 之间的小数
    const hash = createHash('md5').update(userId).digest('hex');
    const ratio = parseInt(hash.slice(0, 8), 16) / 0xFFFFFFFF;

    const variantKeys = Object.keys(this.variants);
    const variantKey = ratio < this.splitRatio ? variantKeys[0] : variantKeys[1];

    this.variantCounts[variantKey]++;
    return { variantKey, config: this.variants[variantKey] };
  }

  /** 获取分流统计 */
  getDistribution() {
    const total = Object.values(this.variantCounts).reduce((a, b) => a + b, 0);
    return Object.fromEntries(
      Object.entries(this.variantCounts).map(([k, count]) => [
        k,
        { count, percentage: total > 0 ? `${((count / total) * 100).toFixed(1)}%` : '0%' },
      ])
    );
  }
}

// 使用示例
async function demo_ABTestRouter() {
  console.log('\n=== 5. ABTestRouter A/B 测试分流 ===');

  const router = new ABTestRouter({
    A: {
      systemPrompt: '你是一个专业助手，回答简洁精准。',
      model: 'claude-opus-4-5',
    },
    B: {
      systemPrompt: '你是一个友善的向导，用轻松的语气回答，多用比喻。',
      model: 'claude-opus-4-5',
    },
  }, 0.5); // 50/50 分流

  // 模拟 20 个用户
  const testUsers = Array.from({ length: 20 }, (_, i) => `user_${i.toString().padStart(3, '0')}`);

  for (const userId of testUsers) {
    const { variantKey } = router.getVariant(userId);
    process.stdout.write(variantKey + ' ');
  }
  console.log();

  console.log('\n  分流统计：', router.getDistribution());

  // 验证稳定性：同一用户多次调用应该得到相同结果
  const user = 'user_007';
  const v1 = router.getVariant(user).variantKey;
  const v2 = router.getVariant(user).variantKey;
  const v3 = router.getVariant(user).variantKey;
  console.log(`\n  稳定性验证（${user} 三次分配）：${v1} / ${v2} / ${v3}（应该全部相同）`);
}

// ═══════════════════════════════════════════════════════════════
// 6. HealthMonitor —— 跟踪成功率和平均延迟
//
// 用途：
// - 实时监控 API 调用的健康状况
// - 自动告警（成功率下降时发警报）
// - 为负载均衡决策提供数据
// ═══════════════════════════════════════════════════════════════

class HealthMonitor {
  /**
   * @param {number} windowSize  - 滑动窗口大小（保留最近 N 次请求的数据）
   */
  constructor(windowSize = 100) {
    this.windowSize = windowSize;
    this.records = []; // 最近 N 次请求的记录 [{ success, latencyMs, timestamp }]
  }

  /**
   * 记录一次请求结果
   * @param {boolean} success    - 是否成功
   * @param {number} latencyMs   - 延迟（毫秒）
   */
  record(success, latencyMs) {
    this.records.push({ success, latencyMs, timestamp: Date.now() });

    // 滑动窗口：超过上限时删除最早的记录
    if (this.records.length > this.windowSize) {
      this.records.shift();
    }
  }

  /**
   * 包装一个异步函数，自动记录成功率和延迟
   * @param {Function} fn  - 要监控的异步函数
   */
  async wrap(fn) {
    const start = Date.now();
    try {
      const result = await fn();
      this.record(true, Date.now() - start);
      return result;
    } catch (err) {
      this.record(false, Date.now() - start);
      throw err;
    }
  }

  /** 获取健康状态快照 */
  getStatus() {
    if (this.records.length === 0) {
      return { status: 'no_data', successRate: null, avgLatencyMs: null, sampleSize: 0 };
    }

    const successCount = this.records.filter(r => r.success).length;
    const successRate = successCount / this.records.length;
    const avgLatency = this.records.reduce((sum, r) => sum + r.latencyMs, 0) / this.records.length;

    const p99Index = Math.floor(this.records.length * 0.99);
    const sortedLatencies = [...this.records].sort((a, b) => a.latencyMs - b.latencyMs);
    const p99Latency = sortedLatencies[p99Index]?.latencyMs ?? avgLatency;

    // 健康等级判断
    let status = 'healthy';
    if (successRate < 0.95) status = 'degraded';
    if (successRate < 0.80) status = 'critical';

    return {
      status,
      successRate: `${(successRate * 100).toFixed(1)}%`,
      avgLatencyMs: Math.round(avgLatency),
      p99LatencyMs: Math.round(p99Latency),
      sampleSize: this.records.length,
    };
  }
}

// 使用示例
async function demo_HealthMonitor() {
  console.log('\n=== 6. HealthMonitor 健康监控 ===');

  const monitor = new HealthMonitor(50);

  // 模拟一批请求（90% 成功）
  for (let i = 0; i < 30; i++) {
    const success = Math.random() > 0.10; // 90% 成功率
    const latency = success
      ? Math.random() * 500 + 100   // 成功：100~600ms
      : Math.random() * 2000 + 500; // 失败：500~2500ms
    monitor.record(success, latency);
  }

  console.log('  当前健康状态：', monitor.getStatus());

  // 使用 wrap 自动监控
  let callCount = 0;
  async function mockAPICall() {
    callCount++;
    await new Promise(r => setTimeout(r, 100));
    if (callCount % 5 === 0) throw new Error('模拟偶发失败');
    return `响应 #${callCount}`;
  }

  console.log('\n  使用 wrap() 自动监控 10 次调用：');
  for (let i = 0; i < 10; i++) {
    try {
      await monitor.wrap(mockAPICall);
      process.stdout.write('✓ ');
    } catch {
      process.stdout.write('✗ ');
    }
  }
  console.log('\n  更新后健康状态：', monitor.getStatus());
}

// ═══════════════════════════════════════════════════════════════
// 7. GracefulDegradation —— 优雅降级
//
// 主模型失败时自动切换备用模型，保证服务可用性。
// 错误分类：
//   - 429 / 529：速率限制 / 过载，可重试 / 降级
//   - 500 / 503：服务器错误，可重试 / 降级
//   - 400 / 401：客户端错误，不重试（参数问题）
// ═══════════════════════════════════════════════════════════════

class GracefulDegradation {
  /**
   * @param {string[]} models  - 模型优先级列表，第一个是主模型，后面是备用模型
   */
  constructor(models = ['claude-opus-4-5', 'claude-haiku-4-5']) {
    this.models = models;
    this.failureCounts = Object.fromEntries(models.map(m => [m, 0]));
  }

  /** 判断错误是否值得重试或降级 */
  _isRetryable(error) {
    const status = error?.status ?? error?.statusCode;
    return [429, 500, 503, 529].includes(status);
  }

  /**
   * 执行 API 调用，失败时自动降级到备用模型
   * @param {object} params  - Anthropic messages.create 参数（不含 model）
   * @returns {object}       - { response, modelUsed, usedFallback }
   */
  async complete(params) {
    for (let i = 0; i < this.models.length; i++) {
      const model = this.models[i];
      const isFallback = i > 0;

      if (isFallback) {
        console.log(`  [降级] 切换到备用模型：${model}`);
      }

      try {
        const response = await anthropic.messages.create({ ...params, model });
        if (isFallback) {
          console.log(`  [降级] 备用模型成功响应`);
        }
        return { response, modelUsed: model, usedFallback: isFallback };

      } catch (error) {
        this.failureCounts[model]++;
        console.error(`  [错误] ${model} 失败：${error.message}`);

        const isLast = i === this.models.length - 1;
        if (isLast || !this._isRetryable(error)) {
          throw error; // 所有模型都失败，或不可重试错误
        }
        // 继续尝试下一个模型
      }
    }
  }

  getFailureStats() {
    return this.failureCounts;
  }
}

// 使用示例
async function demo_GracefulDegradation() {
  console.log('\n=== 7. GracefulDegradation 优雅降级 ===');

  const degradation = new GracefulDegradation(['claude-opus-4-5', 'claude-haiku-4-5']);

  console.log('  （此示例展示降级逻辑，实际 API 调用已跳过）');
  console.log('  使用方式：');
  console.log(`
  const result = await degradation.complete({
    max_tokens: 1024,
    messages: [{ role: 'user', content: '你好' }],
  });

  console.log('使用的模型：', result.modelUsed);
  console.log('是否使用了备用模型：', result.usedFallback);
  console.log('回复：', result.response.content[0].text);
  `);

  console.log('  故障统计：', degradation.getFailureStats());
}

// ═══════════════════════════════════════════════════════════════
// 主函数：依次运行所有示例
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('🚀 生产级 Agent 模式示例集');
  console.log('（模拟模式：不会实际调用 Anthropic API）\n');

  await demo_RequestQueue();
  await demo_MultiSessionManager();
  await demo_ResponseCache();
  await demo_TokenTracker();
  await demo_ABTestRouter();
  await demo_HealthMonitor();
  await demo_GracefulDegradation();

  console.log('\n' + '═'.repeat(60));
  console.log('✅ 全部示例运行完成！');
  console.log('\n各模块适用场景速查：');
  console.log('  RequestQueue      → 高并发场景，防止 API 429 限速');
  console.log('  MultiSessionManager → 多用户聊天应用，会话隔离');
  console.log('  ResponseCache     → FAQ 问答，减少重复 API 调用');
  console.log('  TokenTracker      → SaaS 计费，成本监控');
  console.log('  ABTestRouter      → Prompt 效果对比实验');
  console.log('  HealthMonitor     → 生产监控，自动告警');
  console.log('  GracefulDegradation → 高可用，主模型故障自动切换');
}

// 导出所有类（供其他模块使用）
export {
  RequestQueue,
  MultiSessionManager,
  ResponseCache,
  TokenTracker,
  ABTestRouter,
  HealthMonitor,
  GracefulDegradation,
};

main().catch(console.error);
