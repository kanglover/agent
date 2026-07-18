/**
 * 09_context_engineering.js
 * ==========================
 * 上下文工程 —— 像搭积木一样精确控制 AI 的"工作记忆"
 *
 * 核心理念：
 *   LLM 能"看到"的东西只有 context window（上下文窗口）里的内容。
 *   上下文工程 = 把有限的 token 预算用在刀刃上。
 *
 * 涵盖内容：
 *   1. 模块化 Prompt 构建（像拼 React 组件的 props）
 *   2. buildSystemPrompt(role, context, tools) 函数
 *   3. Prompt Caching（cache_control 用法，省钱 + 提速）
 *   4. 上下文压缩（超过阈值时调用 AI 摘要旧消息）
 *   5. TokenBudgetManager —— Token 预算管理类
 *
 * 运行前：
 *   npm install @anthropic-ai/sdk
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 09_context_engineering.js
 */

import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ─────────────────────────────────────────────
// 1. 模块化 Prompt 构建
// ─────────────────────────────────────────────

/**
 * Prompt 片段 —— 像 React 的 props，每个模块负责一件事
 *
 * 类比：餐厅菜单
 *   - role      = 餐厅类型（中餐厅、西餐厅）
 *   - context   = 今日特供（当前场景信息）
 *   - tools     = 厨房设备列表（可调用的工具）
 *   - constraints = 食材限制（不能用什么）
 */

/**
 * 角色片段 —— 定义 AI 的身份和专长
 */
function rolePrompt(role, expertise) {
  return `你是一位${role}，专精于${expertise}。你的回答应当专业、简洁、直接。`;
}

/**
 * 上下文片段 —— 注入当前场景信息
 * @param {object} ctx - 上下文键值对
 */
function contextPrompt(ctx) {
  if (!ctx || Object.keys(ctx).length === 0) return '';
  const lines = Object.entries(ctx)
    .map(([k, v]) => `- ${k}: ${v}`)
    .join('\n');
  return `\n## 当前上下文\n${lines}`;
}

/**
 * 工具片段 —— 告诉 AI 它能用哪些工具
 * @param {Array<{name:string, description:string}>} tools
 */
function toolsPrompt(tools) {
  if (!tools || tools.length === 0) return '';
  const list = tools.map(t => `- ${t.name}: ${t.description}`).join('\n');
  return `\n## 可用工具\n${list}`;
}

/**
 * 约束片段 —— 设定边界规则
 * @param {string[]} constraints
 */
function constraintsPrompt(constraints) {
  if (!constraints || constraints.length === 0) return '';
  const list = constraints.map(c => `- ${c}`).join('\n');
  return `\n## 约束条件\n${list}`;
}

/**
 * buildSystemPrompt(role, context, tools)
 *
 * 构建完整的 System Prompt，把各个模块像 React props 一样拼在一起。
 * 类比：组装乐高积木 —— 每块积木（模块）独立设计，拼合后是完整作品。
 *
 * @param {string} role         - 角色名称（如 "技术支持专员"）
 * @param {object} context      - 上下文信息（键值对）
 * @param {Array}  tools        - 可用工具列表 [{name, description}]
 * @param {object} [options]    - 可选参数
 * @param {string} [options.expertise]    - 专业领域
 * @param {string[]} [options.constraints] - 约束条件
 */
function buildSystemPrompt(role, context = {}, tools = [], options = {}) {
  const { expertise = '通用知识', constraints = [] } = options;

  const parts = [
    rolePrompt(role, expertise),
    contextPrompt(context),
    toolsPrompt(tools),
    constraintsPrompt(constraints),
  ].filter(Boolean); // 过滤掉空字符串

  return parts.join('\n');
}

// ─────────────────────────────────────────────
// 2. Prompt Caching（提示词缓存）
// ─────────────────────────────────────────────

/**
 * Prompt Caching 使用说明
 *
 * 什么是 Prompt Caching？
 *   Anthropic 会把你标记了 cache_control 的内容缓存起来，
 *   下次请求如果前缀相同，直接从缓存读取，而不重新处理。
 *
 * 省什么？
 *   - 速度：缓存命中后，Time to First Token 明显下降
 *   - 成本：cache read 价格约是普通 input token 的 10%
 *
 * 类比：老师把课本内容打印好发给学生（缓存），
 *   而不是每次上课都重新手写一遍板书。
 *
 * 适合缓存的内容：
 *   - 长篇 System Prompt（说明书、规则书）
 *   - RAG 检索到的参考文档
 *   - Few-shot 示例（长例子）
 *   - 工具定义
 */
async function demoCaching() {
  console.log('\n=== Prompt Caching 演示 ===');

  // 模拟一份很长的系统文档（实际场景可能是几千 token 的规则手册）
  const longSystemDoc = `
## 产品知识库 v2.3

### 退换货政策
- 7天无理由退货
- 15天质量问题换货
- 30天内可申请维修

### 常见问题解答
Q: 如何追踪快递？
A: 登录账户 → 订单列表 → 点击快递单号

Q: 支持哪些支付方式？
A: 支付宝、微信支付、银行卡、花呗

Q: 发票如何开具？
A: 下单时勾选"需要发票"，填写抬头信息
`.repeat(3); // 重复几次模拟长文档

  // 把长文档标记为可缓存（放在 system 最后一块内容上）
  const response = await client.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 200,
    system: [
      {
        type: 'text',
        text: '你是一个客服助手。',
      },
      {
        type: 'text',
        text: longSystemDoc,
        // ← 这里是关键！标记这块内容可以被缓存
        // ephemeral = 短暂缓存（约5分钟，在同一提示词前缀下复用）
        cache_control: { type: 'ephemeral' },
      },
    ],
    messages: [
      { role: 'user', content: '退换货政策是什么？' },
    ],
  });

  console.log('回复:', response.content[0].text);

  // 查看缓存使用情况
  const usage = response.usage;
  console.log('\nToken 使用情况:');
  console.log('  输入 tokens:          ', usage.input_tokens);
  console.log('  输出 tokens:          ', usage.output_tokens);
  // cache_creation_input_tokens: 首次缓存时写入的 token 数（第一次请求时有）
  if (usage.cache_creation_input_tokens !== undefined) {
    console.log('  缓存写入 tokens:      ', usage.cache_creation_input_tokens, '（首次，按写入价格计费）');
  }
  // cache_read_input_tokens: 从缓存读取的 token 数（第二次以后才有）
  if (usage.cache_read_input_tokens !== undefined) {
    console.log('  缓存读取 tokens:      ', usage.cache_read_input_tokens, '（约为写入价格的 10%）');
  }
}

// ─────────────────────────────────────────────
// 3. 上下文压缩 —— 超过阈值时用 AI 摘要旧消息
// ─────────────────────────────────────────────

/**
 * 估算消息列表的 token 数（粗略估算）
 *
 * 精确计算需要用 tiktoken 或 Anthropic 的 token count API。
 * 粗略估算：1 token ≈ 4 个英文字符 ≈ 2~3 个中文字符
 *
 * @param {Array} messages - 消息列表
 */
function estimateTokens(messages) {
  const text = messages.map(m => {
    const content = Array.isArray(m.content)
      ? m.content.map(c => c.text || '').join('')
      : (m.content || '');
    return content;
  }).join('');

  // 中文约 2 字符/token，英文约 4 字符/token，取折中
  return Math.ceil(text.length / 2.5);
}

/**
 * 上下文压缩函数
 *
 * 当消息历史太长（超过 token 阈值）时，把最老的消息压缩成摘要，
 * 保留最新的 N 条消息（保持对话的连贯性）。
 *
 * 类比：你在做长篇笔记时，把旧笔记归纳成一段摘要，
 *   然后把摘要贴在新笔记本的第一页，节省空间。
 *
 * @param {Array}  messages     - 当前消息列表
 * @param {number} tokenLimit   - Token 阈值（默认 50000）
 * @param {number} keepRecent   - 保留最新多少条消息（默认 10）
 */
async function compressContext(messages, tokenLimit = 50000, keepRecent = 10) {
  const estimated = estimateTokens(messages);

  if (estimated <= tokenLimit) {
    return messages; // 未超阈值，不压缩
  }

  console.log(`\n[上下文压缩] 估算 token 数: ${estimated}，超过阈值 ${tokenLimit}，开始压缩`);

  // 分割：旧消息（将被摘要）+ 最新消息（保留）
  const oldMessages = messages.slice(0, -keepRecent);
  const recentMessages = messages.slice(-keepRecent);

  // 把旧消息格式化为文本
  const oldText = oldMessages
    .map(m => {
      const content = Array.isArray(m.content) ? m.content[0].text : m.content;
      return `[${m.role}]: ${content}`;
    })
    .join('\n');

  // 用 AI（用便宜的 Haiku 模型）把旧消息压缩成摘要，节省成本
  const summaryResponse = await client.messages.create({
    model: 'claude-haiku-4-5', // 用便宜的模型做摘要
    max_tokens: 1000,
    messages: [{
      role: 'user',
      content: `请将以下对话历史压缩成一段简洁的摘要（200字以内），保留关键信息和决策：\n\n${oldText}`,
    }],
  });

  const summary = summaryResponse.content[0].text;
  console.log(`[上下文压缩] 完成，${oldMessages.length} 条消息 → ${summary.length} 字摘要`);

  // 把摘要注入为对话历史的第一条（作为背景信息）
  const compressedMessages = [
    {
      role: 'user',
      content: `[对话历史摘要]\n${summary}\n\n以上是之前对话的摘要，请基于此继续对话。`,
    },
    {
      role: 'assistant',
      content: '好的，我已了解之前的对话内容，请继续。',
    },
    ...recentMessages,
  ];

  return compressedMessages;
}

// ─────────────────────────────────────────────
// 4. TokenBudgetManager —— Token 预算管理类
// ─────────────────────────────────────────────

/**
 * Token 预算管理器
 *
 * 类比：旅行预算管理
 *   - 总预算 = 整个对话的 token 上限
 *   - 已花费 = 历史消耗
 *   - 剩余预算 = 还能说多少话
 *   - 每次消费前检查是否超支
 *   - 根据剩余预算动态调整 max_tokens（防止超支）
 */
class TokenBudgetManager {
  /**
   * @param {number} totalBudget  - 总 token 预算（默认 100000）
   * @param {number} reserveRatio - 为 AI 回复预留的比例（默认 20%）
   */
  constructor(totalBudget = 100000, reserveRatio = 0.2) {
    this.totalBudget = totalBudget;
    this.reserveRatio = reserveRatio;
    this.inputTokensUsed = 0;
    this.outputTokensUsed = 0;
    this.callCount = 0;
  }

  /** 已消耗的总 token 数 */
  get totalUsed() {
    return this.inputTokensUsed + this.outputTokensUsed;
  }

  /** 剩余可用 token 数 */
  get remaining() {
    return this.totalBudget - this.totalUsed;
  }

  /** 为 AI 输出预留的 token 数 */
  get reservedForOutput() {
    return Math.floor(this.totalBudget * this.reserveRatio);
  }

  /**
   * 当前推荐的 max_tokens（根据剩余预算动态调整）
   * 类比：购物剩余预算 —— 根据还剩多少钱决定能买多贵的东西
   */
  get recommendedMaxTokens() {
    const available = this.remaining - this.reservedForOutput;
    return Math.max(100, Math.min(available, 4096));
  }

  /**
   * 记录一次 API 调用的消耗
   * @param {object} usage - response.usage 对象
   */
  record(usage) {
    this.inputTokensUsed += usage.input_tokens ?? 0;
    this.outputTokensUsed += usage.output_tokens ?? 0;
    this.callCount++;
  }

  /** 检查是否已接近预算耗尽 */
  isExhausted() {
    return this.remaining < 500;
  }

  /** 打印当前预算状态 */
  printStatus() {
    const usedPercent = ((this.totalUsed / this.totalBudget) * 100).toFixed(1);
    console.log('\n[TokenBudgetManager 状态]');
    console.log(`  总预算:          ${this.totalBudget.toLocaleString()} tokens`);
    console.log(`  已消耗:          ${this.totalUsed.toLocaleString()} tokens（${usedPercent}%）`);
    console.log(`  剩余:            ${this.remaining.toLocaleString()} tokens`);
    console.log(`  API 调用次数:    ${this.callCount}`);
    console.log(`  推荐 max_tokens: ${this.recommendedMaxTokens}`);
    console.log(`  状态:            ${this.isExhausted() ? '⚠️ 接近耗尽' : '✅ 正常'}`);
  }
}

// ─────────────────────────────────────────────
// 主演示函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔════════════════════════════════════════╗');
  console.log('║  09_context_engineering.js  上下文工程 ║');
  console.log('╚════════════════════════════════════════╝');

  // --- 演示1：模块化 System Prompt 构建 ---
  console.log('\n--- 演示1：buildSystemPrompt(role, context, tools) ---');

  const systemPrompt = buildSystemPrompt(
    '技术支持专员',                         // role
    {                                        // context
      用户等级: 'VIP',
      当前版本: 'v2.3.1',
      平台: 'macOS',
    },
    [                                        // tools
      { name: 'search_docs', description: '搜索官方文档' },
      { name: 'create_ticket', description: '创建工单' },
    ],
    {                                        // options
      expertise: 'Node.js 和 AI 工具',
      constraints: ['回答长度不超过200字', '不讨论竞争对手产品'],
    }
  );
  console.log('生成的 System Prompt:\n');
  console.log(systemPrompt);

  // --- 演示2：用构建好的 prompt 发起请求 ---
  console.log('\n--- 演示2：使用模块化 Prompt 发起请求 ---');
  const budget = new TokenBudgetManager(50000);

  try {
    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: budget.recommendedMaxTokens,
      system: systemPrompt,
      messages: [{ role: 'user', content: '如何更新到最新版本？' }],
    });
    budget.record(response.usage);
    console.log('回复:', response.content[0].text);
    budget.printStatus();
  } catch (error) {
    console.error('请求失败:', error.message);
  }

  // --- 演示3：Prompt Caching ---
  try {
    await demoCaching();
  } catch (error) {
    console.error('Caching 演示失败:', error.message);
  }

  // --- 演示4：上下文压缩（模拟数据，不发实际请求）---
  console.log('\n--- 演示4：上下文压缩概念演示 ---');
  const mockMessages = Array.from({ length: 20 }, (_, i) => [
    { role: 'user', content: `这是第${i + 1}个问题：请介绍 Node.js 第${i + 1}个核心特性？` },
    { role: 'assistant', content: `Node.js 第${i + 1}个特性：异步非阻塞 I/O、事件驱动架构……（详细说明略）` },
  ]).flat();

  const estimated = estimateTokens(mockMessages);
  console.log(`模拟消息数: ${mockMessages.length}，估算 token 数: ${estimated}`);
  console.log(estimated > 5000
    ? '超过阈值 5000，实际场景中将触发 compressContext() 压缩旧消息'
    : '未超过阈值，无需压缩'
  );

  console.log('\n所有演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
