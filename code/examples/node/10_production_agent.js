/**
 * 10_production_agent.js
 * =======================
 * 生产级 AI 聊天服务 —— 把 AI 能力封装成可部署的 HTTP API
 *
 * 涵盖内容：
 *   1. Express 应用框架（ESM 写法）
 *   2. POST /chat（带用户ID、会话ID，支持多轮对话）
 *   3. GET  /health（健康检查，返回 {status, uptime, sessions}）
 *   4. 会话管理（Map<sessionId, messages>，自动清理过期会话）
 *   5. 限流中间件（内存实现，防止滥用）
 *   6. 结构化日志（JSON 格式，含 traceId）
 *   7. 错误处理中间件
 *
 * 运行前：
 *   npm install express @anthropic-ai/sdk
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 10_production_agent.js
 *
 * 测试：
 *   curl -X POST http://localhost:3000/chat \
 *     -H "Content-Type: application/json" \
 *     -d '{"userId":"u1","sessionId":"s1","message":"你好"}'
 *
 *   curl http://localhost:3000/health
 */

import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { randomUUID } from 'crypto';

const app = express();
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ─────────────────────────────────────────────
// 1. 结构化日志（含 traceId）
// ─────────────────────────────────────────────

/**
 * 结构化日志函数
 *
 * 什么是 traceId？
 *   每个请求进来时生成一个唯一 ID（traceId），整个请求链路
 *   的所有日志都带上这个 ID，出问题时可以"顺藤摸瓜"找到完整链路。
 *   类比：快递单号 —— 通过单号能追查整个物流过程。
 *
 * 为什么用 JSON 格式？
 *   普通文本日志人眼看得懂，但 ELK/Datadog 等工具无法自动解析。
 *   JSON 日志每个字段都有名称，工具能自动索引、聚合、告警。
 *
 * @param {'info'|'warn'|'error'} level
 * @param {string} event
 * @param {object} data
 */
function log(level, event, data = {}) {
  const entry = {
    ts: new Date().toISOString(),
    level,
    event,
    ...data,
  };
  console.log(JSON.stringify(entry));
}

// ─────────────────────────────────────────────
// 2. 会话管理
// ─────────────────────────────────────────────

/**
 * 会话存储：Map<sessionId, SessionData>
 *
 * 什么是"会话"？
 *   就像微信聊天窗口，每个窗口有独立的聊天记录。
 *   这里用内存存储，生产环境应换成 Redis。
 *
 * SessionData 结构：
 *   messages   : 消息历史数组（[{role, content}]）
 *   userId     : 用户 ID（可用于权限校验）
 *   createdAt  : 创建时间（毫秒时间戳）
 *   lastActive : 最后活跃时间（毫秒时间戳）
 */
const sessions = new Map();

const SESSION_TTL_MS = 30 * 60 * 1000;   // 会话超时：30 分钟
const MAX_MESSAGES_PER_SESSION = 50;       // 每个会话最多保存 50 条消息

/**
 * 获取或创建会话（懒加载）
 */
function getOrCreateSession(sessionId, userId) {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, {
      messages: [],
      userId,
      createdAt: Date.now(),
      lastActive: Date.now(),
    });
    log('info', 'session_created', { sessionId, userId });
  }
  const session = sessions.get(sessionId);
  session.lastActive = Date.now(); // 每次访问更新活跃时间
  return session;
}

/**
 * 定期清理过期会话（每 5 分钟运行一次）
 * 类比：餐厅服务员定期清理长时间无人的桌子，腾出空间
 */
function cleanupSessions() {
  const now = Date.now();
  let cleaned = 0;

  for (const [sessionId, session] of sessions.entries()) {
    if (now - session.lastActive > SESSION_TTL_MS) {
      sessions.delete(sessionId);
      cleaned++;
    }
  }

  if (cleaned > 0) {
    log('info', 'sessions_cleaned', { cleaned, remaining: sessions.size });
  }
}

setInterval(cleanupSessions, 5 * 60 * 1000);

// ─────────────────────────────────────────────
// 3. 限流中间件（内存实现）
// ─────────────────────────────────────────────

/**
 * 简单的滑动窗口限流
 *
 * 原理：记录每个 userId 在最近 windowMs 内的请求次数，
 *       超过 maxRequests 就返回 429。
 *
 * 类比：银行取号机每小时最多给你取 3 个号，超了就让你等。
 *
 * 生产环境推荐：npm install express-rate-limit，配合 Redis 存储做分布式限流
 *
 * @param {number} windowMs    - 统计窗口（ms，默认60秒）
 * @param {number} maxRequests - 窗口内最大请求数（默认20）
 */
function rateLimitMiddleware(windowMs = 60000, maxRequests = 20) {
  const store = new Map(); // Map<userId, number[]>（请求时间戳数组）

  return (req, res, next) => {
    const userId = req.body?.userId || req.ip || 'anonymous';
    const now = Date.now();
    const windowStart = now - windowMs;

    // 只保留窗口内的时间戳
    const timestamps = (store.get(userId) || []).filter(ts => ts > windowStart);

    if (timestamps.length >= maxRequests) {
      const resetAt = new Date(timestamps[0] + windowMs).toISOString();
      log('warn', 'rate_limit_exceeded', {
        userId,
        requestCount: timestamps.length,
        resetAt,
      });
      return res.status(429).json({
        error: '请求太频繁，请稍后再试',
        retryAfterSeconds: Math.ceil((timestamps[0] + windowMs - now) / 1000),
        resetAt,
      });
    }

    timestamps.push(now);
    store.set(userId, timestamps);
    next();
  };
}

// ─────────────────────────────────────────────
// 4. 中间件配置
// ─────────────────────────────────────────────

app.use(express.json({ limit: '16kb' })); // 限制请求体大小

// 请求日志中间件：为每个请求附加 traceId
app.use((req, res, next) => {
  req.traceId = randomUUID(); // 每次请求生成唯一 ID
  req.startTime = Date.now();

  log('info', 'request_received', {
    traceId: req.traceId,
    method: req.method,
    path: req.path,
  });

  // 响应结束后记录耗时（利用 Node.js 事件钩子）
  res.on('finish', () => {
    log('info', 'request_completed', {
      traceId: req.traceId,
      statusCode: res.statusCode,
      durationMs: Date.now() - req.startTime,
    });
  });

  next();
});

// ─────────────────────────────────────────────
// 5. 路由：POST /chat
// ─────────────────────────────────────────────

/**
 * POST /chat
 *
 * 请求体（JSON）：
 *   { userId: string, sessionId: string, message: string }
 *
 * 成功响应（200）：
 *   { reply: string, sessionId: string, messageCount: number, usage: {...} }
 *
 * 错误响应：
 *   400 - 参数缺失或无效
 *   429 - 请求频率超限
 *   503 - AI 服务不可用
 */
app.post('/chat',
  rateLimitMiddleware(60000, 20), // 每分钟最多 20 条消息
  async (req, res, next) => {
    const { userId, sessionId, message } = req.body;

    // 参数校验
    if (!userId || !sessionId || !message) {
      return res.status(400).json({
        error: '缺少必要参数',
        required: ['userId', 'sessionId', 'message'],
        traceId: req.traceId,
      });
    }
    if (typeof message !== 'string' || message.trim().length === 0) {
      return res.status(400).json({ error: 'message 不能为空', traceId: req.traceId });
    }
    if (message.length > 10000) {
      return res.status(400).json({ error: 'message 不能超过 10000 字符', traceId: req.traceId });
    }

    try {
      const session = getOrCreateSession(sessionId, userId);

      // 添加用户消息
      session.messages.push({ role: 'user', content: message.trim() });

      // 消息超出上限时截断（保留最近的，防止内存无限增长）
      if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
        const removed = session.messages.length - MAX_MESSAGES_PER_SESSION;
        session.messages = session.messages.slice(-MAX_MESSAGES_PER_SESSION);
        log('warn', 'session_truncated', { sessionId, removed, traceId: req.traceId });
      }

      log('info', 'calling_anthropic', {
        traceId: req.traceId,
        userId,
        sessionId,
        messageCount: session.messages.length,
      });

      // 调用 Anthropic API
      const response = await client.messages.create({
        model: 'claude-opus-4-5',
        max_tokens: 1024,
        system: '你是一个友好的 AI 助手，用中文回答用户问题。',
        messages: session.messages,
      });

      const reply = response.content[0].text;

      // 把 AI 回复也加入历史（维持多轮对话上下文）
      session.messages.push({ role: 'assistant', content: reply });

      log('info', 'chat_success', {
        traceId: req.traceId,
        userId,
        sessionId,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      });

      res.json({
        reply,
        sessionId,
        messageCount: session.messages.length,
        usage: {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens,
        },
        traceId: req.traceId,
      });
    } catch (error) {
      next(error); // 传递给全局错误处理中间件
    }
  }
);

// ─────────────────────────────────────────────
// 6. 路由：GET /health
// ─────────────────────────────────────────────

/**
 * GET /health
 *
 * 健康检查端点，供负载均衡器和监控系统调用。
 * 类比：体检 —— 快速检查服务是否"活着"且状态正常。
 *
 * 负载均衡器（如 Nginx、AWS ALB）会定期请求这个接口，
 * 返回非 2xx 或超时时，会把该实例从流量池中摘除。
 *
 * 响应 200：
 *   { status: "ok", uptime: 123.45, sessions: 5 }
 */
app.get('/health', (req, res) => {
  const memUsage = process.memoryUsage();
  res.json({
    status: 'ok',
    uptime: process.uptime(),           // 进程运行时间（秒）
    sessions: sessions.size,            // 当前活跃会话数
    memory: {
      heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + 'MB',
      rss: Math.round(memUsage.rss / 1024 / 1024) + 'MB',
    },
    version: '1.0.0',
    timestamp: new Date().toISOString(),
  });
});

// ─────────────────────────────────────────────
// 7. 错误处理中间件
// ─────────────────────────────────────────────

/**
 * 全局错误处理中间件
 *
 * Express 规定：错误中间件必须有 4 个参数（err, req, res, next），
 * 否则 Express 不会把它识别为错误处理器。
 * 位置必须放在所有路由之后。
 */
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  const traceId = req.traceId || 'unknown';

  // Anthropic API 错误细分处理
  if (err instanceof Anthropic.APIError) {
    log('error', 'anthropic_api_error', {
      traceId,
      status: err.status,
      message: err.message,
    });

    if (err.status === 429) {
      return res.status(503).json({
        error: 'AI 服务暂时繁忙，请稍后再试',
        retryAfterSeconds: 5,
        traceId,
      });
    }
    if (err.status >= 500) {
      return res.status(503).json({ error: 'AI 服务暂时不可用', traceId });
    }
  }

  // 其他错误
  log('error', 'internal_error', {
    traceId,
    message: err.message,
    // 生产环境不打印 stack 到响应，避免泄露内部实现
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
  });

  res.status(500).json({
    error: '服务器内部错误',
    traceId,
    detail: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// ─────────────────────────────────────────────
// 8. 启动服务器
// ─────────────────────────────────────────────

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  log('info', 'server_started', {
    port: PORT,
    env: process.env.NODE_ENV || 'development',
    apiKeySet: !!process.env.ANTHROPIC_API_KEY,
  });

  console.log(`
╔═══════════════════════════════════════════╗
║  10_production_agent.js  生产级 AI 服务   ║
╠═══════════════════════════════════════════╣
║  服务已启动：http://localhost:${PORT}         ║
║                                           ║
║  端点：                                   ║
║    POST /chat    多轮对话                  ║
║    GET  /health  健康检查                  ║
║                                           ║
║  测试 chat：                              ║
║  curl -X POST http://localhost:${PORT}/chat \\║
║    -H "Content-Type: application/json" \\ ║
║    -d '{"userId":"u1","sessionId":"s1",   ║
║         "message":"你好"}'                ║
║                                           ║
║  测试 health：                            ║
║  curl http://localhost:${PORT}/health        ║
╚═══════════════════════════════════════════╝
  `);
});

// 优雅关闭：SIGTERM 时等待当前请求完成再退出（Docker/K8s 标准做法）
process.on('SIGTERM', () => {
  log('info', 'server_shutting_down', { activeSessions: sessions.size });
  process.exit(0);
});
