/**
 * 08_error_handling.js
 * ====================
 * 健壮的错误处理 —— 让你的 AI 应用不再因为一个网络抖动就崩掉
 *
 * 类比：就像 fetch 的 try/catch，但加了自动重试、超时保护、熔断器
 *
 * 涵盖内容：
 *   1. Anthropic SDK 错误类型捕获（APIError、RateLimitError 等）
 *   2. retryWithBackoff —— 指数退避 + jitter（随机抖动）
 *   3. AbortSignal.timeout() 超时控制
 *   4. CircuitBreaker 熔断器（简单实现）
 *   5. 结构化错误日志（JSON 格式）
 *   6. 综合示例：把以上技术组合使用
 *
 * 运行前：
 *   npm install @anthropic-ai/sdk
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 08_error_handling.js
 */

import Anthropic from '@anthropic-ai/sdk';

// 初始化客户端，关掉 SDK 内置重试，由我们自己的逻辑接管
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  maxRetries: 0,
});

// ─────────────────────────────────────────────
// 1. 结构化日志 —— JSON 格式，方便 ELK / Datadog 接入
// ─────────────────────────────────────────────

/**
 * 打印结构化日志（JSON 格式）
 *
 * 类比：普通日志像日记（"出错了"），结构化 JSON 日志像档案（时间、
 * 错误码、请求 ID 全都有），系统能自动解析、检索和告警。
 *
 * @param {'info'|'warn'|'error'} level - 日志级别
 * @param {string} event - 事件名（snake_case）
 * @param {object} data  - 附加上下文数据
 */
function log(level, event, data = {}) {
  const entry = {
    timestamp: new Date().toISOString(), // ISO 8601，便于排序
    level,
    event,
    ...data,
  };
  // 生产环境输出到 stdout，由日志收集器（如 Fluentd）采集
  console.log(JSON.stringify(entry));
}

// ─────────────────────────────────────────────
// 2. 错误类型识别 —— 不同错误，不同处理策略
// ─────────────────────────────────────────────

/**
 * 判断某个错误是否"值得重试"
 *
 * 类比：
 *   - 429（限流）→ 等一会儿再试，就像超市收银台排队
 *   - 529（过载）→ 服务器忙，稍等
 *   - 5xx          → 服务器临时故障，可以重试
 *   - 400          → 你传错参数了，重试没用
 *   - 401          → API Key 错了，重试没用
 *   - 网络断开    → 值得重试
 */
function isRetryable(error) {
  if (error instanceof Anthropic.APIError) {
    return error.status === 429 || (error.status >= 500 && error.status < 600);
  }
  // 网络层错误
  if (error instanceof Anthropic.APIConnectionError) return true;
  if (error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT') return true;
  return false;
}

/**
 * 把 SDK 错误翻译成人类可读的中文说明
 */
function describeError(error) {
  if (error instanceof Anthropic.AuthenticationError)
    return 'API Key 无效或已过期（401）';
  if (error instanceof Anthropic.PermissionDeniedError)
    return '没有权限访问该模型（403）';
  if (error instanceof Anthropic.NotFoundError)
    return '模型或资源不存在（404）';
  if (error instanceof Anthropic.RateLimitError)
    return '请求太频繁，触发 Rate Limit（429）';
  if (error instanceof Anthropic.InternalServerError)
    return `Anthropic 服务器内部错误（${error.status}）`;
  if (error instanceof Anthropic.APIConnectionError)
    return '网络连接失败，检查网络或代理';
  if (error instanceof Anthropic.APIError)
    return `API 错误 ${error.status}: ${error.message}`;
  if (error.name === 'AbortError' || error.name === 'TimeoutError')
    return '请求超时，已主动取消';
  return `未知错误: ${error.message}`;
}

// ─────────────────────────────────────────────
// 3. retryWithBackoff —— 指数退避 + jitter
// ─────────────────────────────────────────────

/**
 * 带指数退避重试的包装函数
 *
 * 什么是"指数退避"？
 *   第1次失败 → 等 1s  再试
 *   第2次失败 → 等 2s  再试
 *   第3次失败 → 等 4s  再试
 *   类比：堵车时先等1分钟，还堵等2分钟，再堵等4分钟……
 *
 * 什么是"jitter（抖动）"？
 *   100个请求同时失败，同时在第2秒重试 → 又一起打垮服务器
 *   加随机抖动后，大家错开重试 → 服务器压力被分散
 *   类比：超市开了新收银台，大家别一起冲，错开几秒更好
 *
 * @param {Function} fn          - 要执行的异步函数
 * @param {number}   maxRetries  - 最大重试次数（默认 3）
 * @param {number}   baseDelay   - 基础等待时间（ms，默认 1000）
 * @param {number}   maxDelay    - 最大等待时间（ms，默认 30000）
 */
async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000, maxDelay = 30000) {
  let lastError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const result = await fn();
      if (attempt > 0) {
        log('info', 'retry_succeeded', { attempt });
      }
      return result;
    } catch (error) {
      lastError = error;

      // 最后一次尝试失败，直接抛出
      if (attempt === maxRetries) {
        log('error', 'max_retries_exceeded', {
          attempt,
          maxRetries,
          error: describeError(error),
        });
        throw error;
      }

      // 不值得重试的错误，立刻抛出
      if (!isRetryable(error)) {
        log('error', 'non_retryable_error', {
          error: describeError(error),
          status: error.status ?? null,
        });
        throw error;
      }

      // 计算等待时间：2^attempt * baseDelay + 随机 jitter
      const exponentialDelay = Math.pow(2, attempt) * baseDelay;
      const jitter = Math.random() * baseDelay;
      let delay = Math.min(exponentialDelay + jitter, maxDelay);

      // 如果是 429，优先使用服务器 Retry-After 头
      const retryAfterHeader = error.headers?.['retry-after'];
      if (retryAfterHeader) {
        delay = parseFloat(retryAfterHeader) * 1000;
        log('info', 'using_server_retry_after', { delayMs: delay });
      }

      log('warn', 'retry_scheduled', {
        attempt: attempt + 1,
        maxRetries,
        waitMs: Math.round(delay),
        reason: describeError(error),
      });

      await sleep(delay);
    }
  }

  throw lastError;
}

// 工具函数：等待指定毫秒
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ─────────────────────────────────────────────
// 4. AbortSignal.timeout() —— 超时控制
// ─────────────────────────────────────────────

/**
 * 带超时的 API 调用
 *
 * 为什么需要超时？
 *   服务器有时候卡住，一直不返回。不设超时的话程序会永远等下去。
 *   类比：打电话给客服，超过3分钟没人接就挂掉重拨。
 *
 * AbortSignal.timeout() 是 Node.js 17.3+ 的原生 API，
 * 无需第三方库，超时后自动抛出 AbortError / TimeoutError。
 *
 * @param {string} message    - 用户消息
 * @param {number} timeoutMs  - 超时时间（ms，默认 15000）
 */
async function callWithTimeout(message, timeoutMs = 15000) {
  log('info', 'api_call_start', { timeoutMs });

  try {
    const response = await client.messages.create(
      {
        model: 'claude-opus-4-5',
        max_tokens: 256,
        messages: [{ role: 'user', content: message }],
      },
      {
        signal: AbortSignal.timeout(timeoutMs), // 超时信号传给 SDK
      }
    );

    log('info', 'api_call_success', {
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    });

    return response.content[0].text;
  } catch (error) {
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      log('error', 'request_timeout', { timeoutMs });
      throw new Error(`请求超过 ${timeoutMs}ms 超时`);
    }
    throw error;
  }
}

// ─────────────────────────────────────────────
// 5. CircuitBreaker 熔断器
// ─────────────────────────────────────────────

/**
 * 熔断器 —— 防止"雪崩效应"
 *
 * 三种状态：
 *   CLOSED（闭合）  → 正常工作，请求直接通过
 *   OPEN（断开）    → 服务不可用，所有请求快速失败
 *   HALF_OPEN（半开）→ 放一个探测请求进去，成功则恢复 CLOSED
 *
 * 类比：家里的电路保险丝
 *   - 正常：电流通过（CLOSED）
 *   - 过载：保险丝断开，避免电线起火（OPEN）
 *   - 修复后：先试接一路，确认安全再全部恢复（HALF_OPEN → CLOSED）
 *
 * 作用：下游服务故障时快速失败，不让调用方一直等超时，保护整个系统
 */
class CircuitBreaker {
  /**
   * @param {number} failureThreshold - 连续失败多少次触发熔断（默认 5）
   * @param {number} recoveryTime     - 熔断后多久进入 HALF_OPEN（ms，默认 60000）
   * @param {string} name             - 熔断器名称（用于日志）
   */
  constructor(failureThreshold = 5, recoveryTime = 60000, name = 'default') {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.failureThreshold = failureThreshold;
    this.recoveryTime = recoveryTime;
    this.lastFailureTime = null;
    this.name = name;
  }

  /**
   * 通过熔断器执行一个函数
   */
  async execute(fn) {
    if (this.state === 'OPEN') {
      const elapsed = Date.now() - this.lastFailureTime;
      if (elapsed < this.recoveryTime) {
        const remaining = Math.ceil((this.recoveryTime - elapsed) / 1000);
        throw new Error(`[熔断器:${this.name}] 开路中，${remaining}s 后再试`);
      }
      // 超过恢复时间，进入半开状态
      this.state = 'HALF_OPEN';
      log('info', 'circuit_half_open', { name: this.name });
    }

    try {
      const result = await fn();
      this._onSuccess();
      return result;
    } catch (error) {
      this._onFailure();
      throw error;
    }
  }

  _onSuccess() {
    const wasHalfOpen = this.state === 'HALF_OPEN';
    this.failureCount = 0;
    this.state = 'CLOSED';
    if (wasHalfOpen) {
      log('info', 'circuit_closed', { name: this.name, reason: '探测请求成功，恢复正常' });
    }
  }

  _onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.state === 'HALF_OPEN' || this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      log('error', 'circuit_opened', {
        name: this.name,
        failureCount: this.failureCount,
        recoveryTimeMs: this.recoveryTime,
      });
    }
  }

  getStatus() {
    return {
      name: this.name,
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
        ? new Date(this.lastFailureTime).toISOString()
        : null,
    };
  }
}

// ─────────────────────────────────────────────
// 6. 综合示例 —— 把所有技术组合在一起
// ─────────────────────────────────────────────

/**
 * 生产级别的 API 调用函数
 * 集成：结构化日志 + 指数退避重试 + 超时 + 断路器
 *
 * 类比：就像一辆有安全带、安全气囊、防抱死制动的汽车，
 * 多层防护叠加，才能真正保证稳定。
 */
async function robustApiCall(prompt, options = {}) {
  const {
    maxTokens = 256,
    maxRetries = 3,
    timeoutMs = 20000,
    circuitBreaker = null,
    context = {},
  } = options;

  const start = Date.now();

  // 单次 API 调用（带超时）
  const callOnce = async () => {
    try {
      const response = await client.messages.create(
        {
          model: 'claude-opus-4-5',
          max_tokens: maxTokens,
          messages: [{ role: 'user', content: prompt }],
        },
        { signal: AbortSignal.timeout(timeoutMs) }
      );

      log('info', 'api_call_success', {
        durationMs: Date.now() - start,
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        ...context,
      });

      return response.content[0].text;
    } catch (error) {
      log('error', 'api_call_failed', {
        error: describeError(error),
        retryable: isRetryable(error),
        durationMs: Date.now() - start,
        ...context,
      });
      throw error;
    }
  };

  // 带重试的调用
  const callWithRetry = () => retryWithBackoff(callOnce, maxRetries, 500);

  // 如果提供了熔断器，用熔断器包裹
  if (circuitBreaker) {
    return circuitBreaker.execute(callWithRetry);
  }

  return callWithRetry();
}

// ─────────────────────────────────────────────
// 主函数：依次运行所有示例
// ─────────────────────────────────────────────
async function main() {
  console.log('╔═══════════════════════════════════════╗');
  console.log('║  08_error_handling.js  错误处理示例   ║');
  console.log('╚═══════════════════════════════════════╝\n');

  // 创建共享熔断器
  const breaker = new CircuitBreaker(5, 60000, 'AnthropicAPI');

  // 示例1：正常调用（带完整保护链）
  console.log('--- 示例1：带超时 + 重试 + 熔断的正常调用 ---');
  try {
    const result = await robustApiCall('用一句话解释什么是错误处理？', {
      maxTokens: 80,
      timeoutMs: 20000,
      maxRetries: 2,
      circuitBreaker: breaker,
      context: { userId: 'demo_user', task: 'explain_error_handling' },
    });
    console.log('回复:', result);
  } catch (error) {
    console.log('调用失败:', describeError(error));
  }

  // 示例2：查看熔断器状态
  console.log('\n--- 示例2：熔断器当前状态 ---');
  console.log(JSON.stringify(breaker.getStatus(), null, 2));

  // 示例3：演示错误类型识别
  console.log('\n--- 示例3：错误类型识别 ---');
  try {
    // 故意传入无效 max_tokens 触发参数错误
    await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: -1, // 无效值
      messages: [{ role: 'user', content: 'test' }],
    });
  } catch (error) {
    console.log('错误类型:', error.constructor.name);
    console.log('中文说明:', describeError(error));
    console.log('是否可重试:', isRetryable(error));
  }

  console.log('\n所有示例运行完毕。');
}

main().catch(err => {
  log('error', 'unhandled_exception', { error: err.message });
  process.exit(1);
});
