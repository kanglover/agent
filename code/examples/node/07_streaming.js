// 运行: node 07_streaming.js
// 依赖: npm install @anthropic-ai/sdk express
//
// 流式输出（Streaming）完整指南
//
// 为什么需要流式？
//   非流式：等 Claude 全部生成完（可能要 10-30 秒）才看到内容 → 体验差
//   流式：  Claude 生成一个字你就立刻看到一个字 → 像"打字机"一样实时显示
//
// 三种使用场景：
//   1. 命令行：直接 process.stdout.write 打印
//   2. Express SSE 接口：服务端推事件（Server-Sent Events），前端用 EventSource 接收
//   3. 工具调用中的 streaming：工具调用结果出来后，AI 分析部分继续流式输出
//
// SSE（Server-Sent Events）类比：
//   普通 HTTP：你发一个请求，服务器给一个响应，连接关闭
//   SSE：你发一个请求，服务器持续发送数据，连接保持直到完成
//   就像订阅了一个"直播频道"，服务器会不断推送内容给你

import Anthropic from '@anthropic-ai/sdk';
import express from 'express';

const client = new Anthropic();
const app = express();
app.use(express.json());

// ═══════════════════════════════════════════════════════════════
// 第一节：基础 Stream API（for await...of）
// ═══════════════════════════════════════════════════════════════

/**
 * 最简单的流式输出示例
 * for await...of 语法：每当 Claude 生成新内容，循环就执行一次
 *
 * @param {string} prompt
 * @returns {Promise<string>} - 完整的回复文本
 */
async function basicStreaming(prompt) {
  console.log('\n=== 基础流式输出 ===');
  console.log(`提问: ${prompt}\n`);
  console.log('Claude 回答（实时显示）：\n');

  let fullText = '';

  // client.messages.stream() 返回一个异步可迭代对象
  // 每次 for await 迭代，都会拿到一个 SSE 事件
  const stream = client.messages.stream({
    model: 'claude-opus-4-5',
    max_tokens: 512,
    messages: [{ role: 'user', content: prompt }],
  });

  for await (const event of stream) {
    // 事件类型说明：
    // - 'message_start'        ：流开始，包含 message 基本信息
    // - 'content_block_start'  ：一个内容块开始（文本块或工具调用块）
    // - 'content_block_delta'  ：内容增量（有新字符可读）
    //   - delta.type = 'text_delta'       → 普通文字
    //   - delta.type = 'input_json_delta' → 工具调用参数（JSON 片段）
    // - 'content_block_stop'   ：一个内容块结束
    // - 'message_delta'        ：消息级别的增量（stop_reason 等）
    // - 'message_stop'         ：整条消息结束

    if (
      event.type === 'content_block_delta' &&
      event.delta.type === 'text_delta'
    ) {
      process.stdout.write(event.delta.text); // 打字机效果
      fullText += event.delta.text;
    }
  }

  // 流结束后，可以获取完整的最终 response 对象
  const finalMessage = await stream.getFinalMessage();
  console.log(`\n\n[流结束] token 用量: 输入=${finalMessage.usage.input_tokens}, 输出=${finalMessage.usage.output_tokens}`);

  return fullText;
}

// ═══════════════════════════════════════════════════════════════
// 第二节：流式输出 + 工具调用混合
// 工具调用时参数以 JSON 片段的形式流式传输
// ═══════════════════════════════════════════════════════════════

/**
 * 带工具调用的流式输出
 * 工具参数也是逐步流式传输的（input_json_delta），最终拼接成完整 JSON
 *
 * @param {string} prompt
 */
async function streamingWithTools(prompt) {
  console.log('\n=== 工具调用 + 流式输出 ===');
  console.log(`提问: ${prompt}\n`);

  const tools = [{
    name: 'calculate',
    description: '执行数学计算',
    input_schema: {
      type: 'object',
      properties: {
        expression: { type: 'string', description: '数学表达式' },
      },
      required: ['expression'],
    },
  }];

  const messages = [{ role: 'user', content: prompt }];
  let iterationCount = 0;

  while (iterationCount < 5) {
    iterationCount++;
    console.log(`\n[第 ${iterationCount} 轮]`);

    const stream = client.messages.stream({
      model: 'claude-opus-4-5',
      max_tokens: 512,
      tools,
      messages,
    });

    // 用于收集工具调用参数（JSON 片段会逐步累积）
    const toolCallAccumulator = {}; // { toolUseId → { name, jsonBuffer } }
    let currentToolUseId = null;
    let hasTextOutput = false;

    for await (const event of stream) {
      switch (event.type) {
        case 'content_block_start':
          if (event.content_block.type === 'tool_use') {
            // 工具调用块开始
            currentToolUseId = event.content_block.id;
            toolCallAccumulator[currentToolUseId] = {
              name: event.content_block.name,
              jsonBuffer: '',  // 用于拼接 JSON 片段
            };
            console.log(`🔧 Claude 要调用工具: ${event.content_block.name}`);
          } else if (event.content_block.type === 'text') {
            if (!hasTextOutput) {
              process.stdout.write('Claude 分析（流式）：');
              hasTextOutput = true;
            }
          }
          break;

        case 'content_block_delta':
          if (event.delta.type === 'text_delta') {
            // 普通文字增量
            process.stdout.write(event.delta.text);
          } else if (event.delta.type === 'input_json_delta') {
            // 工具参数的 JSON 片段，需要累积拼接
            if (currentToolUseId && toolCallAccumulator[currentToolUseId]) {
              toolCallAccumulator[currentToolUseId].jsonBuffer += event.delta.partial_json;
            }
          }
          break;

        case 'content_block_stop':
          currentToolUseId = null; // 当前块结束，重置
          break;
      }
    }

    if (hasTextOutput) console.log(); // 换行

    const finalMessage = await stream.getFinalMessage();

    if (finalMessage.stop_reason === 'end_turn') {
      // 没有工具调用，对话结束
      break;
    }

    if (finalMessage.stop_reason === 'tool_use') {
      // 把 Claude 的回复加入历史
      messages.push({ role: 'assistant', content: finalMessage.content });

      // 执行工具调用并收集结果
      const toolResults = [];
      for (const block of finalMessage.content) {
        if (block.type !== 'tool_use') continue;

        // 解析工具参数（流式传输时是累积的 JSON 字符串，这里直接用 block.input）
        console.log(`  执行 ${block.name}(${JSON.stringify(block.input)})`);

        let result;
        if (block.name === 'calculate') {
          try {
            const sanitized = block.input.expression.replace(/[^0-9+\-*/().% ]/g, '');
            // eslint-disable-next-line no-new-func
            const calcResult = new Function(`return ${sanitized}`)();
            result = { result: calcResult, success: true };
          } catch {
            result = { error: '计算失败', success: false };
          }
        }

        console.log(`  结果: ${JSON.stringify(result)}`);
        toolResults.push({
          type: 'tool_result',
          tool_use_id: block.id,
          content: JSON.stringify(result),
        });
      }

      messages.push({ role: 'user', content: toolResults });
      continue; // 继续下一轮（Claude 会根据工具结果继续流式输出）
    }

    break;
  }
}

// ═══════════════════════════════════════════════════════════════
// 第三节：AbortController 中断流
// 类比 fetch 的 AbortController：可以随时取消正在进行的请求
// ═══════════════════════════════════════════════════════════════

/**
 * 演示如何在 3 秒后强制中断流式输出
 * 实际使用场景：用户点击"停止"按钮
 */
async function streamingWithAbort() {
  console.log('\n=== AbortController 中断演示 ===');
  console.log('将在 1 秒后中断输出...\n');

  // AbortController 就像一个"遥控停止按钮"
  // controller.abort() 相当于按下停止
  const controller = new AbortController();

  // 设置 1 秒后自动中断（模拟用户点击停止）
  const timer = setTimeout(() => {
    console.log('\n\n[中断信号发出]');
    controller.abort();
  }, 1000);

  try {
    const stream = client.messages.stream(
      {
        model: 'claude-opus-4-5',
        max_tokens: 1024,
        messages: [{
          role: 'user',
          content: '请详细介绍一下人工智能的发展历史，从 1950 年代开始。',
        }],
      },
      // 把 AbortSignal 传给 SDK
      { signal: controller.signal }
    );

    process.stdout.write('Claude 输出：');
    let charCount = 0;

    for await (const event of stream) {
      if (
        event.type === 'content_block_delta' &&
        event.delta.type === 'text_delta'
      ) {
        process.stdout.write(event.delta.text);
        charCount += event.delta.text.length;
      }
    }

    clearTimeout(timer);
    console.log(`\n\n[正常结束] 共输出 ${charCount} 个字符`);

  } catch (error) {
    clearTimeout(timer);
    if (error.name === 'AbortError' || error.message?.includes('abort')) {
      console.log('\n[流已成功中断] ✅');
    } else {
      console.error('\n[发生错误]', error.message);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// 第四节：Express SSE 接口
// SSE（Server-Sent Events）= 服务端推送事件
// ═══════════════════════════════════════════════════════════════

/**
 * POST /stream
 * 接收问题，以 SSE 格式流式返回 Claude 的回答
 *
 * SSE 协议格式：
 *   data: {"type":"delta","text":"你"}\n\n
 *   data: {"type":"delta","text":"好"}\n\n
 *   data: {"type":"done","fullText":"你好..."}\n\n
 *
 * 前端消费代码（作为注释展示）：
 * ──────────────────────────────────────────
 * const evtSource = new EventSource('/stream?question=...');  // GET 版本
 *
 * // 或者 POST + fetchEventSource（需要 @microsoft/fetch-event-source 包）：
 * import { fetchEventSource } from '@microsoft/fetch-event-source';
 * fetchEventSource('/stream', {
 *   method: 'POST',
 *   headers: { 'Content-Type': 'application/json' },
 *   body: JSON.stringify({ question: '你好' }),
 *   onmessage(event) {
 *     const data = JSON.parse(event.data);
 *     if (data.type === 'delta') {
 *       document.getElementById('output').textContent += data.text;
 *     }
 *     if (data.type === 'done') {
 *       console.log('完成！完整文本：', data.fullText);
 *     }
 *     if (data.type === 'error') {
 *       console.error('错误：', data.message);
 *     }
 *   },
 * });
 * ──────────────────────────────────────────
 */
app.post('/stream', async (req, res) => {
  const { question, systemPrompt } = req.body;

  if (!question) {
    return res.status(400).json({ error: '请提供 question 字段' });
  }

  // ── 设置 SSE 响应头 ────────────────────────────────────────
  // Content-Type: text/event-stream  → 告诉浏览器这是 SSE
  // Cache-Control: no-cache          → 禁止缓存（SSE 必须实时）
  // Connection: keep-alive           → 保持长连接
  // X-Accel-Buffering: no            → 禁止 Nginx 缓冲（重要！）
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.setHeader('Access-Control-Allow-Origin', '*'); // 允许跨域
  res.flushHeaders(); // 立即发送响应头

  /**
   * 发送一个 SSE 事件
   * SSE 格式：每条事件以 "data: " 开头，以两个换行符结尾
   * @param {object} data
   */
  const sendEvent = (data) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  // 监听客户端断开连接
  let abortController = new AbortController();
  req.on('close', () => {
    console.log('客户端断开连接，中断流');
    abortController.abort();
  });

  let fullText = '';

  try {
    const streamParams = {
      model: 'claude-opus-4-5',
      max_tokens: 1024,
      messages: [{ role: 'user', content: question }],
    };

    if (systemPrompt) {
      streamParams.system = systemPrompt;
    }

    const stream = client.messages.stream(
      streamParams,
      { signal: abortController.signal }
    );

    // 发送流式事件
    for await (const event of stream) {
      if (
        event.type === 'content_block_delta' &&
        event.delta.type === 'text_delta'
      ) {
        fullText += event.delta.text;
        // 把每个文字增量发送给前端
        sendEvent({ type: 'delta', text: event.delta.text });
      }
    }

    // 流结束，发送完成事件
    const finalMessage = await stream.getFinalMessage();
    sendEvent({
      type: 'done',
      fullText,
      usage: finalMessage.usage,
      stopReason: finalMessage.stop_reason,
    });

  } catch (error) {
    if (error.name !== 'AbortError') {
      sendEvent({ type: 'error', message: error.message });
    }
  } finally {
    res.end(); // 关闭 SSE 连接
  }
});

/**
 * GET /stream/health - 健康检查
 */
app.get('/stream/health', (req, res) => {
  res.json({ status: 'ok', message: 'Streaming 服务正常运行' });
});

// ═══════════════════════════════════════════════════════════════
// 主函数：启动服务 + 运行命令行示例
// ═══════════════════════════════════════════════════════════════

const PORT = 3007;

async function main() {
  console.log('🌊 Streaming 完整示例\n');

  try {
    // 运行命令行流式示例
    await basicStreaming('用三点描述量子计算机与传统计算机的区别，每点一句话。');
    await streamingWithTools('帮我计算 (999 * 1001 - 1) / 2 的结果，精确值。');
    await streamingWithAbort();

    // 启动 Express SSE 服务
    app.listen(PORT, () => {
      console.log(`\n🚀 SSE 服务启动：http://localhost:${PORT}`);
      console.log('\n测试命令：');
      console.log(`curl -X POST http://localhost:${PORT}/stream \\`);
      console.log(`  -H "Content-Type: application/json" \\`);
      console.log(`  -d '{"question": "用一句话解释什么是 SSE"}' \\`);
      console.log(`  --no-buffer`);
      console.log('\n按 Ctrl+C 停止服务');
    });

  } catch (error) {
    console.error('❌ 错误：', error.message);
    process.exit(1);
  }
}

main();
