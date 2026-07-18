// 运行: node 01_hello_claude.js
// 依赖: npm install @anthropic-ai/sdk
//
// 这个文件演示了与 Claude API 交互的最基础用法。
// 可以把它理解成：
//   就像 fetch('/api/chat', { method: 'POST', body: JSON.stringify({messages}) })
//   只是 Anthropic SDK 帮你封装了细节，让你专注写对话逻辑。

import Anthropic from '@anthropic-ai/sdk';

// ─────────────────────────────────────────────
// 初始化客户端
// SDK 会自动读取环境变量 ANTHROPIC_API_KEY
// 也可以手动传入：new Anthropic({ apiKey: '...' })
// ─────────────────────────────────────────────
const client = new Anthropic();

// ═══════════════════════════════════════════════════════════════
// 第一节：最简单的一次对话（单轮）
// ═══════════════════════════════════════════════════════════════

/**
 * 最基础的 API 调用示例
 * 对应 HTTP POST /v1/messages，SDK 帮你处理 headers / JSON 序列化
 *
 * @returns {Promise<void>}
 */
async function basicChat() {
  console.log('=== 1. 基础单轮对话 ===\n');

  // messages.create 是核心方法，类似 fetch 的作用
  const response = await client.messages.create({
    model: 'claude-opus-4-5',          // 使用哪个模型
    max_tokens: 1024,                   // 最多生成多少 token（大约 750 个英文单词）
    messages: [
      {
        role: 'user',                   // 角色：user = 你说的话
        content: '用一句话解释什么是 API？',
      },
    ],
  });

  // ─── 解析 response 结构 ────────────────────────────────────
  // response.id          - 本次请求的唯一 ID，方便排查问题
  // response.type        - 固定是 'message'
  // response.role        - 固定是 'assistant'
  // response.model       - 实际使用的模型名称
  // response.stop_reason - 为什么停止：'end_turn' / 'max_tokens' / 'tool_use'
  // response.usage       - token 用量统计
  //   .input_tokens      - 你发送的 token 数
  //   .output_tokens     - Claude 回复的 token 数
  // response.content     - 数组，每个元素是一个内容块
  //   [{ type: 'text', text: '...' }]

  console.log('Claude 说：', response.content[0].text);
  console.log('\n📊 Token 用量：');
  console.log('  输入：', response.usage.input_tokens, 'tokens');
  console.log('  输出：', response.usage.output_tokens, 'tokens');
  console.log('  停止原因：', response.stop_reason);
}

// ═══════════════════════════════════════════════════════════════
// 第二节：System Prompt（给 AI 设定角色 / 行为规则）
// ═══════════════════════════════════════════════════════════════

/**
 * System prompt 就像"员工手册"——告诉 Claude 它是谁、怎么工作。
 * 它在整个对话过程中始终有效，用户看不到也改不掉。
 *
 * @returns {Promise<void>}
 */
async function withSystemPrompt() {
  console.log('\n=== 2. 使用 System Prompt ===\n');

  const response = await client.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 512,
    // system 字段：角色设定 / 行为规范 / 输出格式要求
    system: `你是一位专门为编程初学者服务的老师。
规则：
1. 回答必须使用中文
2. 用最简单的比喻解释概念
3. 每个解释控制在 3 句话以内`,
    messages: [
      { role: 'user', content: '什么是函数？' },
    ],
  });

  console.log('Claude（扮演老师）：', response.content[0].text);
}

// ═══════════════════════════════════════════════════════════════
// 第三节：多轮对话（对话历史 = messages 数组）
// ═══════════════════════════════════════════════════════════════

/**
 * Claude 本身是"无记忆"的——每次请求都是独立的。
 * 要让它"记住"上下文，你需要把历史消息一起发过去。
 * 就像你跟朋友聊天，发新消息时把整段聊天记录都截图发过去一样。
 *
 * @returns {Promise<void>}
 */
async function multiTurnChat() {
  console.log('\n=== 3. 多轮对话 ===\n');

  // 对话历史：你来我往的消息数组
  // 规则：user 和 assistant 必须交替出现，且必须以 user 开头
  const conversationHistory = [];

  /**
   * 发送一条消息并把 AI 回复追加到历史中
   * @param {string} userMessage - 用户输入
   * @returns {Promise<string>} - AI 的回复文本
   */
  async function chat(userMessage) {
    // 把用户消息加入历史
    conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 512,
      messages: conversationHistory, // 把完整历史都发过去
    });

    const assistantReply = response.content[0].text;

    // 把 AI 回复也加入历史，下一轮请求时一并发送
    conversationHistory.push({
      role: 'assistant',
      content: assistantReply,
    });

    return assistantReply;
  }

  // 模拟三轮对话
  const reply1 = await chat('我叫小明，我在学 JavaScript。');
  console.log('第1轮 User: 我叫小明，我在学 JavaScript。');
  console.log('第1轮 Claude:', reply1);

  const reply2 = await chat('你还记得我叫什么吗？我在学什么？');
  console.log('\n第2轮 User: 你还记得我叫什么吗？我在学什么？');
  console.log('第2轮 Claude:', reply2);

  const reply3 = await chat('给我推荐一个适合我当前水平的学习资源。');
  console.log('\n第3轮 User: 给我推荐一个适合我当前水平的学习资源。');
  console.log('第3轮 Claude:', reply3);

  console.log('\n当前对话历史长度：', conversationHistory.length, '条消息');
}

// ═══════════════════════════════════════════════════════════════
// 第四节：流式输出（Stream）
// ═══════════════════════════════════════════════════════════════

/**
 * 普通模式：等 Claude 写完整段话再一次性返回（等待时屏幕空白）
 * 流式模式：Claude 写一个字，你就立刻看到一个字（像打字机一样）
 *
 * 适合：聊天界面、长文本生成，能大大改善用户体验。
 *
 * @returns {Promise<void>}
 */
async function streamingChat() {
  console.log('\n=== 4. 流式输出（Stream）===\n');
  console.log('Claude 正在流式回答：\n');

  // 使用 stream() 方法代替 create()
  const stream = client.messages.stream({
    model: 'claude-opus-4-5',
    max_tokens: 256,
    messages: [
      { role: 'user', content: '用三点描述 Node.js 的优势，简短一些。' },
    ],
  });

  // for await...of 逐个处理事件
  // 这就像读水管里的水：水来一点处理一点，不用等水桶装满
  for await (const event of stream) {
    // 只关心文本增量事件
    if (
      event.type === 'content_block_delta' &&
      event.delta.type === 'text_delta'
    ) {
      // process.stdout.write 不换行，实现打字机效果
      process.stdout.write(event.delta.text);
    }
  }

  // 流结束后获取完整 response（包含 usage 等信息）
  const finalMessage = await stream.getFinalMessage();
  console.log('\n\n流式结束，总 token：',
    finalMessage.usage.input_tokens + finalMessage.usage.output_tokens
  );
}

// ═══════════════════════════════════════════════════════════════
// 第五节：完整 response 结构解析（放大镜看 response）
// ═══════════════════════════════════════════════════════════════

/**
 * 打印 response 对象的全部字段，帮助理解数据结构
 *
 * @returns {Promise<void>}
 */
async function inspectResponse() {
  console.log('\n=== 5. Response 结构解析 ===\n');

  const response = await client.messages.create({
    model: 'claude-opus-4-5',
    max_tokens: 64,
    messages: [{ role: 'user', content: 'Hi' }],
  });

  console.log('完整 response 字段：');
  console.log({
    id: response.id,                  // 请求 ID，格式如 msg_01Xk...
    type: response.type,              // 永远是 'message'
    role: response.role,              // 永远是 'assistant'
    model: response.model,            // 实际调用的模型
    stop_reason: response.stop_reason,// 停止原因
    stop_sequence: response.stop_sequence, // 触发停止的特殊序列（通常为 null）
    usage: response.usage,            // { input_tokens, output_tokens }
    content_length: response.content.length, // content 数组长度
    first_content_type: response.content[0]?.type, // 通常是 'text'
  });
}

// ═══════════════════════════════════════════════════════════════
// 主函数：依次运行所有示例
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('🚀 Claude API 基础示例\n');
  console.log('类比：这就像 fetch(\'/api/chat\', { method: \'POST\', body: ... })');
  console.log('SDK 帮你处理了鉴权、重试、JSON 解析等脏活\n');
  console.log('─'.repeat(50));

  try {
    await basicChat();
    await withSystemPrompt();
    await multiTurnChat();
    await streamingChat();
    await inspectResponse();

    console.log('\n✅ 所有示例运行完毕！');
    console.log('\n💡 下一步：查看 02_tool_use_basic.js 学习工具调用');
  } catch (error) {
    // 常见错误处理
    if (error.status === 401) {
      console.error('❌ 认证失败：请检查 ANTHROPIC_API_KEY 环境变量');
    } else if (error.status === 429) {
      console.error('❌ 请求过于频繁：稍等几秒后重试');
    } else if (error.status === 529) {
      console.error('❌ API 服务过载：稍后重试');
    } else {
      console.error('❌ 发生错误：', error.message);
    }
    process.exit(1);
  }
}

main();
