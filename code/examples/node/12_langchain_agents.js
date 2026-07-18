/**
 * 12_langchain_agents.js
 * =======================
 * LangChain Agents —— 让 AI 自主决定用什么工具完成任务
 *
 * 什么是 Agent？
 *   普通 AI：你问一个问题，它回答一个问题（被动响应）
 *   Agent：你给它一个目标，它自己决定"先搜资料、再计算、再总结"（主动规划）
 *
 *   类比：普通 AI 是"执行员"（给什么做什么），
 *         Agent 是"项目经理"（给目标，自己排工序）
 *
 * 涵盖内容：
 *   1. DynamicTool —— 简单工具定义（单字符串输入）
 *   2. DynamicStructuredTool —— 带 Zod Schema 的结构化工具（多参数）
 *   3. createReactAgent —— ReAct 模式 Agent 构建
 *   4. AgentExecutor —— 运行 Agent 并控制循环次数
 *   5. BufferMemory —— 对话记忆（跨轮次记住上下文）
 *   6. 流式 Agent 输出（streamEvents）
 *
 * 运行前：
 *   npm install @langchain/anthropic @langchain/core langchain zod
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 12_langchain_agents.js
 */

import { ChatAnthropic } from '@langchain/anthropic';
import { DynamicTool, DynamicStructuredTool } from '@langchain/core/tools';
import { createReactAgent } from 'langchain/agents';
import { AgentExecutor } from 'langchain/agents';
import { BufferMemory } from 'langchain/memory';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { z } from 'zod';

// ─────────────────────────────────────────────
// 初始化模型
// ─────────────────────────────────────────────

const model = new ChatAnthropic({
  model: 'claude-opus-4-5',
  maxTokens: 2048,
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ─────────────────────────────────────────────
// 1. DynamicTool —— 最简单的工具定义方式
// ─────────────────────────────────────────────

/**
 * DynamicTool 的三要素：
 *   - name：工具名（AI 通过名字选择工具）
 *   - description：工具说明（AI 根据描述判断何时用）
 *   - func：实际执行的函数（接受字符串，返回字符串）
 *
 * 类比：给新员工配备工具箱，每个工具都贴着标签说明用途
 *
 * DynamicTool 的限制：输入只能是一个字符串（简单场景够用）
 * 需要多个参数时，用下面的 DynamicStructuredTool
 */

// 工具1：获取当前时间
const getCurrentTimeTool = new DynamicTool({
  name: 'get_current_time',
  description: '获取当前的日期和时间。当用户询问现在几点、今天是几号时使用。',
  func: async () => {
    return `当前时间：${new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })}`;
  },
});

// 工具2：天气查询（模拟）
const getWeatherTool = new DynamicTool({
  name: 'get_weather',
  description: '查询某个城市的天气。输入城市名称，返回天气信息。',
  func: async (city) => {
    const weatherData = {
      北京: '晴，23°C，微风',
      上海: '多云，26°C，东南风3级',
      广州: '阵雨，28°C，南风4级',
      深圳: '阴，27°C，偏南风',
    };
    return `${city}天气：${weatherData[city] || `${city}：晴，22°C，微风（模拟数据）`}`;
  },
});

// ─────────────────────────────────────────────
// 2. DynamicStructuredTool —— 带 Zod Schema 的结构化工具
// ─────────────────────────────────────────────

/**
 * DynamicStructuredTool 的优势：
 *   支持多个类型化参数，AI 会自动解析参数格式。
 *   用 Zod 定义 Schema，内置类型验证。
 *
 * 与 DynamicTool 的对比：
 *   DynamicTool：   func(inputString) → 只有一个字符串参数
 *   DynamicStructuredTool：func({ param1, param2, ... }) → 结构化多参数
 *
 * 类比：
 *   DynamicTool 是"写个便签"（一句话说清楚）
 *   DynamicStructuredTool 是"填写结构化表单"（每个字段有类型和描述）
 */
const calculatorTool = new DynamicStructuredTool({
  name: 'calculator',
  description: '进行数学计算。支持加减乘除和括号，如 "2 + 3 * 4"。',
  schema: z.object({
    expression: z.string().describe('数学表达式，只含数字和 +、-、*、/、()'),
  }),
  func: async ({ expression }) => {
    try {
      const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, '');
      // 注意：eval 有安全风险，生产环境用 math.js 或 mathjs
      const result = Function(`"use strict"; return (${sanitized})`)();
      return `${expression} = ${result}`;
    } catch {
      return `计算失败：表达式无效（${expression}）`;
    }
  },
});

const createReminderTool = new DynamicStructuredTool({
  name: 'create_reminder',
  description: '创建一个提醒事项，需要提供内容和时间。',
  schema: z.object({
    content: z.string().describe('提醒的具体内容'),
    time: z.string().describe('提醒时间，如 "明天上午9点" 或 "2025-01-01 09:00"'),
    priority: z.enum(['low', 'medium', 'high']).optional().describe('优先级（可选）'),
  }),
  func: async ({ content, time, priority = 'medium' }) => {
    const reminder = {
      id: Math.random().toString(36).slice(2, 8),
      content,
      time,
      priority,
      createdAt: new Date().toISOString(),
    };
    return `提醒已创建：${JSON.stringify(reminder)}`;
  },
});

const tools = [getCurrentTimeTool, getWeatherTool, calculatorTool, createReminderTool];

// ─────────────────────────────────────────────
// 3. createReactAgent + AgentExecutor
// ─────────────────────────────────────────────

/**
 * ReAct（Reasoning + Acting）模式
 *
 * Agent 的内部思考循环：
 *   1. Thought（推理）：我要完成什么，需要用哪个工具？
 *   2. Action（行动）：调用工具
 *   3. Observation（观察）：工具返回了什么？
 *   4. 重复，直到有足够信息给出最终答案
 *
 * 类比：侦探破案
 *   推理 → 调查 → 观察证据 → 再推理 → ... → 最终结论
 *
 * AgentExecutor 的作用：
 *   管理 ReAct 循环，限制最大迭代次数，避免死循环
 */

const agentSystemPrompt = `你是一个能干的 AI 助手，可以使用工具来完成任务。
请用中文回答，并在必要时使用工具获取准确信息。
当你有足够信息时，直接给出最终答案，不要过度调用工具。`;

async function demo1_basicAgent() {
  console.log('\n=== 1. 基础 ReAct Agent ===');

  const agent = await createReactAgent({
    llm: model,
    tools,
    messageModifier: agentSystemPrompt,
  });

  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: true,     // 打印每步推理过程（调试用）
    maxIterations: 5,  // 最多循环 5 次
  });

  // 这个任务需要用到多个工具（时间 + 计算）
  const result = await executor.invoke({
    input: '现在几点了？顺便帮我算一下 365 * 24 * 60，这是一年有多少分钟。',
  });

  console.log('\n最终答案:', result.output);
}

// ─────────────────────────────────────────────
// 4. BufferMemory —— 对话记忆
// ─────────────────────────────────────────────

/**
 * BufferMemory
 *
 * 什么是"记忆"？
 *   默认 Agent 每次调用是独立的，不记得上一次说了什么。
 *   BufferMemory 把历史消息保存在内存里，每次调用时一起传给 AI。
 *
 *   类比：普通 AI 是"金鱼记忆"（每次聊天从头开始），
 *         加了 BufferMemory 的 AI 是"有笔记本的学生"（记得之前说过什么）
 *
 * 关键配置：
 *   - memoryKey     : Prompt 中对应的变量名（{chat_history}）
 *   - returnMessages: true → 返回 Message 对象（配合 MessagesPlaceholder 使用）
 *   - inputKey/outputKey: 告诉 Memory 哪些字段是输入/输出
 */
async function demo2_agentWithMemory() {
  console.log('\n=== 2. 带 BufferMemory 对话记忆的 Agent ===');

  const memory = new BufferMemory({
    memoryKey: 'chat_history',  // 对应 Prompt 中的 {chat_history}
    returnMessages: true,
    inputKey: 'input',
    outputKey: 'output',
  });

  // 带记忆的 Prompt 必须包含 MessagesPlaceholder 注入历史消息
  const promptWithMemory = ChatPromptTemplate.fromMessages([
    ['system', agentSystemPrompt],
    new MessagesPlaceholder('chat_history'),   // ← 历史消息注入这里
    ['human', '{input}'],
    new MessagesPlaceholder('agent_scratchpad'), // ← Agent 推理过程
  ]);

  const agent = await createReactAgent({
    llm: model,
    tools,
    prompt: promptWithMemory,
  });

  const executor = new AgentExecutor({
    agent,
    tools,
    memory,
    verbose: false,
    maxIterations: 5,
  });

  // 第一轮：告诉 Agent 用户的名字
  console.log('用户（第1轮）: 我叫小明，帮我查一下北京的天气');
  const result1 = await executor.invoke({ input: '我叫小明，帮我查一下北京的天气' });
  console.log('AI（第1轮）:', result1.output);

  // 第二轮：验证记忆是否生效
  console.log('\n用户（第2轮）: 我刚才说我叫什么名字？');
  const result2 = await executor.invoke({ input: '我刚才说我叫什么名字？' });
  console.log('AI（第2轮）:', result2.output);
  // 预期：AI 会说"你叫小明"，因为 BufferMemory 保存了第一轮
}

// ─────────────────────────────────────────────
// 5. 流式 Agent 输出
// ─────────────────────────────────────────────

/**
 * streamEvents 流式输出
 *
 * 可以监听 Agent 每一步的事件：
 *   on_tool_start   → 工具开始执行
 *   on_tool_end     → 工具执行完毕
 *   on_llm_stream   → AI 正在生成文字
 *   on_chain_end    → 最终结果
 *
 * 类比：直播比赛的实况解说，而不是等比赛结束再看录像
 */
async function demo3_streamingAgent() {
  console.log('\n=== 3. 流式 Agent 输出（streamEvents）===');

  const agent = await createReactAgent({
    llm: model,
    tools,
    messageModifier: agentSystemPrompt,
  });

  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: false,
    maxIterations: 5,
  });

  console.log('用户: 帮我计算 123 * 456，同时查一下上海的天气\n');

  const eventStream = executor.streamEvents(
    { input: '帮我计算 123 * 456，同时查一下上海的天气' },
    { version: 'v2' }
  );

  for await (const event of eventStream) {
    if (event.event === 'on_tool_start') {
      console.log(`[工具调用] ${event.name}`, '输入:', JSON.stringify(event.data.input));
    } else if (event.event === 'on_tool_end') {
      console.log(`[工具返回] ${event.data.output}`);
    } else if (event.event === 'on_chain_end' && event.name === 'AgentExecutor') {
      console.log('\n[最终答案]', event.data.output.output);
    }
  }
}

// ─────────────────────────────────────────────
// 主函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔═══════════════════════════════════════════╗');
  console.log('║  12_langchain_agents.js  LangChain Agents  ║');
  console.log('╚═══════════════════════════════════════════╝');

  try {
    await demo1_basicAgent();
    await demo2_agentWithMemory();
    await demo3_streamingAgent();
  } catch (error) {
    console.error('演示出错:', error.message);
    if (error.message.includes('Cannot find package') || error.message.includes('Module not found')) {
      console.error('\n请先安装依赖:');
      console.error('  npm install @langchain/anthropic @langchain/core langchain zod');
    }
  }

  console.log('\n所有演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
