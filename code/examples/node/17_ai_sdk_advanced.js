/**
 * Vercel AI SDK 进阶用法
 * 覆盖：streamObject、Tool中间件、多步Agent、自定义Provider、
 *        embedMany、Next.js Server Actions集成、错误处理
 *
 * 运行前提：
 *   npm install ai @ai-sdk/anthropic @ai-sdk/openai zod
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   export OPENAI_API_KEY=sk-...
 */

'use strict';

const { anthropic } = require('@ai-sdk/anthropic');
const { openai }    = require('@ai-sdk/openai');
const {
  streamObject,
  generateText,
  embedMany,
  tool,
  RetryError,
  NoObjectGeneratedError,
  experimental_createProviderRegistry: createProviderRegistry,
} = require('ai');
const { z } = require('zod');

// ─────────────────────────────────────────────────────────────────────────────
// 1. streamObject —— 流式结构化输出
//
//    普通 generateObject 等模型把整个 JSON 写完才返回；
//    streamObject 边生成边推送，每次 yield 一个"部分填充"的对象。
//    适合场景：大表单生成、AI 报告、产品信息卡片（边收边渲染 UI）。
//
//    类比：外卖平台的"实时进度"——不用等所有菜做好，已完成的先上桌。
// ─────────────────────────────────────────────────────────────────────────────
async function demo_streamObject() {
  console.log('\n=== 1. streamObject 流式结构化输出 ===');

  // 定义期望模型填充的数据结构
  const ArticleSchema = z.object({
    title:    z.string().describe('文章标题'),
    summary:  z.string().describe('100字以内的摘要'),
    keywords: z.array(z.string()).describe('3-5个关键词'),
    sections: z.array(z.object({
      heading: z.string(),
      body:    z.string(),
    })).describe('正文段落列表'),
  });

  const { partialObjectStream, object } = await streamObject({
    model:  anthropic('claude-opus-4-5'),
    schema: ArticleSchema,
    prompt: '写一篇关于"大语言模型如何改变软件开发"的简短技术文章',
  });

  let titlePrinted = false;
  // partialObjectStream 是异步迭代器，字段按生成顺序逐渐出现
  for await (const partial of partialObjectStream) {
    if (partial.title && !titlePrinted) {
      console.log('  流式收到 title:', partial.title);
      titlePrinted = true;
    }
  }

  // await object 拿到最终完整对象（等价于 generateObject 的结果）
  const final = await object;
  console.log('  最终关键词:', final.keywords);
  console.log('  段落数量:', final.sections?.length ?? 0);
}


// ─────────────────────────────────────────────────────────────────────────────
// 2. Tool 中间件 —— 拦截工具调用
//
//    思路：在 tool.execute 外面套一层函数，执行前/后插入日志、鉴权、缓存等。
//    类比：快递站的"签收确认"——包裹到达和取走都会留记录。
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 日志中间件工厂：包装任意工具，为其加上调用日志。
 * @param {string} name      工具名（用于日志标识）
 * @param {object} rawTool   原始 tool() 定义对象
 * @returns 包装后的工具对象（可直接传给 generateText.tools）
 */
function withLogging(name, rawTool) {
  return {
    ...rawTool,
    execute: async (args, options) => {
      console.log(`  [中间件] "${name}" 被调用，参数:`, JSON.stringify(args));
      const start = Date.now();
      const result = await rawTool.execute(args, options);
      console.log(`  [中间件] "${name}" 完成，耗时 ${Date.now() - start}ms`);
      return result;
    },
  };
}

/**
 * 缓存中间件工厂：对相同参数的工具调用返回缓存结果，避免重复请求。
 * @param {object} rawTool   原始 tool() 定义对象
 */
function withCache(rawTool) {
  const cache = new Map();
  return {
    ...rawTool,
    execute: async (args, options) => {
      const key = JSON.stringify(args);
      if (cache.has(key)) {
        console.log('  [缓存] 命中缓存，直接返回');
        return cache.get(key);
      }
      const result = await rawTool.execute(args, options);
      cache.set(key, result);
      return result;
    },
  };
}

async function demo_toolMiddleware() {
  console.log('\n=== 2. Tool 中间件（日志 + 缓存）===');

  // 原始工具：查询城市天气（模拟）
  const weatherBase = tool({
    description: '查询指定城市的当前天气',
    parameters: z.object({
      city: z.string().describe('城市名称，如"北京"'),
    }),
    execute: async ({ city }) => {
      // 实际项目中这里调用真实天气 API
      return { city, temperature: '25°C', condition: '晴' };
    },
  });

  // 先加缓存，再加日志（中间件从外到内执行）
  const wrappedWeather = withLogging('weather', withCache(weatherBase));

  const { text } = await generateText({
    model:    anthropic('claude-opus-4-5'),
    tools:    { weather: wrappedWeather },
    maxSteps: 3,
    prompt:   '北京今天天气怎么样？',
  });

  console.log('  模型最终回复:', text);
}


// ─────────────────────────────────────────────────────────────────────────────
// 3. 多步 Agent —— maxSteps + onStepFinish
//
//    单步调用：模型 -> 工具(1次) -> 结果
//    多步调用：模型 -> 工具 -> 模型 -> 工具 -> ... -> 最终回答
//    maxSteps 控制最多循环几轮；onStepFinish 在每步结束时触发回调。
//    这正是 ReAct Agent Loop 的核心机制。
// ─────────────────────────────────────────────────────────────────────────────
async function demo_multiStepAgent() {
  console.log('\n=== 3. 多步 Agent（maxSteps + onStepFinish）===');

  const stepLog = [];

  const { text, steps } = await generateText({
    model: anthropic('claude-opus-4-5'),
    tools: {
      add: tool({
        description: '对两个数字求和',
        parameters:  z.object({ a: z.number(), b: z.number() }),
        execute:     async ({ a, b }) => ({ result: a + b }),
      }),
      multiply: tool({
        description: '对两个数字求积',
        parameters:  z.object({ a: z.number(), b: z.number() }),
        execute:     async ({ a, b }) => ({ result: a * b }),
      }),
    },

    // 最多走 10 轮工具调用，模型自行决定何时停止
    maxSteps: 10,

    // 每步结束后的回调，常用于日志/进度展示
    onStepFinish: (step) => {
      const calls = step.toolCalls?.map(c => c.toolName).join(', ') || '无工具调用';
      stepLog.push(calls);
      console.log(`  第 ${stepLog.length} 步 [${step.stepType}]: ${calls}`);
    },

    prompt: '先把 3 和 7 相加，再把结果乘以 4，告诉我最终答案。',
  });

  console.log('  共走了', steps.length, '步');
  console.log('  最终答案:', text);
}


// ─────────────────────────────────────────────────────────────────────────────
// 4. 自定义 Provider —— createProviderRegistry
//
//    当项目同时使用多家模型时，注册一个"模型别名表"：
//    业务代码只写 "fast" / "smart"，换模型只改注册表一处。
//    类比：公司内网的 DNS —— 用名字找机器，不用记 IP 地址。
// ─────────────────────────────────────────────────────────────────────────────
function demo_customProvider() {
  console.log('\n=== 4. 自定义 Provider（模型注册表）===');

  // 注册表：给每个模型起别名
  const registry = createProviderRegistry({
    // "fast"  → claude-haiku（速度快、价格低）
    fast:  anthropic('claude-haiku-4-5'),
    // "smart" → claude-opus（能力最强）
    smart: anthropic('claude-opus-4-5'),
    // "embed" → OpenAI embedding 模型
    embed: openai.embedding('text-embedding-3-small'),
  });

  console.log('  注册表已创建，包含别名: fast, smart, embed');
  console.log('  使用方式:');
  console.log('    generateText({ model: registry.languageModel("fast"), ... })');
  console.log('    embedMany({ model: registry.textEmbeddingModel("embed"), ... })');
  console.log('  好处: 在 config 文件里切换模型，业务代码零改动');

  return registry;
}


// ─────────────────────────────────────────────────────────────────────────────
// 5. embedMany —— 批量 Embedding
//
//    Embedding 把文本转成数字向量（一组坐标）。
//    语义相近的文本，它们的向量在空间中距离也近。
//    用途：语义搜索、RAG 知识库、文档聚类、相似问题推荐。
//    embedMany 一次 API 调用处理多条，比循环调用 embed 更高效。
// ─────────────────────────────────────────────────────────────────────────────
async function demo_embedMany() {
  console.log('\n=== 5. embedMany 批量 Embedding ===');

  // Anthropic 暂无自己的 embedding 模型，这里用 OpenAI
  if (!process.env.OPENAI_API_KEY) {
    console.log('  [跳过] 未设置 OPENAI_API_KEY，以下为示例代码:');
    console.log(`
    const { embeddings, usage } = await embedMany({
      model:  openai.embedding('text-embedding-3-small'),
      values: ['大语言模型是什么？', '如何写好 Prompt？', 'RAG 检索增强生成'],
    });
    // embeddings[i] 是第 i 条文本的向量（float[] 数组）
    // usage.tokens 统计消耗的 token 数
    `);
    return;
  }

  const texts = [
    '大语言模型是什么？',
    '如何写好 Prompt？',
    'RAG 检索增强生成简介',
    'Claude 和 GPT-4 有什么区别？',
  ];

  const { embeddings, usage } = await embedMany({
    model:  openai.embedding('text-embedding-3-small'),
    values: texts,
  });

  console.log(`  输入 ${texts.length} 条文本`);
  console.log(`  每条向量维度: ${embeddings[0].length}`);
  console.log(`  消耗 tokens: ${usage.tokens}`);

  // 余弦相似度：衡量两个向量方向的接近程度（-1~1，越大越相似）
  const cosine = (a, b) => {
    const dot   = a.reduce((s, v, i) => s + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
    const normB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
    return dot / (normA * normB);
  };

  const sim01 = cosine(embeddings[0], embeddings[1]);
  const sim02 = cosine(embeddings[0], embeddings[2]);
  console.log(`  "${texts[0]}" vs "${texts[1]}" 相似度: ${sim01.toFixed(4)}`);
  console.log(`  "${texts[0]}" vs "${texts[2]}" 相似度: ${sim02.toFixed(4)}`);
}


// ─────────────────────────────────────────────────────────────────────────────
// 6. Next.js Server Actions 集成（代码模板，不实际运行）
//
//    Server Actions 是 Next.js 13+ 的服务端函数，可以直接从客户端 React
//    组件调用，配合 ai/rsc 的 createStreamableValue 实现实时流式输出。
//    类比：餐厅的"服务员呼叫系统"——客户端按按钮，服务端开始做菜并实时报告进度。
// ─────────────────────────────────────────────────────────────────────────────
function demo_nextjsServerAction() {
  console.log('\n=== 6. Next.js Server Actions 集成（代码模板）===');

  console.log(`
// ── app/actions/chat.ts ─────────────────────────────────────────────────────
'use server';                          // 声明这是 Server Action（在服务端执行）

import { streamText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';
import { createStreamableValue } from 'ai/rsc';  // RSC = React Server Components

export async function chat(userMessage: string) {
  // createStreamableValue 创建一个可从服务端向客户端推送更新的"流式值"
  const stream = createStreamableValue('');

  // IIFE（立即执行的异步函数）：不阻塞 Server Action 返回，后台持续推送
  (async () => {
    const { textStream } = await streamText({
      model:    anthropic('claude-opus-4-5'),
      messages: [{ role: 'user', content: userMessage }],
    });

    for await (const chunk of textStream) {
      stream.update(chunk);   // 每个文字块实时推给客户端
    }
    stream.done();            // 通知客户端流结束
  })();

  return { output: stream.value };   // 把流式值句柄传给客户端
}

// ── app/page.tsx（客户端组件）──────────────────────────────────────────────
'use client';
import { useState } from 'react';
import { readStreamableValue } from 'ai/rsc';
import { chat } from './actions/chat';

export default function Page() {
  const [text, setText] = useState('');

  const handleSend = async (msg: string) => {
    const { output } = await chat(msg);
    // readStreamableValue 是一个异步迭代器，每次 yield 最新的累积文本
    for await (const delta of readStreamableValue(output)) {
      setText(delta ?? '');   // 直接 setState，React 自动重渲染
    }
  };

  return (
    <div>
      <button onClick={() => handleSend('你好，介绍一下自己')}>发送</button>
      <p>{text}</p>
    </div>
  );
}
`);
}


// ─────────────────────────────────────────────────────────────────────────────
// 7. 错误处理 —— RetryError 与 NoObjectGeneratedError
//
//    AI SDK 内置了语义化错误类型，让 catch 块更精准：
//    - NoObjectGeneratedError: 模型输出了内容，但不符合 schema
//    - RetryError:             自动重试 N 次后仍然失败
//    - APICallError:           HTTP 层面的错误（含 statusCode）
// ─────────────────────────────────────────────────────────────────────────────
async function demo_errorHandling() {
  console.log('\n=== 7. 错误处理（RetryError / NoObjectGeneratedError）===');

  // ── 7a. NoObjectGeneratedError ─────────────────────────────────────────────
  // 当 streamObject/generateObject 无法生成符合 schema 的 JSON 时抛出。
  // err.text 保存了模型实际输出的原始文本，方便调试。
  try {
    // 故意使用一个几乎不可能满足的约束来触发此错误
    const { object } = await streamObject({
      model:      anthropic('claude-opus-4-5'),
      schema:     z.object({
        // 要求模型输出一个恰好 100 位的纯数字字符串 —— 几乎不可能
        secretCode: z.string().regex(/^\d{100}$/).describe('必须是100位纯数字'),
      }),
      prompt:     '请生成一个对象',
      maxRetries: 0,  // 禁用内置重试，让错误立刻抛出
    });
    await object;  // 等待流结束，错误在此处抛出
  } catch (err) {
    if (err instanceof NoObjectGeneratedError) {
      console.log('  捕获 NoObjectGeneratedError:');
      console.log('    err.message:', err.message);
      // err.text 是模型的原始输出（未通过 schema 验证）
      const preview = (err.text ?? '').slice(0, 80);
      console.log('    模型原始输出(前80字):', preview || '(空)');
    } else {
      throw err;
    }
  }

  // ── 7b. RetryError ─────────────────────────────────────────────────────────
  // AI SDK 默认在网络抖动时自动重试（默认 maxRetries=2，即最多试3次）。
  // 超过重试次数后抛出 RetryError。
  console.log('\n  RetryError 结构说明:');
  console.log('    err.message   — "Failed after N attempts"');
  console.log('    err.errors    — 每次重试失败的原始错误数组');
  console.log('    err.lastError — 最后一次失败的错误对象');

  console.log('\n  处理示例代码:');
  console.log(`
  try {
    const { text } = await generateText({
      model:      anthropic('claude-opus-4-5'),
      prompt:     '...',
      maxRetries: 3,          // SDK 内置重试，默认 2
    });
  } catch (err) {
    if (err instanceof RetryError) {
      console.error('重试 3 次后仍然失败');
      console.error('最后一次错误:', err.lastError?.message);
      // 典型处置：上报监控 / 降级到备用模型 / 返回缓存结果
    }
  }
  `);

  // ── 7c. 通用最佳实践 ───────────────────────────────────────────────────────
  console.log('  通用建议:');
  console.log('  ① 用 instanceof 区分错误类型，而非只检查 message 字符串');
  console.log('  ② 4xx HTTP 错误（参数问题）不重试；5xx（服务器问题）可重试');
  console.log('  ③ streamObject 失败时用 err.text 查看模型的实际输出来定位问题');
  console.log('  ④ 生产环境设置 AbortSignal.timeout(ms) 防止请求无限挂起');
}


// ─────────────────────────────────────────────────────────────────────────────
// 主函数：依次运行所有演示
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  console.log('Vercel AI SDK 进阶演示');
  console.log('='.repeat(50));

  // 演示 4、6 不需要网络，直接运行
  demo_customProvider();
  demo_nextjsServerAction();

  // 以下演示需要真实的 ANTHROPIC_API_KEY
  if (!process.env.ANTHROPIC_API_KEY) {
    console.log('\n[提示] 未检测到 ANTHROPIC_API_KEY，跳过需要网络的演示。');
    console.log('       请运行: export ANTHROPIC_API_KEY=sk-ant-...');
    console.log('       再重新执行本文件。');
    return;
  }

  try {
    await demo_streamObject();
    await demo_toolMiddleware();
    await demo_multiStepAgent();
    await demo_embedMany();     // 内部已处理无 OPENAI_API_KEY 的情况
    await demo_errorHandling();
  } catch (err) {
    console.error('\n运行出错:', err.message ?? err);
    process.exit(1);
  }

  console.log('\n所有演示完成。');
}

main();
