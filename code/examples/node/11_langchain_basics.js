/**
 * 11_langchain_basics.js
 * =======================
 * LangChain 基础 —— 用"链式组合"思想构建 AI 应用
 *
 * LangChain 是什么？
 *   一个帮你把 AI 能力"像搭乐高一样"组合起来的框架。
 *   你不需要手写每一步，把"提示词 → AI → 解析结果"组合成链，
 *   就像流水线工厂：原材料进去，成品出来。
 *
 * 与原生 Anthropic SDK 的对比：
 *   原生 SDK：手动写 messages 数组、手动提取 response.content[0].text
 *   LangChain：声明式链式调用，.pipe() 连接各步骤，代码更简洁
 *   切换模型：原生需要改很多代码；LangChain 只改一行 import
 *
 * 涵盖内容：
 *   1. ChatAnthropic 基础调用
 *   2. ChatPromptTemplate 模板
 *   3. LCEL 链（.pipe()）
 *   4. StrOutputParser、JsonOutputParser
 *   5. RunnableParallel（并行执行）
 *   6. 流式输出（chain.stream()）
 *
 * 运行前：
 *   npm install @langchain/anthropic @langchain/core langchain
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 11_langchain_basics.js
 */

import { ChatAnthropic } from '@langchain/anthropic';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { StringOutputParser, JsonOutputParser } from '@langchain/core/output_parsers';
import { RunnableParallel, RunnablePassthrough } from '@langchain/core/runnables';

// ─────────────────────────────────────────────
// 初始化模型
// ─────────────────────────────────────────────

/**
 * ChatAnthropic —— LangChain 封装的 Anthropic 模型
 *
 * 与原生 SDK 对比：
 *   原生：const client = new Anthropic();
 *         await client.messages.create({ model, max_tokens, messages })
 *   LangChain：const model = new ChatAnthropic({...});
 *               await model.invoke([...messages])
 *
 * LangChain 的核心优势：所有 LLM 用统一接口（invoke/stream/batch），
 * 切换 OpenAI/Google/本地模型只需换一行 import，其余代码不变。
 */
const model = new ChatAnthropic({
  model: 'claude-opus-4-5',
  maxTokens: 1024,
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ─────────────────────────────────────────────
// 1. 基础调用
// ─────────────────────────────────────────────

async function demo1_basicCall() {
  console.log('\n=== 1. 基础调用 ===');

  // 原生 SDK 写法（对比）：
  // const client = new Anthropic();
  // const resp = await client.messages.create({
  //   model: 'claude-opus-4-5', max_tokens: 100,
  //   messages: [{ role: 'user', content: '1+1=?' }]
  // });
  // const text = resp.content[0].text;  ← 需要手动提取

  // LangChain 写法：直接 invoke，返回 AIMessage 对象
  const response = await model.invoke('1+1=？用一句话回答。');
  console.log('回复 (AIMessage.content):', response.content);
  console.log('Token 使用:', response.usage_metadata);
}

// ─────────────────────────────────────────────
// 2. ChatPromptTemplate —— 可复用的提示词模板
// ─────────────────────────────────────────────

/**
 * ChatPromptTemplate
 *
 * 类比：Word 文档模板 —— 你定义好格式，使用时只填写变量部分。
 * {topic}、{role} 是占位符，调用时传入具体值自动填充。
 *
 * 与原生 SDK 对比：
 *   原生：手动用模板字符串拼 messages 数组，容易出错
 *   LangChain：声明式模板，变量自动注入，支持多轮对话模板
 */
const explainTemplate = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    '你是一位{role}，擅长用简单的比喻解释复杂概念。每次回答控制在{maxWords}字以内。'
  ),
  HumanMessagePromptTemplate.fromTemplate(
    '请用一个生活中的比喻解释：{topic}'
  ),
]);

async function demo2_promptTemplate() {
  console.log('\n=== 2. ChatPromptTemplate ===');

  // 格式化模板，生成标准的 Messages 列表
  const messages = await explainTemplate.formatMessages({
    role: '技术老师',
    maxWords: '100',
    topic: '什么是递归？',
  });

  console.log('格式化后的消息:');
  messages.forEach(m => console.log(`  [${m._getType()}]: ${m.content}`));

  // 发送请求
  const response = await model.invoke(messages);
  console.log('\nAI 回复:', response.content);
}

// ─────────────────────────────────────────────
// 3. LCEL 链 —— 用 .pipe() 组合步骤
// ─────────────────────────────────────────────

/**
 * LCEL（LangChain Expression Language）
 *
 * 什么是 .pipe()？
 *   就像 Unix 管道符 |，把上一步的输出作为下一步的输入：
 *   prompt → model → parser
 *
 * 类比：流水线工厂
 *   原材料（变量）→ 组装站（prompt）→ AI 工人（model）→ 质检（parser）→ 成品
 *
 * 与原生 SDK 对比：
 *   原生：步骤1（格式化prompt）+ 步骤2（调用API）+ 步骤3（提取文本）= 3 行代码
 *   LangChain：chain.invoke({...}) 一行搞定，且可复用
 */

// StrOutputParser：把 AI 的 AIMessage 对象 → 提取出纯文字字符串
// 类比：快递员（model）给你一个包装盒（AIMessage），
//        StrOutputParser 帮你拆开取出里面的东西（纯文本）
const strParser = new StringOutputParser();

// 创建链：模板 → 模型 → 字符串解析器
const explainChain = explainTemplate.pipe(model).pipe(strParser);

async function demo3_lcelChain() {
  console.log('\n=== 3. LCEL 链（.pipe()）===');

  const result = await explainChain.invoke({
    role: '幼儿园老师',
    maxWords: '50',
    topic: '什么是数据库？',
  });

  console.log('链式调用结果 (直接是字符串):', result);
  console.log('类型:', typeof result); // string（已经被 StringOutputParser 解析）
}

// ─────────────────────────────────────────────
// 4. JsonOutputParser —— 让 AI 输出结构化 JSON
// ─────────────────────────────────────────────

/**
 * JsonOutputParser
 *
 * 类比：让 AI 填表格，而不是写作文。
 * 你告诉 AI "用 JSON 格式回答"，JsonOutputParser 自动解析结果。
 *
 * 与原生 SDK 对比：
 *   原生：response.content[0].text → JSON.parse(text)（可能抛出 SyntaxError）
 *   LangChain：JsonOutputParser 内置了解析 + 基本错误处理
 */
const jsonTemplate = ChatPromptTemplate.fromMessages([
  ['system', '你是一个数据提取助手，总是用 JSON 格式回答。'],
  ['human', '从下面的文本中提取姓名、年龄和职业：{text}\n请输出 JSON，格式：{{"name":"...","age":0,"job":"..."}}'],
]);

const jsonParser = new JsonOutputParser();
const jsonChain = jsonTemplate.pipe(model).pipe(jsonParser);

async function demo4_jsonParser() {
  console.log('\n=== 4. JsonOutputParser ===');

  const result = await jsonChain.invoke({
    text: '张伟，35岁，是一名软件工程师，在北京工作了十年。',
  });

  console.log('解析结果（已是 JS 对象）:', result);
  console.log('类型:', typeof result); // object（已自动 JSON.parse）
  console.log('访问字段 result.name:', result.name);
}

// ─────────────────────────────────────────────
// 5. RunnableParallel —— 并行执行多个链
// ─────────────────────────────────────────────

/**
 * RunnableParallel
 *
 * 类比：同时开多个窗口工作，而不是一个接一个处理。
 * 多个任务同时执行，总耗时 ≈ 最慢的那一个任务的时间。
 *
 * 与原生 SDK 对比：
 *   原生：手动 Promise.all([call1(), call2()])，但每个调用还要手写
 *   LangChain：RunnableParallel 声明式并行，代码更清晰、可复用
 */
async function demo5_parallel() {
  console.log('\n=== 5. RunnableParallel 并行执行 ===');

  // 两个不同风格的解释 Prompt
  const casualTemplate = ChatPromptTemplate.fromMessages([
    ['human', '用大白话解释一下：{concept}（30字以内）'],
  ]);
  const technicalTemplate = ChatPromptTemplate.fromMessages([
    ['human', '用技术术语精确定义：{concept}（30字以内）'],
  ]);

  // RunnableParallel：同时执行 casual 和 technical 两条链
  const parallelChain = new RunnableParallel({
    casual:    casualTemplate.pipe(model).pipe(strParser),
    technical: technicalTemplate.pipe(model).pipe(strParser),
    // RunnablePassthrough 直接把输入原样传递，不做处理
    // 类比：传送带上的"透明货架"，东西经过时不做任何加工
    original:  new RunnablePassthrough(),
  });

  const start = Date.now();
  const results = await parallelChain.invoke({ concept: 'API' });
  const elapsed = Date.now() - start;

  console.log('原始输入:', results.original);
  console.log('通俗解释:', results.casual);
  console.log('技术定义:', results.technical);
  console.log(`\n两个 AI 调用并行完成，总耗时: ${elapsed}ms（串行约需 2 倍时间）`);
}

// ─────────────────────────────────────────────
// 6. 流式输出 —— 字一个一个地"打印"出来
// ─────────────────────────────────────────────

/**
 * 流式输出（chain.stream()）
 *
 * 什么是流式输出？
 *   AI 生成回复时，不是等全部生成完才返回，而是边生成边发送。
 *   用户能立刻看到第一个字，体验更好（尤其是长回复）。
 *   类比：打字机效果（每个字逐一出现）vs 等全文写完再给你看。
 *
 * 与原生 SDK 对比：
 *   原生：for await (const ev of client.messages.stream({...})) { ... }
 *   LangChain：for await (const chunk of chain.stream({...})) { ... }
 *   接口几乎一样，但 LangChain 链中每一个步骤都支持流式透传
 */
async function demo6_streaming() {
  console.log('\n=== 6. 流式输出（chain.stream()）===');

  const streamChain = ChatPromptTemplate.fromMessages([
    ['human', '用100字介绍一下 {topic}，包括它的用途和特点。'],
  ]).pipe(model).pipe(strParser);

  process.stdout.write('流式输出（逐字打印）: ');

  // stream() 返回 AsyncIterable，每次 yield 一个文字片段
  for await (const chunk of await streamChain.stream({ topic: 'LangChain' })) {
    process.stdout.write(chunk); // 不换行，直接写到终端
  }

  console.log('\n\n流式输出完成。');
}

// ─────────────────────────────────────────────
// 主函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔═════════════════════════════════════════╗');
  console.log('║  11_langchain_basics.js  LangChain 基础 ║');
  console.log('╚═════════════════════════════════════════╝');

  try {
    await demo1_basicCall();
    await demo2_promptTemplate();
    await demo3_lcelChain();
    await demo4_jsonParser();
    await demo5_parallel();
    await demo6_streaming();
  } catch (error) {
    console.error('演示出错:', error.message);
    if (error.message.includes('Cannot find package') || error.message.includes('Module not found')) {
      console.error('\n请先安装依赖:');
      console.error('  npm install @langchain/anthropic @langchain/core langchain');
    }
  }

  console.log('\n所有演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
