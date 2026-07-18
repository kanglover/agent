/**
 * 14_langgraph_react_agent.js
 * ============================
 * LangGraph ReAct Agent —— 生产级带记忆和人工干预的 AI Agent
 *
 * 与 LangChain 的 createReactAgent 有何不同？
 *   LangChain Agent：黑盒，内部实现不透明，难以定制
 *   LangGraph Agent：图结构，每个节点可见可控，支持：
 *     - 检查点（Checkpoint）：随时暂停 / 恢复执行
 *     - Human-in-the-Loop：等待人工确认再继续
 *     - 跨对话持久化记忆（通过 thread_id 区分会话）
 *
 * 涵盖内容：
 *   1. createReactAgent（@langchain/langgraph/prebuilt）
 *   2. 自定义工具集成（tool() 函数方式）
 *   3. MemorySaver 检查点（跨对话记忆）
 *   4. streamEvents 流式输出
 *   5. Human-in-the-Loop（interrupt 中断等待人工确认）
 *
 * 运行前：
 *   npm install @langchain/langgraph @langchain/anthropic @langchain/core zod
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 14_langgraph_react_agent.js
 */

import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { MemorySaver } from '@langchain/langgraph';
import { ChatAnthropic } from '@langchain/anthropic';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
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
// 1. 自定义工具（使用 @langchain/core/tools 的 tool() 函数）
// ─────────────────────────────────────────────

/**
 * tool() 是 LangGraph 推荐的工具定义方式
 * 比 DynamicStructuredTool 更简洁 —— 直接传函数 + schema
 *
 * 结构：tool(执行函数, { name, description, schema })
 */

// 工具1：搜索网络（模拟）
const searchTool = tool(
  async ({ query }) => {
    console.log(`  [工具] search: "${query}"`);
    const results = {
      '量子计算': '量子计算利用量子力学原理（叠加、纠缠），在特定问题上比传统计算机快指数级。',
      'Node.js':  'Node.js 是基于 V8 引擎的 JS 运行时，非阻塞 I/O 让它特别适合网络 I/O 密集应用。',
      'LangGraph': 'LangGraph 是 LangChain 的图形工作流框架，支持循环、条件分支和 Human-in-the-Loop。',
    };
    return results[query] || `"${query}" 的搜索结果：关于 ${query} 的最新资讯（模拟数据）。`;
  },
  {
    name: 'search',
    description: '搜索关于某个话题的信息，当需要最新资讯或背景知识时使用',
    schema: z.object({
      query: z.string().describe('搜索关键词'),
    }),
  }
);

// 工具2：执行代码（模拟）
const executeCodeTool = tool(
  async ({ code, language }) => {
    console.log(`  [工具] execute_code (${language}): ${code.slice(0, 50)}...`);
    // 实际场景：在沙箱（如 Docker 容器）中安全执行代码
    // 演示：返回模拟结果
    if (language === 'javascript' && code.includes('console.log')) {
      return `执行成功（模拟）:\n> ${code}\n→ Hello, World!`;
    }
    return `代码执行成功（模拟）:\n语言: ${language}\n→ [执行结果]`;
  },
  {
    name: 'execute_code',
    description: '在安全沙箱中执行代码片段，返回执行结果',
    schema: z.object({
      code:     z.string().describe('要执行的代码'),
      language: z.enum(['python', 'javascript', 'bash']).describe('编程语言'),
    }),
  }
);

// 工具3：保存笔记
const saveNoteTool = tool(
  async ({ title, content }) => {
    console.log(`  [工具] save_note: "${title}"`);
    return `笔记已保存 —— ID: ${Date.now()}, 标题: ${title}, 时间: ${new Date().toISOString()}`;
  },
  {
    name: 'save_note',
    description: '保存一条笔记供以后参考',
    schema: z.object({
      title:   z.string().describe('笔记标题'),
      content: z.string().describe('笔记内容'),
    }),
  }
);

const tools = [searchTool, executeCodeTool, saveNoteTool];

// ─────────────────────────────────────────────
// 2. MemorySaver —— 跨对话记忆检查点
// ─────────────────────────────────────────────

/**
 * MemorySaver
 *
 * 什么是"检查点"（Checkpoint）？
 *   Agent 每次执行后，把当前状态（消息历史、工具调用记录）存起来。
 *   下次对话时，通过 thread_id 恢复上次状态，就像"接着上次说"。
 *
 *   类比：视频游戏存档 —— 随时存档，随时读档继续，不用从头开始。
 *
 * MemorySaver：内存存储（进程重启清空）
 * 生产环境推荐：SqliteSaver 或 PostgresSaver（数据库持久化）
 */
const checkpointer = new MemorySaver();

const agent = createReactAgent({
  llm: model,
  tools,
  checkpointSaver: checkpointer,
  messageModifier: `你是一个能干的 AI 助手，可以搜索信息、执行代码和保存笔记。
请用中文回答，保持简洁（每次回答控制在 150 字以内）。`,
});

// ─────────────────────────────────────────────
// 3. 演示：跨对话记忆
// ─────────────────────────────────────────────

async function demo1_agentWithMemory() {
  console.log('\n=== 1. createReactAgent + MemorySaver 跨对话记忆 ===');

  // thread_id 是会话标识符，相同 thread_id 的请求共享历史记忆
  const config = { configurable: { thread_id: 'user_001_session_a' } };

  // 第一轮：告诉 Agent 用户名字并搜索信息
  console.log('用户（第1轮）: 我叫小明，帮我搜索一下量子计算');
  const result1 = await agent.invoke(
    { messages: [new HumanMessage('我叫小明，帮我搜索一下量子计算')] },
    config
  );
  console.log('AI（第1轮）:', result1.messages.at(-1).content);

  // 第二轮：验证记忆（同一 thread_id，AI 应记得小明）
  console.log('\n用户（第2轮）: 把刚才的内容保存成笔记');
  const result2 = await agent.invoke(
    { messages: [new HumanMessage('把刚才的内容保存成笔记')] },
    config // 相同 thread_id → 自动恢复上一轮状态
  );
  console.log('AI（第2轮）:', result2.messages.at(-1).content);

  // 换新 thread_id = 新的独立会话（不共享记忆）
  const newConfig = { configurable: { thread_id: 'user_001_session_b' } };
  console.log('\n用户（新会话）: 我叫什么名字？');
  const result3 = await agent.invoke(
    { messages: [new HumanMessage('我叫什么名字？')] },
    newConfig
  );
  console.log('AI（新会话，无记忆）:', result3.messages.at(-1).content);
  // 预期：AI 说不知道，因为这是新 thread_id
}

// ─────────────────────────────────────────────
// 4. streamEvents —— 详细的流式事件输出
// ─────────────────────────────────────────────

/**
 * streamEvents 提供粒度最细的流式事件：
 *   on_chat_model_stream → AI 正在生成文字（逐字流式）
 *   on_tool_start         → 工具开始执行
 *   on_tool_end           → 工具执行完毕
 *   on_chain_end（LangGraph）→ 最终结果
 *
 * 与 stream() 的区别：
 *   stream()      → 每个图节点完成后推送状态变化
 *   streamEvents()→ 更细粒度，AI 生字也能逐字捕获
 */
async function demo2_streamEvents() {
  console.log('\n=== 2. streamEvents 流式输出 ===');

  const config = { configurable: { thread_id: 'stream_demo_001' } };

  console.log('用户: 搜索 LangGraph，然后执行一段 JS 代码\n');

  const eventStream = agent.streamEvents(
    {
      messages: [
        new HumanMessage('搜索 LangGraph，然后执行一段打印 Hello World 的 JavaScript 代码'),
      ],
    },
    { ...config, version: 'v2' }
  );

  for await (const event of eventStream) {
    switch (event.event) {
      // AI 生字时逐字输出（流式打字机效果）
      case 'on_chat_model_stream':
        if (event.data.chunk?.content) {
          process.stdout.write(event.data.chunk.content);
        }
        break;

      case 'on_tool_start':
        process.stdout.write('\n');
        console.log(`\n[工具开始] ${event.name}  输入: ${JSON.stringify(event.data.input)}`);
        break;

      case 'on_tool_end':
        console.log(`[工具完成] ${event.name}  输出: ${event.data.output}`);
        process.stdout.write('\n');
        break;

      case 'on_chain_end':
        if (event.name === 'LangGraph') {
          process.stdout.write('\n');
          console.log('\n[流式完成]');
        }
        break;
    }
  }
}

// ─────────────────────────────────────────────
// 5. Human-in-the-Loop（HITL）概念说明
// ─────────────────────────────────────────────

/**
 * Human-in-the-Loop 是什么？
 *
 *   在 Agent 执行过程中，碰到"危险操作"或"需要确认"的步骤时，
 *   暂停 Agent，等待人类审批，确认后再继续（或取消）。
 *
 *   类比：银行大额转账弹出"确认要转 10000 元吗？"
 *         点"确认"才真正执行，点"取消"则中止。
 *
 * 实现方式（核心 API）：
 *
 *   // 1. 在工具或节点内部调用 interrupt(message)
 *   //    这会抛出一个特殊的中断信号，暂停图的执行
 *   import { interrupt } from '@langchain/langgraph';
 *
 *   const dangerousDeleteTool = tool(
 *     async ({ itemId }) => {
 *       // 暂停并请求人工确认
 *       const answer = interrupt(`确认删除 ${itemId} 吗？输入 yes/no`);
 *       if (answer !== 'yes') return '已取消';
 *       // ... 执行删除
 *     },
 *     { name: 'delete_item', schema: z.object({ itemId: z.string() }) }
 *   );
 *
 *   // 2. 第一次 invoke → 触发 interrupt，返回中断状态
 *   const result = await agent.invoke(messages, { configurable: { thread_id: 'xyz' } });
 *   // result 包含 interrupt 信息：{ __interrupt__: [{ value: "确认删除..." }] }
 *
 *   // 3. 人工看到提示，做出决定后，用 Command 恢复
 *   import { Command } from '@langchain/langgraph';
 *   const finalResult = await agent.invoke(
 *     new Command({ resume: 'yes' }),  // 把人工答案传入
 *     { configurable: { thread_id: 'xyz' } }  // 同一 thread_id！
 *   );
 *
 * 注意事项：
 *   - interrupt() 必须配合 checkpointSaver 使用，否则状态无法保存
 *   - thread_id 必须相同，否则无法恢复中断前的状态
 *   - 生产环境通常把"等待确认"的状态存入数据库，用 SqliteSaver / PostgresSaver
 *
 * 适用场景：
 *   - 删除 / 修改重要数据前的二次确认
 *   - 发送邮件 / 消息前的人工审查
 *   - 花费超过阈值时的报批流程
 *   - 代码部署前的人工审核
 */
function demo3_humanInTheLoop() {
  console.log('\n=== 3. Human-in-the-Loop（概念说明） ===');
  console.log(`
流程：
  1. agent.invoke(messages, config)        → Agent 执行，遇到危险操作调用 interrupt()
  2. interrupt() 抛出中断，图暂停          → 返回 { __interrupt__: [{ value: "..." }] }
  3. 调用方展示提示给用户                  → "即将删除，确认？(yes/no)"
  4. agent.invoke(Command({resume: "yes"}), config) → 传入答案，恢复执行
  5. interrupt() 返回 "yes"，工具继续      → 删除操作执行完毕

关键 API：
  interrupt(prompt)            在节点/工具内暂停，prompt 发送给调用方
  new Command({ resume: ans }) 传入人工答案，恢复执行
  MemorySaver / SqliteSaver    保存暂停状态（必须，否则无法恢复）
`);
}

// ─────────────────────────────────────────────
// 主函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║  14_langgraph_react_agent.js  LangGraph Agent  ║');
  console.log('╚════════════════════════════════════════════════╝');

  try {
    await demo1_agentWithMemory();
    await demo2_streamEvents();
    demo3_humanInTheLoop();
  } catch (error) {
    console.error('演示出错:', error.message);
    if (error.message.includes('Cannot find package')) {
      console.error('\n请先安装依赖:');
      console.error('  npm install @langchain/langgraph @langchain/anthropic @langchain/core zod');
    }
  }

  console.log('\n所有演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
