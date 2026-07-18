/**
 * 19_openai_agents_sdk.js
 * =======================
 * OpenAI Agents SDK —— 轻量级多 Agent 协作框架
 *
 * 【核心理念】
 * 用最少的代码搭建能"自主决策 + 调用工具 + 移交给其他 Agent"的智能体。
 *
 * 【三大核心原语】
 * - Agent   ：有名字、有指令、有工具的"角色"
 * - Tool    ：Agent 可以调用的函数（搜索、计算、查数据库等）
 * - Handoff ：把对话控制权移交给另一个更专业的 Agent
 *
 * 类比：公司的前台电话系统：前台接电话后判断问题类型，
 *       技术问题转技术部，投诉转客诉部，开票转财务部——这就是 Handoff。
 *
 * 安装：npm install @openai/agents zod
 * 使用 Claude 时：设置 OPENAI_API_KEY 和 OPENAI_BASE_URL（通过 OpenRouter）
 */

// ─────────────────────────────────────────────
// 导入（实际运行时取消注释）
// ─────────────────────────────────────────────
// import { Agent, tool, Runner } from '@openai/agents';
// import { z } from 'zod';

// ─────────────────────────────────────────────
// 1. 最简单的 Agent 定义
// ─────────────────────────────────────────────
/*
const simpleAgent = new Agent({
  name: '通用助手',           // Agent 的名字（Handoff 时会显示给对方 Agent）
  instructions: `
    你是一位友善的助手，用中文回答所有问题。
    回答要简洁，不超过 3 句话。
  `,
  // model 默认是 'gpt-4o'
  // 使用 Claude 时：model: 'anthropic/claude-opus-4-5'（需要 OpenRouter）
});

// 运行 Agent（最简单的方式）
const result = await Runner.run(simpleAgent, '什么是 JavaScript 闭包？');
console.log(result.finalOutput);
// finalOutput 是 Agent 给用户的最终回复字符串

// 查看完整执行轨迹（包括每次工具调用和中间思考）
for (const item of result.newItems) {
  console.log('执行步骤：', item.type, JSON.stringify(item).slice(0, 80));
}
*/

// ─────────────────────────────────────────────
// 2. 定义工具 —— 让 Agent 能"做事"
//    工具 = 有描述的函数，Agent 自动决定何时调用
//    写法和 Vercel AI SDK 几乎一样，学一次两边都会
// ─────────────────────────────────────────────
/*
import { z } from 'zod';

// 工具示例1：查询订单状态
const queryOrderTool = tool({
  name: 'query_order',
  description: '根据订单号查询订单的当前状态、金额和物流信息',
  parameters: z.object({
    order_id: z.string().describe('订单号，格式如 ORD-12345'),
  }),
  execute: async ({ order_id }) => {
    // 实际项目中这里调用数据库或物流 API
    const mockData = {
      'ORD-12345': { status: '已发货', amount: 299, logistics: '顺丰 SF123456，明日送达' },
      'ORD-99999': { status: '待付款', amount: 99, logistics: '未发货' },
    };
    const order = mockData[order_id];
    if (!order) return `未找到订单 ${order_id}，请检查订单号是否正确`;
    return JSON.stringify(order);
  },
});

// 工具示例2：提交退款申请
const submitRefundTool = tool({
  name: 'submit_refund',
  description: '为指定订单发起退款申请，成功后返回退款单号',
  parameters: z.object({
    order_id: z.string().describe('需要退款的订单号'),
    reason: z.string().describe('退款原因，需具体说明'),
    amount: z.number().describe('退款金额，单位：元'),
  }),
  execute: async ({ order_id, reason, amount }) => {
    const refundId = `REF-${Date.now()}`;
    console.log(`[退款系统] 新申请：${refundId}，订单：${order_id}，金额：¥${amount}`);
    return `退款申请已提交。退款单号：${refundId}，金额 ¥${amount}，
            预计 3-5 个工作日到账。原因已记录：${reason}`;
  },
});

// 工具示例3：创建技术工单
const createTicketTool = tool({
  name: 'create_ticket',
  description: '为用户反馈的技术问题创建工单',
  parameters: z.object({
    title: z.string().describe('问题标题，简短描述'),
    description: z.string().describe('问题详细描述'),
    severity: z.enum(['低', '中', '高', '紧急']).describe('严重程度'),
  }),
  execute: async ({ title, description, severity }) => {
    const ticketId = `BUG-${Math.floor(Math.random() * 10000)}`;
    return `工单已创建：${ticketId}（${severity}优先级）。
            技术团队将在 24 小时内响应，进展会通过邮件通知您。`;
  },
});
*/

// ─────────────────────────────────────────────
// 3. Runner.run —— 执行 Agent
//    内部机制：自动循环"思考→调用工具→观察结果→继续思考"
//    直到 Agent 认为任务完成
// ─────────────────────────────────────────────
/*
const orderAgent = new Agent({
  name: '订单专员',
  instructions: `
    你是订单专员，负责处理订单查询和物流跟踪。
    总是用 query_order 工具获取真实数据，不要猜测。
    如果用户需要退款，告诉他你来帮他联系退款专员。
  `,
  tools: [queryOrderTool],
});

// 基础执行
const result = await Runner.run(orderAgent, '我的订单 ORD-12345 什么时候到？');
console.log('回复：', result.finalOutput);

// 带上下文的执行
const contextResult = await Runner.run(
  orderAgent,
  '帮我查一下 ORD-12345 和 ORD-99999 两个订单',
  {
    maxTurns: 10,           // 最多循环 10 步（防死循环）
    context: {              // 传递自定义上下文（工具执行时可以访问）
      userId: 'user_001',
      language: 'zh-CN',
    },
  }
);
*/

// ─────────────────────────────────────────────
// 4. Handoff —— Agent 间移交控制权
//    这是 Agents SDK 最独特的功能！
//    当一个 Agent 判断某问题不在自己职责范围内，
//    它会"把用户移交"给更合适的专业 Agent
// ─────────────────────────────────────────────
/*
// 定义专业 Agent（先定义，后面前台 Agent 会引用它们）
const refundAgent = new Agent({
  name: '退款专员',
  instructions: `
    你是退款专员，处理退款申请和售后问题。

    退款政策：
    - 7天无理由退货（商品完好）
    - 质量问题随时可退，运费由商家承担
    - 已激活的虚拟商品不支持退款

    先理解用户情绪，态度要温和，再用工具处理。
  `,
  tools: [queryOrderTool, submitRefundTool],
  // 退款专员可以再次移交给订单专员（如果需要查订单信息）
  handoffs: [orderAgent],
});

const techAgent = new Agent({
  name: '技术专员',
  instructions: `
    你是技术支持专员，解决产品问题和 Bug 反馈。
    遇到严重 Bug（数据丢失、无法登录）标记为"紧急"。
    对所有反馈都表示感谢。
  `,
  tools: [createTicketTool],
});

// 前台 Agent：接待用户并分流
const frontDeskAgent = new Agent({
  name: '智能客服',
  instructions: `
    你是智能客服前台，负责判断问题类型并分配给专业专员。

    分流规则：
    - 查订单状态、物流进度 → 移交"订单专员"
    - 退款、退货、换货 → 移交"退款专员"
    - 产品 Bug、功能问题 → 移交"技术专员"
    - 其他（营业时间、活动等）→ 自己回答

    移交前说："我来帮您转接对应的专员，请稍候。"
  `,
  // handoffs 列表：前台知道可以移交给哪些专员
  handoffs: [orderAgent, refundAgent, techAgent],
});

// 运行：用户的问题会自动被分流到合适的 Agent
const r1 = await Runner.run(frontDeskAgent, '我的订单 ORD-12345 三天没发货了！');
console.log(r1.finalOutput);

const r2 = await Runner.run(frontDeskAgent, '你们 App 在 iPhone 上点击登录直接崩溃');
console.log(r2.finalOutput);
*/

// ─────────────────────────────────────────────
// 5. 配置 Claude 作为底层模型
//    三种方式，推荐 OpenRouter（兼容性最好）
// ─────────────────────────────────────────────
/*
// ── 方式一：OpenRouter（推荐）──
// OpenRouter 是 API 聚合平台，用同一套 OpenAI 格式访问 Claude、Gemini 等
// 环境变量：
//   OPENAI_API_KEY=<你的 OpenRouter API Key>
//   OPENAI_BASE_URL=https://openrouter.ai/api/v1

const claudeAgent = new Agent({
  name: 'Claude 助手',
  instructions: '你是由 Claude 驱动的助手，擅长长文本分析。',
  model: 'anthropic/claude-opus-4-5',      // OpenRouter 格式
  // 或者：'anthropic/claude-sonnet-4-5'
  // 完整列表：https://openrouter.ai/models
});

// ── 方式二：通过 modelSettings 控制模型参数 ──
const preciseAgent = new Agent({
  name: '精准助手',
  instructions: '你负责提取结构化数据，只返回 JSON，不要其他内容。',
  model: 'gpt-4o',
  modelSettings: {
    temperature: 0,        // 温度为0，输出确定性最强
    maxTokens: 500,        // 限制输出长度
    topP: 1,
  },
});

// ── 方式三：流式输出 ──
const stream = Runner.runStreamed(frontDeskAgent, '我想退款');
for await (const event of stream) {
  // event.type 可能是：
  //   'agent_updated_stream_event'  - Agent 切换（Handoff 发生时）
  //   'raw_response_event'          - 底层模型的流式 token
  //   'run_item_stream_event'       - 工具调用结果
  if (event.type === 'raw_response_event' && event.data.type === 'content_block_delta') {
    process.stdout.write(event.data.delta?.text ?? '');
  }
}
*/

// ─────────────────────────────────────────────
// 6. 与 LangGraph 的对比
// ─────────────────────────────────────────────
/*
  ┌──────────────────┬────────────────────────────┬────────────────────────────┐
  │ 维度             │ OpenAI Agents SDK          │ LangGraph                  │
  ├──────────────────┼────────────────────────────┼────────────────────────────┤
  │ 核心抽象         │ Agent + Handoff            │ Node + Edge（有向图）      │
  │ 学习曲线         │ 低，概念少，上手快         │ 中高，需理解图结构         │
  │ 流程控制         │ AI 自主决策何时移交        │ 开发者明确定义状态转移规则 │
  │ 适合场景         │ 客服、助手、灵活工作流     │ 需精确控制的业务流程       │
  │ 可视化           │ 无内置支持                 │ 支持图结构可视化           │
  │ 状态管理         │ 自动（消息历史）           │ 手动定义 StateGraph        │
  │ 模型支持         │ 原生 OpenAI，可接 Claude   │ 通过 LangChain 多模型      │
  │ 社区成熟度       │ 较新，快速迭代             │ 相对成熟，文档丰富         │
  └──────────────────┴────────────────────────────┴────────────────────────────┘

  选择建议：
  - 快速上线 / 客服场景 → Agents SDK（写更少代码）
  - 复杂业务流程 / 需要精确审批步骤 → LangGraph（更精确可控）
  - 学习 Agent 概念 → Agents SDK（概念最清晰）
*/

// ─────────────────────────────────────────────
// 7. 实战完整示例：电商智能客服分流系统
//    前台 Agent → 判断类型 → 移交给对应专员
// ─────────────────────────────────────────────
/*
import { Agent, tool, Runner } from '@openai/agents';
import { z } from 'zod';

// ── 工具定义 ──

const queryOrderTool = tool({
  name: 'query_order',
  description: '查询订单详情',
  parameters: z.object({ order_id: z.string() }),
  execute: async ({ order_id }) =>
    JSON.stringify({ order_id, status: '已发货', logistics: '顺丰 SF123456，预计明日送达' }),
});

const submitRefundTool = tool({
  name: 'submit_refund',
  description: '提交退款申请',
  parameters: z.object({
    order_id: z.string(),
    reason: z.string(),
    amount: z.number(),
  }),
  execute: async ({ order_id, reason, amount }) =>
    `退款已提交，单号 REF-${Date.now()}，¥${amount}，3-5 个工作日到账`,
});

const createTicketTool = tool({
  name: 'create_ticket',
  description: '创建技术支持工单',
  parameters: z.object({
    title: z.string(),
    severity: z.enum(['低', '中', '高', '紧急']),
  }),
  execute: async ({ title, severity }) =>
    `工单 BUG-${Math.floor(Math.random() * 9999)} 已创建（${severity}），24h 内回复`,
});

// ── Agent 定义（由内向外定义） ──

const orderAgent = new Agent({
  name: '订单专员',
  instructions: '专门处理订单查询和物流跟踪，用 query_order 工具获取真实数据。',
  tools: [queryOrderTool],
});

const refundAgent = new Agent({
  name: '退款专员',
  instructions: '处理退款和售后，先理解情绪，再用工具处理。7天无理由退货，质量问题随时退。',
  tools: [submitRefundTool, queryOrderTool],
});

const techAgent = new Agent({
  name: '技术专员',
  instructions: '处理 Bug 和技术问题，严重问题标记紧急，感谢用户反馈。',
  tools: [createTicketTool],
});

const triageAgent = new Agent({
  name: '智能客服',
  instructions: `
    你是客服前台，判断问题类型并分流：
    - 订单/物流 → 移交"订单专员"
    - 退款/售后 → 移交"退款专员"
    - Bug/技术  → 移交"技术专员"
    - 其他      → 自己回答
  `,
  handoffs: [orderAgent, refundAgent, techAgent],
});

// ── 运行示例 ──

async function runCustomerServiceDemo() {
  const testCases = [
    '我的订单 ORD-12345 发货了吗？',
    '我买的耳机戴了一天就坏了，要退款！',
    '你们 App 在 iOS 17 上打开直接崩溃，每次都这样',
    '你们的客服几点下班？',
  ];

  for (const question of testCases) {
    console.log('\n用户：', question);
    const result = await Runner.run(triageAgent, question);
    console.log('客服：', result.finalOutput);
    console.log('─'.repeat(50));
  }
}

runCustomerServiceDemo().catch(console.error);
*/

// ─────────────────────────────────────────────
// 速查：常用 API
// ─────────────────────────────────────────────
/*
  // Agent 构造参数
  new Agent({
    name: string,               // 必填，Agent 身份标签
    instructions: string,       // 必填，系统提示词
    tools?: Tool[],             // 可调用的工具列表
    handoffs?: Agent[],         // 可移交的专业 Agent 列表
    model?: string,             // 模型 ID，默认 gpt-4o
    modelSettings?: {           // 模型参数
      temperature?, maxTokens?, topP?, ...
    },
  });

  // tool() 参数
  tool({
    name: string,
    description: string,
    parameters: ZodSchema,      // 用 z.object({...}) 定义参数
    execute: async (args) => string | object,
  });

  // Runner.run 参数
  Runner.run(agent, input: string, options?: {
    context?: object,           // 传给工具的上下文
    maxTurns?: number,          // 最大步骤数，默认 10
  });
  // 返回：{ finalOutput: string, newItems: RunItem[] }

  // 流式版本
  Runner.runStreamed(agent, input);
  // 返回 AsyncIterable<StreamEvent>

  // 常用模型 ID（通过 OpenRouter）：
  //   'anthropic/claude-opus-4-5'
  //   'anthropic/claude-sonnet-4-5'
  //   'openai/gpt-4o'
  //   'google/gemini-2.0-flash'

  // 官方文档：https://openai.github.io/openai-agents-js/
  // OpenRouter：https://openrouter.ai/docs
*/

console.log('📖 19_openai_agents_sdk.js 已加载');
console.log('');
console.log('本文件包含完整的 OpenAI Agents SDK 示例代码（注释形式）。');
console.log('取消注释对应章节并安装依赖后即可运行：');
console.log('  npm install @openai/agents zod');
console.log('');
console.log('使用 Claude 模型（推荐通过 OpenRouter）：');
console.log('  export OPENAI_API_KEY=<your_openrouter_key>');
console.log('  export OPENAI_BASE_URL=https://openrouter.ai/api/v1');
console.log('  然后设置 model: "anthropic/claude-opus-4-5"');
