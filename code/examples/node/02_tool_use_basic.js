// 运行: node 02_tool_use_basic.js
// 依赖: npm install @anthropic-ai/sdk
//
// 工具调用（Tool Use）= 让 Claude 能够"调用外部能力"
//
// 类比：Claude 就像一个聪明的助理，但它只能"说话"，不能"动手"。
// 工具调用让它可以下指令给你的代码去执行，然后把结果告诉它，
// 它再根据结果继续回答。
//
// 流程图：
//   你 → [发消息 + 工具列表] → Claude
//   Claude → [决定调用哪个工具，返回 tool_use 块]
//   你 → [执行工具，得到结果]
//   你 → [把结果发回给 Claude]
//   Claude → [根据结果给出最终回答]

import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic();

// ═══════════════════════════════════════════════════════════════
// 第一节：定义工具（Tool Definitions）
// 工具定义 = 给 Claude 看的"工具说明书"
// Claude 读这份说明书来决定要不要调用、怎么传参数
// ═══════════════════════════════════════════════════════════════

/**
 * 工具1：查询天气（mock 数据）
 *
 * JSON Schema 字段解释：
 * - name        工具的唯一名称，Claude 用这个名字来"点菜"
 * - description 告诉 Claude 这个工具是干什么的（越清晰越好）
 * - input_schema 参数的 JSON Schema 规范
 *   - type       固定写 'object'
 *   - properties 每个参数的定义
 *     - type     参数类型：string / number / boolean / array / object
 *     - description 参数含义（帮助 Claude 正确传参）
 *     - enum     可选值列表（Claude 只能传这些值之一）
 *   - required   必填参数列表
 */
const weatherTool = {
  name: 'getWeather',
  description: '查询指定城市的当前天气。返回温度、天气状况和湿度。',
  input_schema: {
    type: 'object',
    properties: {
      city: {
        type: 'string',
        description: '城市名称，例如：北京、上海、广州',
      },
      unit: {
        type: 'string',
        description: '温度单位',
        enum: ['celsius', 'fahrenheit'], // 只允许这两个值
      },
    },
    required: ['city'], // city 是必填，unit 是可选
  },
};

/**
 * 工具2：计算器
 * 让 Claude 能做精确数学运算（AI 数学容易出错，交给代码更可靠）
 */
const calculateTool = {
  name: 'calculate',
  description: '执行数学计算。支持加减乘除、幂运算、取模。',
  input_schema: {
    type: 'object',
    properties: {
      expression: {
        type: 'string',
        description: '数学表达式字符串，例如：(100 + 200) * 3 / 2',
      },
    },
    required: ['expression'],
  },
};

/**
 * 工具3：数据库查询（mock）
 * 模拟从数据库中搜索用户记录
 */
const searchDatabaseTool = {
  name: 'searchDatabase',
  description: '在用户数据库中搜索记录。可按姓名或邮箱查询。',
  input_schema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: '搜索关键词（姓名或邮箱的一部分）',
      },
      limit: {
        type: 'number',
        description: '最多返回多少条记录，默认 5',
      },
    },
    required: ['query'],
  },
};

// 把所有工具放入数组，后面发给 Claude
const tools = [weatherTool, calculateTool, searchDatabaseTool];

// ═══════════════════════════════════════════════════════════════
// 第二节：工具的实际实现（你的业务代码）
// 这部分 Claude 看不到，只有你的代码在执行
// ═══════════════════════════════════════════════════════════════

/**
 * 实际的天气查询逻辑（这里用 mock 数据代替真实 API）
 * @param {string} city
 * @param {'celsius'|'fahrenheit'} unit
 * @returns {object}
 */
function getWeather(city, unit = 'celsius') {
  // mock 数据库
  const weatherData = {
    '北京': { temp: 28, condition: '晴', humidity: 45 },
    '上海': { temp: 32, condition: '多云', humidity: 78 },
    '广州': { temp: 35, condition: '阵雨', humidity: 88 },
    '成都': { temp: 24, condition: '阴', humidity: 65 },
  };

  const data = weatherData[city] ?? { temp: 20, condition: '未知', humidity: 50 };
  const temp = unit === 'fahrenheit' ? data.temp * 9 / 5 + 32 : data.temp;
  const unitSymbol = unit === 'fahrenheit' ? '°F' : '°C';

  return {
    city,
    temperature: `${temp}${unitSymbol}`,
    condition: data.condition,
    humidity: `${data.humidity}%`,
    timestamp: new Date().toISOString(),
  };
}

/**
 * 计算器实现（用 Function 构造器，生产环境应用更安全的解析库）
 * @param {string} expression
 * @returns {object}
 */
function calculate(expression) {
  try {
    // 安全过滤：只允许数字和运算符
    const sanitized = expression.replace(/[^0-9+\-*/().% ]/g, '');
    // eslint-disable-next-line no-new-func
    const result = new Function(`return ${sanitized}`)();
    return { expression, result, success: true };
  } catch (e) {
    return { expression, error: '表达式无效', success: false };
  }
}

/**
 * 数据库搜索（mock）
 * @param {string} query
 * @param {number} limit
 * @returns {object}
 */
function searchDatabase(query, limit = 5) {
  const mockUsers = [
    { id: 1, name: '张三', email: 'zhangsan@example.com', age: 28 },
    { id: 2, name: '李四', email: 'lisi@example.com', age: 32 },
    { id: 3, name: '王五', email: 'wangwu@example.com', age: 25 },
    { id: 4, name: '赵六', email: 'zhaoliu@example.com', age: 41 },
    { id: 5, name: '孙七', email: 'sunqi@example.com', age: 19 },
    { id: 6, name: '周八', email: 'zhouba@example.com', age: 36 },
  ];

  const results = mockUsers
    .filter(u => u.name.includes(query) || u.email.includes(query))
    .slice(0, limit);

  return { query, count: results.length, results };
}

// ═══════════════════════════════════════════════════════════════
// 第三节：分发工具调用（Tool Dispatcher）
// Claude 说"我要调用 getWeather"，这里负责真正执行
// ═══════════════════════════════════════════════════════════════

/**
 * 根据工具名称和参数执行对应的函数
 * @param {string} toolName - Claude 要调用的工具名
 * @param {object} toolInput - Claude 传入的参数
 * @returns {object} - 工具执行结果
 */
function executeTool(toolName, toolInput) {
  console.log(`  ⚙️  执行工具: ${toolName}`);
  console.log(`  📥 参数: ${JSON.stringify(toolInput)}`);

  let result;
  switch (toolName) {
    case 'getWeather':
      result = getWeather(toolInput.city, toolInput.unit);
      break;
    case 'calculate':
      result = calculate(toolInput.expression);
      break;
    case 'searchDatabase':
      result = searchDatabase(toolInput.query, toolInput.limit);
      break;
    default:
      result = { error: `未知工具: ${toolName}` };
  }

  console.log(`  📤 结果: ${JSON.stringify(result)}\n`);
  return result;
}

// ═══════════════════════════════════════════════════════════════
// 第四节：完整工具调用流程（含并行工具调用）
// ═══════════════════════════════════════════════════════════════

/**
 * 完整的工具调用流程演示
 * Claude 可能在一次回复中同时要求调用多个工具（并行）
 *
 * @param {string} userMessage - 用户的问题
 * @returns {Promise<string>} - 最终回答
 */
async function runWithTools(userMessage) {
  console.log(`\n用户: ${userMessage}`);
  console.log('─'.repeat(40));

  // messages 数组，记录完整对话历史
  const messages = [
    { role: 'user', content: userMessage },
  ];

  // ── 循环直到 Claude 给出最终回答 ──────────────────────────
  // 为什么要循环？
  // 因为工具调用可能发生多次：
  //   第1轮：Claude 调用工具A → 我们执行 → 把结果发回
  //   第2轮：Claude 调用工具B → 我们执行 → 把结果发回
  //   第3轮：Claude 给出最终文字回答（stop_reason = 'end_turn'）
  let iteration = 0;
  const MAX_ITERATIONS = 10; // 防止死循环

  while (iteration < MAX_ITERATIONS) {
    iteration++;
    console.log(`\n[第 ${iteration} 轮请求]`);

    const response = await client.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 1024,
      tools,           // 每次都要把工具列表发过去
      messages,
    });

    console.log(`stop_reason: ${response.stop_reason}`);

    // ── 情况A：Claude 给出了最终文字回答 ──────────────────
    if (response.stop_reason === 'end_turn') {
      const textContent = response.content.find(c => c.type === 'text');
      const finalAnswer = textContent?.text ?? '（无文字回答）';
      console.log(`\nClaude 最终回答:\n${finalAnswer}`);
      return finalAnswer;
    }

    // ── 情况B：Claude 要求调用工具 ─────────────────────────
    if (response.stop_reason === 'tool_use') {
      // 把 Claude 的这次回复（包含 tool_use 块）加入历史
      messages.push({ role: 'assistant', content: response.content });

      // 找出所有工具调用块（可能有多个——并行调用）
      const toolUseBlocks = response.content.filter(c => c.type === 'tool_use');
      console.log(`Claude 要调用 ${toolUseBlocks.length} 个工具（并行）`);

      // 并行执行所有工具
      // Promise.all 让多个工具同时跑，不用一个一个等
      const toolResults = await Promise.all(
        toolUseBlocks.map(async (toolUse) => {
          const result = executeTool(toolUse.name, toolUse.input);
          return {
            type: 'tool_result',
            tool_use_id: toolUse.id,    // 必须和 tool_use 的 id 对应
            content: JSON.stringify(result),
          };
        })
      );

      // 把所有工具结果一起发回给 Claude
      messages.push({
        role: 'user',
        content: toolResults,
      });

      // 继续下一轮，等 Claude 处理结果
      continue;
    }

    // 其他 stop_reason（如 max_tokens），退出循环
    console.warn(`未预期的 stop_reason: ${response.stop_reason}`);
    break;
  }

  return '（达到最大迭代次数，未获得最终回答）';
}

// ═══════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('🔧 工具调用（Tool Use）基础示例\n');
  console.log('流程：用户提问 → Claude 决定调用工具 → 你的代码执行 → 结果发回 → Claude 给出最终回答\n');

  try {
    // 示例1：单工具调用
    await runWithTools('北京现在天气怎么样？');

    // 示例2：需要计算的问题
    await runWithTools('帮我计算 (1234 * 5678 + 9012) / 3 等于多少？');

    // 示例3：可能触发并行调用（同时查多个城市天气）
    await runWithTools('帮我比较一下北京和上海今天的天气，哪个更适合出行？');

    // 示例4：数据库查询
    await runWithTools('数据库里有没有姓张的用户？');

    console.log('\n✅ 所有示例完成！');
    console.log('💡 下一步：查看 03_react_agent.js 学习 ReAct Agent 模式');
  } catch (error) {
    console.error('❌ 错误：', error.message);
    process.exit(1);
  }
}

main();
