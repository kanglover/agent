// 运行: node 03_react_agent.js
// 依赖: npm install @anthropic-ai/sdk
//
// ReAct Agent = Reasoning + Acting（推理 + 行动）
//
// 普通 AI 对话：用户说 → AI 答（一来一回）
// ReAct Agent：用户说 → AI 思考 → AI 行动（调工具）→ 观察结果 → 继续思考 → ... → 最终回答
//
// 类比：就像一个聪明的侦探：
//   1. 接到案子（用户任务）
//   2. 思考：我需要什么线索？（Reasoning）
//   3. 行动：去查档案 / 访问证人（Acting = 调用工具）
//   4. 观察：得到了什么信息？
//   5. 重复 2-4，直到案子告破（任务完成）
//
// MAX_STEPS 保护：防止侦探永远调查下去（无限循环）

import Anthropic from '@anthropic-ai/sdk';
import { promises as fs } from 'fs';
import path from 'path';

const client = new Anthropic();

// ═══════════════════════════════════════════════════════════════
// 工具实现（Agent 可以使用的"能力"）
// ═══════════════════════════════════════════════════════════════

/**
 * 搜索网络（mock）
 * 实际项目中可替换为真实搜索 API（Serper / Tavily 等）
 * @param {string} query
 * @returns {object}
 */
function searchWeb(query) {
  // mock 搜索结果
  const mockResults = {
    'Node.js': {
      title: 'Node.js 官方文档',
      snippet: 'Node.js 是基于 Chrome V8 引擎的 JavaScript 运行时，适合构建高性能网络应用。',
      url: 'https://nodejs.org',
    },
    'Python': {
      title: 'Python 官方文档',
      snippet: 'Python 是一种简洁易读的编程语言，广泛用于数据科学、AI 和后端开发。',
      url: 'https://python.org',
    },
    'Claude API': {
      title: 'Anthropic 开发者文档',
      snippet: 'Claude API 提供对话、工具调用、视觉等能力，支持流式输出。',
      url: 'https://docs.anthropic.com',
    },
  };

  // 在 mock 数据中找最匹配的
  for (const [key, value] of Object.entries(mockResults)) {
    if (query.toLowerCase().includes(key.toLowerCase())) {
      return { query, results: [value], total: 1 };
    }
  }

  return {
    query,
    results: [{ title: '搜索结果', snippet: `关于 "${query}" 的相关信息（mock）`, url: '#' }],
    total: 1,
  };
}

/**
 * 读取文件内容
 * @param {string} filePath
 * @returns {Promise<object>}
 */
async function readFile(filePath) {
  try {
    const absolutePath = path.resolve(filePath);
    const content = await fs.readFile(absolutePath, 'utf-8');
    return {
      path: absolutePath,
      content: content.slice(0, 2000), // 只返回前 2000 字符，避免 token 爆炸
      size: content.length,
      truncated: content.length > 2000,
    };
  } catch (error) {
    return { path: filePath, error: `读取失败: ${error.message}` };
  }
}

/**
 * 写入文件
 * @param {string} filePath
 * @param {string} content
 * @returns {Promise<object>}
 */
async function writeFile(filePath, content) {
  try {
    const absolutePath = path.resolve(filePath);
    await fs.writeFile(absolutePath, content, 'utf-8');
    return { path: absolutePath, bytesWritten: content.length, success: true };
  } catch (error) {
    return { path: filePath, error: `写入失败: ${error.message}`, success: false };
  }
}

/**
 * 数学计算
 * @param {string} expression
 * @returns {object}
 */
function calculate(expression) {
  try {
    const sanitized = expression.replace(/[^0-9+\-*/().% ]/g, '');
    // eslint-disable-next-line no-new-func
    const result = new Function(`return ${sanitized}`)();
    return { expression, result, success: true };
  } catch {
    return { expression, error: '表达式无效', success: false };
  }
}

// ═══════════════════════════════════════════════════════════════
// 工具定义（给 Claude 看的说明书）
// ═══════════════════════════════════════════════════════════════

const TOOL_DEFINITIONS = [
  {
    name: 'searchWeb',
    description: '搜索互联网获取信息。适合查询新闻、技术文档、事实性问题。',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '搜索关键词' },
      },
      required: ['query'],
    },
  },
  {
    name: 'readFile',
    description: '读取本地文件内容。适合查看代码、配置文件、文本文档。',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '文件路径（绝对路径或相对路径）' },
      },
      required: ['path'],
    },
  },
  {
    name: 'writeFile',
    description: '将内容写入本地文件。适合保存结果、生成报告、创建代码。',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '文件路径' },
        content: { type: 'string', description: '要写入的文本内容' },
      },
      required: ['path', 'content'],
    },
  },
  {
    name: 'calculate',
    description: '执行数学计算。对精确计算使用这个工具而不是自己心算。',
    input_schema: {
      type: 'object',
      properties: {
        expression: { type: 'string', description: '数学表达式，如 (100 * 365) / 12' },
      },
      required: ['expression'],
    },
  },
];

// ═══════════════════════════════════════════════════════════════
// ReActAgent 类
// ═══════════════════════════════════════════════════════════════

class ReActAgent {
  /**
   * @param {object} options
   * @param {number} options.maxSteps - 最多执行多少步（防无限循环）
   * @param {boolean} options.verbose  - 是否打印详细日志
   */
  constructor({ maxSteps = 10, verbose = true } = {}) {
    this.maxSteps = maxSteps;         // 最大步数保护
    this.verbose = verbose;           // 日志开关
    this.stepCount = 0;               // 当前步数计数器
    this.stepLog = [];                // 步骤追踪日志

    // 工具实际执行函数的映射表
    this.toolExecutors = {
      searchWeb: (input) => searchWeb(input.query),
      readFile: (input) => readFile(input.path),
      writeFile: (input) => writeFile(input.path, input.content),
      calculate: (input) => calculate(input.expression),
    };
  }

  /**
   * 日志输出（仅 verbose 模式）
   * @param {string} message
   * @param {'info'|'tool'|'result'|'final'} type
   */
  log(message, type = 'info') {
    if (!this.verbose) return;
    const icons = { info: '💭', tool: '🔧', result: '📋', final: '✅' };
    console.log(`${icons[type] ?? '·'} ${message}`);
  }

  /**
   * 执行单个工具调用
   * @param {object} toolUse - Claude 返回的 tool_use 块
   * @returns {Promise<object>} - tool_result 块
   */
  async executeTool(toolUse) {
    const executor = this.toolExecutors[toolUse.name];

    if (!executor) {
      return {
        type: 'tool_result',
        tool_use_id: toolUse.id,
        content: JSON.stringify({ error: `未知工具: ${toolUse.name}` }),
        is_error: true,
      };
    }

    this.log(`调用工具: ${toolUse.name}(${JSON.stringify(toolUse.input)})`, 'tool');

    try {
      const result = await executor(toolUse.input);
      this.log(`工具结果: ${JSON.stringify(result).slice(0, 200)}`, 'result');

      // 记录到步骤日志
      this.stepLog.push({
        step: this.stepCount,
        tool: toolUse.name,
        input: toolUse.input,
        result,
      });

      return {
        type: 'tool_result',
        tool_use_id: toolUse.id,
        content: JSON.stringify(result),
      };
    } catch (error) {
      return {
        type: 'tool_result',
        tool_use_id: toolUse.id,
        content: JSON.stringify({ error: error.message }),
        is_error: true,
      };
    }
  }

  /**
   * 运行 Agent，处理给定任务直到完成或达到最大步数
   *
   * @param {string} task - 用户的任务描述
   * @returns {Promise<{answer: string, steps: number, log: Array}>}
   */
  async run(task) {
    console.log(`\n${'═'.repeat(50)}`);
    console.log(`🤖 ReAct Agent 启动`);
    console.log(`📝 任务: ${task}`);
    console.log(`🛡️  最大步数: ${this.maxSteps}`);
    console.log('═'.repeat(50));

    this.stepCount = 0;
    this.stepLog = [];

    // 对话历史
    const messages = [
      { role: 'user', content: task },
    ];

    // System prompt：告诉 Claude 它是一个 ReAct Agent
    const systemPrompt = `你是一个能够使用工具解决问题的智能助理（ReAct Agent）。

工作方式：
1. 仔细分析用户的任务
2. 判断是否需要使用工具获取信息
3. 调用合适的工具（可以多次调用）
4. 根据工具结果继续思考和行动
5. 得出最终结论并给出清晰的回答

注意：
- 优先使用工具获取准确信息，而不是依赖记忆
- 数学计算必须使用 calculate 工具，不要自己心算
- 回答使用中文`;

    // ── Agent 主循环 ────────────────────────────────────────────
    while (this.stepCount < this.maxSteps) {
      this.stepCount++;
      this.log(`\n--- 步骤 ${this.stepCount}/${this.maxSteps} ---`, 'info');

      // 向 Claude 发送请求
      const response = await client.messages.create({
        model: 'claude-opus-4-5',
        max_tokens: 1024,
        system: systemPrompt,
        tools: TOOL_DEFINITIONS,
        messages,
      });

      this.log(`stop_reason: ${response.stop_reason}`, 'info');

      // ── 任务完成：Claude 给出最终回答 ───────────────────────
      if (response.stop_reason === 'end_turn') {
        const textContent = response.content.find(c => c.type === 'text');
        const answer = textContent?.text ?? '（Agent 未给出文字回答）';

        this.log(`\n最终回答:\n${answer}`, 'final');
        console.log(`\n📊 Agent 统计: 共执行 ${this.stepCount} 步，调用工具 ${this.stepLog.length} 次`);

        return {
          answer,
          steps: this.stepCount,
          toolCalls: this.stepLog.length,
          log: this.stepLog,
        };
      }

      // ── 需要调用工具 ─────────────────────────────────────────
      if (response.stop_reason === 'tool_use') {
        // 把 Claude 的回复（含 tool_use 块）加入历史
        messages.push({ role: 'assistant', content: response.content });

        // 找出所有工具调用
        const toolUseBlocks = response.content.filter(c => c.type === 'tool_use');
        this.log(`本步骤调用 ${toolUseBlocks.length} 个工具`, 'info');

        // 并行执行所有工具调用
        const toolResults = await Promise.all(
          toolUseBlocks.map(toolUse => this.executeTool(toolUse))
        );

        // 把工具结果发回给 Claude
        messages.push({ role: 'user', content: toolResults });
        continue;
      }

      // 意外的 stop_reason
      this.log(`意外的 stop_reason: ${response.stop_reason}，终止循环`, 'info');
      break;
    }

    // 达到最大步数，强制结束
    console.warn(`\n⚠️  已达到最大步数 (${this.maxSteps})，强制结束`);
    return {
      answer: `（已达到最大步数 ${this.maxSteps}，任务未完成）`,
      steps: this.stepCount,
      toolCalls: this.stepLog.length,
      log: this.stepLog,
    };
  }
}

// ═══════════════════════════════════════════════════════════════
// 主函数：运行几个示例任务
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('🚀 ReAct Agent 示例\n');

  // 创建 Agent 实例
  const agent = new ReActAgent({ maxSteps: 10, verbose: true });

  try {
    // 任务1：需要搜索 + 计算的复合任务
    await agent.run(
      '帮我了解一下 Node.js 是什么，并计算如果我每天学习 2 小时，学完 100 小时需要多少天？'
    );

    // 任务2：纯计算任务
    await agent.run(
      '一个矩形的面积是 1234.5 平方米，如果长是 45.6 米，宽是多少？用工具精确计算。'
    );

    console.log('\n✅ 所有任务完成！');
    console.log('💡 下一步：查看 04_memory_management.js 学习对话记忆管理');
  } catch (error) {
    console.error('❌ 错误：', error.message);
    process.exit(1);
  }
}

main();
