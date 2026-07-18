/**
 * 18_mcp_client.js
 * ================
 * MCP（Model Context Protocol）客户端完整实现
 *
 * 【什么是 MCP？】
 * MCP 是 Anthropic 提出的开放协议，让 AI 模型能安全地访问外部工具和数据源。
 * 类比：USB 接口标准——任何设备只要符合 USB 规范就能即插即用。
 *       MCP 让任何工具只要符合规范，就能被任何支持 MCP 的 AI 客户端使用。
 *
 * 【三层架构】
 *   你的程序（MCP Client）
 *       ↕ stdio / HTTP
 *   MCP Server（工具提供方，如文件系统、数据库、浏览器）
 *       ↕
 *   外部资源（真实文件、真实数据库、真实网页…）
 *
 * 安装：npm install @modelcontextprotocol/sdk @anthropic-ai/sdk
 * 运行：node 18_mcp_client.js
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

// ─────────────────────────────────────────────
// 1. 创建并连接 MCP Client（通过 spawn 子进程）
//    StdioClientTransport 会启动子进程作为 MCP Server，
//    通过 stdin/stdout 管道通信——就像两个程序用"纸条传话"
// ─────────────────────────────────────────────

/**
 * 连接到一个 MCP Server
 *
 * @param {string} serverLabel  - 给这个 Server 取个名字（调试用）
 * @param {string} command      - 启动命令，例如 'npx'
 * @param {string[]} args       - 命令参数，例如 ['-y', '@modelcontextprotocol/server-filesystem', '/tmp']
 * @param {object} extraEnv     - 额外的环境变量（如 API Key）
 * @returns {Promise<Client>}   - 已连接的 MCP Client 实例
 */
async function connectToMCPServer(serverLabel, command, args, extraEnv = {}) {
  // StdioClientTransport：启动子进程，通过 stdin/stdout 通信
  const transport = new StdioClientTransport({
    command,
    args,
    env: {
      ...process.env,  // 继承父进程环境变量（包括 PATH）
      ...extraEnv,     // 覆盖或新增特定变量
    },
  });

  const client = new Client(
    {
      name: `demo-client-${serverLabel}`,  // 客户端标识（对方会看到）
      version: '1.0.0',
    },
    {
      capabilities: {
        tools: {},      // 声明：我想使用工具
        resources: {},  // 声明：我想读取资源
        prompts: {},    // 声明：我想获取提示词模板
      },
    }
  );

  // connect() 发起握手，协商协议版本和能力
  await client.connect(transport);
  console.log(`[${serverLabel}] ✓ 已连接`);

  return client;
}

// ─────────────────────────────────────────────
// 2. 动态发现工具 —— client.listTools()
//    MCP 最大优势：不需要提前知道 Server 有什么，运行时自动发现
//    类比：插上 USB 设备后，电脑自动列出它的功能
// ─────────────────────────────────────────────

async function listTools(client, serverLabel) {
  // listTools() 返回 { tools: [ { name, description, inputSchema } ] }
  const response = await client.listTools();
  const tools = response.tools;

  console.log(`\n[${serverLabel}] 发现 ${tools.length} 个工具：`);
  for (const t of tools) {
    // inputSchema 是 JSON Schema，描述工具接受的参数
    const paramKeys = Object.keys(t.inputSchema?.properties ?? {}).join(', ');
    console.log(`  - ${t.name}(${paramKeys})  →  ${t.description ?? '无描述'}`);
  }

  return tools;
}

// ─────────────────────────────────────────────
// 3. 调用工具 —— client.callTool(name, args)
//    注意：参数格式必须符合工具的 inputSchema
// ─────────────────────────────────────────────

async function callTool(client, toolName, toolArgs) {
  console.log(`\n→ 调用工具 [${toolName}]，参数：`, JSON.stringify(toolArgs));

  const result = await client.callTool({
    name: toolName,
    arguments: toolArgs,  // 注意：MCP SDK 用 arguments，不是 params
  });

  // result.content 是数组，元素类型：
  //   { type: 'text', text: '...' }         文本
  //   { type: 'image', data: '...', mimeType: '...' }  图片（base64）
  //   { type: 'resource', resource: {...} }  资源引用

  const textParts = result.content
    .filter(c => c.type === 'text')
    .map(c => c.text);

  const output = textParts.join('\n');
  console.log(`← 工具返回：${output.slice(0, 200)}${output.length > 200 ? '…' : ''}`);

  if (result.isError) {
    console.error('  [!] 工具返回了错误标志');
  }

  return result;
}

// ─────────────────────────────────────────────
// 4. 资源读取 —— listResources / readResource
//    Resources = MCP 中的"只读数据视图"
//    工具（Tools）用于执行操作；资源（Resources）用于读取数据
//    类比：工具是"动词"，资源是"名词"
// ─────────────────────────────────────────────

async function listAndReadResources(client, serverLabel) {
  // 列出所有资源（部分 Server 不支持，catch 掉）
  let resources = [];
  try {
    const response = await client.listResources();
    resources = response.resources;
    console.log(`\n[${serverLabel}] 发现 ${resources.length} 个资源：`);
    for (const r of resources) {
      console.log(`  - ${r.uri}  (${r.mimeType ?? '未知类型'})  ${r.name ?? ''}`);
    }
  } catch (err) {
    console.log(`[${serverLabel}] 不支持 listResources：${err.message}`);
    return;
  }

  // 读取第一个资源的内容
  if (resources.length > 0) {
    const uri = resources[0].uri;
    console.log(`\n读取资源：${uri}`);
    try {
      const result = await client.readResource({ uri });
      for (const content of result.contents) {
        if (content.text) {
          console.log(`内容（文本）：${content.text.slice(0, 200)}`);
        } else if (content.blob) {
          console.log(`内容（二进制）：${content.blob.length} 字节`);
        }
      }
    } catch (err) {
      console.log(`读取失败：${err.message}`);
    }
  }
}

// ─────────────────────────────────────────────
// 5. 将 MCP 工具转换为 Anthropic tools 格式
//    这是让 Claude 能"看懂" MCP 工具的关键转换步骤
//
//    MCP 格式：      { name, description, inputSchema: { type, properties } }
//    Anthropic 格式：{ name, description, input_schema: { type, properties } }
//    区别：字段名 inputSchema → input_schema（下划线 vs 驼峰）
// ─────────────────────────────────────────────

function convertMCPToolsToAnthropicFormat(mcpTools, serverPrefix = '') {
  return mcpTools.map(tool => ({
    // 加前缀防止多个 Server 之间工具同名冲突
    name: serverPrefix ? `${serverPrefix}__${tool.name}` : tool.name,
    description: tool.description ?? '',
    input_schema: tool.inputSchema ?? { type: 'object', properties: {} },
  }));
}

// ─────────────────────────────────────────────
// 6. Agent Loop —— 把 MCP 工具给 Claude 使用
//    完整的 ReAct 循环：
//    思考（Claude 决定调工具）→ 行动（MCP 执行工具）→ 观察（结果返回 Claude）→ 循环
// ─────────────────────────────────────────────

/**
 * 完整的 Claude + MCP 工具调用循环
 *
 * @param {Client} mcpClient        - 已连接的 MCP Client
 * @param {string} userMessage      - 用户问题
 * @param {number} maxSteps         - 最大循环次数（防死循环）
 */
async function agentLoopWithMCP(mcpClient, userMessage, maxSteps = 10) {
  console.log('\n' + '═'.repeat(60));
  console.log('用户：', userMessage);
  console.log('═'.repeat(60));

  // 步骤1：获取 MCP 工具列表并转换格式
  const mcpToolsResponse = await mcpClient.listTools();
  const anthropicTools = convertMCPToolsToAnthropicFormat(mcpToolsResponse.tools);

  console.log(`\n可用工具：${anthropicTools.map(t => t.name).join(', ')}`);

  // 步骤2：初始化消息历史
  const messages = [{ role: 'user', content: userMessage }];

  // 步骤3：循环，直到 Claude 不再调用工具
  for (let step = 1; step <= maxSteps; step++) {
    console.log(`\n--- 步骤 ${step} ---`);

    // 调用 Claude（附带工具列表）
    const response = await anthropic.messages.create({
      model: 'claude-opus-4-5',
      max_tokens: 4096,
      tools: anthropicTools,
      messages,
    });

    // 打印 Claude 的文字思考
    const textBlocks = response.content.filter(b => b.type === 'text');
    if (textBlocks.length > 0) {
      console.log('Claude：', textBlocks.map(b => b.text).join(''));
    }

    // 如果 Claude 不再调用工具，对话结束
    if (response.stop_reason === 'end_turn') {
      console.log('\n✓ Agent 完成任务');
      return textBlocks.map(b => b.text).join('');
    }

    // Claude 需要调用工具（stop_reason === 'tool_use'）
    // 把 Claude 的回复加入历史
    messages.push({ role: 'assistant', content: response.content });

    // 处理所有工具调用（Claude 可能同时请求多个工具）
    const toolResults = [];
    const toolUseBlocks = response.content.filter(b => b.type === 'tool_use');

    for (const toolUse of toolUseBlocks) {
      console.log(`\n调用工具：${toolUse.name}`);
      console.log('参数：', JSON.stringify(toolUse.input, null, 2));

      let resultContent;
      let isError = false;

      try {
        // 通过 MCP Client 实际执行工具
        const mcpResult = await mcpClient.callTool({
          name: toolUse.name,                // 原始工具名（不含前缀）
          arguments: toolUse.input,
        });

        resultContent = mcpResult.content
          .filter(c => c.type === 'text')
          .map(c => c.text)
          .join('\n');
        isError = mcpResult.isError ?? false;

      } catch (err) {
        resultContent = `工具执行失败：${err.message}`;
        isError = true;
        console.error('工具错误：', err.message);
      }

      console.log('工具结果：', resultContent.slice(0, 150));

      // 构造 Anthropic 要求的 tool_result 格式
      toolResults.push({
        type: 'tool_result',
        tool_use_id: toolUse.id,    // 必须和 tool_use block 的 id 匹配
        content: resultContent,
        is_error: isError,
      });
    }

    // 把工具结果加入消息历史，继续下一轮
    messages.push({ role: 'user', content: toolResults });
  }

  throw new Error(`超过最大步骤数 ${maxSteps}，Agent 可能陷入循环`);
}

// ─────────────────────────────────────────────
// 7. 多 Server 工具合并管理器
//    场景：同时连接文件系统 Server + 数据库 Server + 搜索 Server
//    用前缀区分不同来源的工具
// ─────────────────────────────────────────────

class MultiServerMCPManager {
  constructor() {
    // Map<前缀, { client, mcpTools }> —— 存储所有 Server 的连接和工具
    this.servers = new Map();
    // 合并后所有工具（Anthropic 格式，带前缀）
    this.allAnthropicTools = [];
    // 工具名（带前缀） → { client, 原始工具名 } 的映射表（路由用）
    this.toolRouteMap = new Map();
  }

  /**
   * 添加并连接一个 MCP Server
   * @param {string} prefix    - 工具名前缀，例如 'fs'、'db'
   * @param {string} command   - 启动命令
   * @param {string[]} args    - 命令参数
   * @param {object} env       - 额外环境变量
   */
  async addServer(prefix, command, args, env = {}) {
    const client = await connectToMCPServer(prefix, command, args, env);
    const mcpTools = await listTools(client, prefix);
    const anthropicTools = convertMCPToolsToAnthropicFormat(mcpTools, prefix);

    // 存储连接信息
    this.servers.set(prefix, { client, mcpTools });

    // 注册工具路由（前缀__工具名 → 哪个 client + 原始工具名）
    for (const tool of anthropicTools) {
      this.toolRouteMap.set(tool.name, {
        client,
        originalName: tool.name.replace(`${prefix}__`, ''),
      });
    }

    // 加入合并工具列表
    this.allAnthropicTools.push(...anthropicTools);

    console.log(`[MultiServerManager] ✓ 注册 "${prefix}"，当前共 ${this.allAnthropicTools.length} 个工具`);
  }

  /**
   * 调用工具（自动路由到对应 Server）
   * @param {string} toolNameWithPrefix  - 带前缀的工具名，如 'fs__read_file'
   * @param {object} args                - 工具参数
   */
  async callTool(toolNameWithPrefix, args) {
    const route = this.toolRouteMap.get(toolNameWithPrefix);
    if (!route) {
      throw new Error(`未知工具：${toolNameWithPrefix}。可用工具：${[...this.toolRouteMap.keys()].join(', ')}`);
    }
    return callTool(route.client, route.originalName, args);
  }

  /** 完整的 Agent Loop（使用所有 Server 的合并工具） */
  async agentLoop(userMessage, maxSteps = 10) {
    console.log('\n' + '═'.repeat(60));
    console.log('用户：', userMessage);
    console.log(`可用工具（${this.allAnthropicTools.length} 个）：`, this.allAnthropicTools.map(t => t.name).join(', '));
    console.log('═'.repeat(60));

    const messages = [{ role: 'user', content: userMessage }];

    for (let step = 1; step <= maxSteps; step++) {
      console.log(`\n--- 步骤 ${step} ---`);

      const response = await anthropic.messages.create({
        model: 'claude-opus-4-5',
        max_tokens: 4096,
        tools: this.allAnthropicTools,
        messages,
      });

      const textBlocks = response.content.filter(b => b.type === 'text');
      if (textBlocks.length > 0) console.log('Claude：', textBlocks.map(b => b.text).join(''));

      if (response.stop_reason === 'end_turn') {
        console.log('\n✓ 完成');
        return textBlocks.map(b => b.text).join('');
      }

      messages.push({ role: 'assistant', content: response.content });

      const toolResults = [];
      for (const block of response.content.filter(b => b.type === 'tool_use')) {
        console.log(`调用：${block.name}，参数：`, JSON.stringify(block.input));
        let content;
        let isError = false;
        try {
          const result = await this.callTool(block.name, block.input);
          content = result.content.filter(c => c.type === 'text').map(c => c.text).join('\n');
          isError = result.isError ?? false;
        } catch (err) {
          content = `错误：${err.message}`;
          isError = true;
        }
        console.log('结果：', content.slice(0, 100));
        toolResults.push({ type: 'tool_result', tool_use_id: block.id, content, is_error: isError });
      }
      messages.push({ role: 'user', content: toolResults });
    }
    throw new Error(`超过最大步骤数 ${maxSteps}`);
  }

  /** 关闭所有连接 */
  async closeAll() {
    for (const [prefix, { client }] of this.servers) {
      await client.close().catch(() => {});
      console.log(`[${prefix}] 连接已关闭`);
    }
    this.servers.clear();
    this.allAnthropicTools = [];
    this.toolRouteMap.clear();
  }
}

// ─────────────────────────────────────────────
// 8. 主函数：演示完整流程
// ─────────────────────────────────────────────

async function main() {
  console.log('🚀 MCP Client 示例集\n');
  console.log('本示例演示 MCP Client 的核心 API 和架构。');
  console.log('完整运行需要安装 @modelcontextprotocol/server-filesystem 等 Server。\n');

  console.log('=== 架构说明 ===');
  console.log(`
  // 方式一：单个 Server（最常用）
  const client = await connectToMCPServer(
    'filesystem',                              // 给这个 Server 起个名字
    'npx',                                     // 用 npx 启动
    ['-y', '@modelcontextprotocol/server-filesystem', '/tmp']
  );

  const tools = await listTools(client, 'filesystem');
  // 发现：read_file, write_file, list_directory, create_directory ...

  // 直接调用工具
  await callTool(client, 'write_file', { path: '/tmp/hello.txt', contents: 'Hello!' });
  await callTool(client, 'read_file', { path: '/tmp/hello.txt' });

  // 让 Claude 自主调用工具（Agent Loop）
  const answer = await agentLoopWithMCP(
    client,
    '帮我在 /tmp 目录创建一个 todo.txt，写入 3 条待办事项'
  );

  await client.close();
  `);

  console.log('\n=== 多 Server 合并架构 ===');
  console.log(`
  const manager = new MultiServerMCPManager();

  // 同时连接多个 Server
  await manager.addServer('fs', 'npx', ['-y', '@modelcontextprotocol/server-filesystem', '/data']);
  await manager.addServer('db', 'npx', ['-y', '@modelcontextprotocol/server-sqlite', 'app.db']);
  await manager.addServer('web', 'npx', ['-y', '@modelcontextprotocol/server-fetch']);

  // Claude 会自动在 fs__、db__、web__ 工具中选择合适的
  await manager.agentLoop(
    '查询数据库里销量最高的产品，把结果写入 /data/report.txt，然后 fetch 官网确认价格'
  );

  await manager.closeAll();
  `);

  console.log('\n=== 常用 MCP Server 速查 ===');
  console.log('  文件系统：npx @modelcontextprotocol/server-filesystem <允许的目录>');
  console.log('  SQLite：  npx @modelcontextprotocol/server-sqlite <数据库文件.db>');
  console.log('  HTTP请求：npx @modelcontextprotocol/server-fetch');
  console.log('  GitHub：  npx @modelcontextprotocol/server-github');
  console.log('  Brave搜索：npx @modelcontextprotocol/server-brave-search');
  console.log('\n  更多 Server：https://github.com/modelcontextprotocol/servers');
  console.log('\n✅ MCP Client 架构展示完成！');
}

// 导出供其他文件使用
export {
  connectToMCPServer,
  listTools,
  callTool,
  listAndReadResources,
  convertMCPToolsToAnthropicFormat,
  agentLoopWithMCP,
  MultiServerMCPManager,
};

main().catch(console.error);
