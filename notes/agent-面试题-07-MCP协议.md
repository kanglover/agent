# MCP（Model Context Protocol）面试题精选

> 来源：模型训练知识（截止 2026 年 1 月），覆盖 MCP 正式发布后的完整技术细节
> 参考：https://modelcontextprotocol.io | https://spec.modelcontextprotocol.io | https://github.com/modelcontextprotocol/servers

---

## 一、MCP 基础概念

**Q1. 什么是 MCP（Model Context Protocol）？**

答题要点：
- MCP 是 Anthropic 于 2024 年 11 月发布的**开放标准协议**，用于统一 LLM 与外部数据源、工具之间的连接方式
- 核心目标：解决 AI 模型与各种上下文来源集成时的"碎片化问题"，用一套协议替代大量定制化集成
- 类比：MCP 就像 AI 世界的 **USB-C 接口**，以前每个工具用不同接口，MCP 提供统一标准插口
- 底层通信：基于 **JSON-RPC 2.0** 协议
- 架构模式：**Client-Server 架构**

---

**Q2. MCP 的三层架构是什么？**

答题要点：

| 角色 | 位置 | 职责 |
|------|------|------|
| **MCP Host（宿主）** | 用户侧应用 | 运行 LLM 的应用程序（如 Claude Desktop、IDE 插件），协调整体交互 |
| **MCP Client（客户端）** | 嵌入在 Host 内 | 与 Server 保持 1:1 连接，处理协议通信 |
| **MCP Server（服务器）** | 独立进程/服务 | 轻量级程序，暴露特定能力（工具/资源/提示词），连接具体数据源 |

数据流：`Host → Client → (JSON-RPC) → Server → 外部数据源/工具`

---

**Q3. MCP 的三大核心原语（Primitives）是什么？**

答题要点：

| 原语 | 控制方 | 作用 |
|------|--------|------|
| **Resources（资源）** | 应用/Host 控制 | 暴露文件、数据库、API 内容等结构化数据，供模型读取上下文 |
| **Tools（工具）** | 模型控制 | 可执行的操作，模型决定何时调用（类比函数调用） |
| **Prompts（提示词）** | 用户控制 | 预定义的提示词模板，可参数化，由用户主动触发 |

关键区别：Tools 是模型主动调用，Resources 是 Host 按需注入上下文，Prompts 是用户触发的模板。

---

**Q4. MCP 支持哪些传输机制（Transport）？**

答题要点：

| 传输方式 | 通信模式 | 适用场景 |
|----------|---------|---------|
| **stdio** | 标准输入输出，Client 启动 Server 子进程 | 本地工具、文件系统、命令行工具 |
| **HTTP + SSE** | HTTP POST 发请求，SSE 推送响应 | 远程 API、云服务、跨机器部署 |

选择原则：本地工具用 stdio（零网络开销）；远程服务用 HTTP+SSE。

---

## 二、MCP vs Function Calling

**Q5. MCP 和 Function Calling（函数调用）有什么区别？**

答题要点：

| 维度 | Function Calling | MCP Tools |
|------|-----------------|-----------|
| **定义位置** | 在 API 请求的 `tools` 参数中内联定义 | 在独立的 MCP Server 中定义，运行时动态发现 |
| **复用性** | 每次请求都要重新传递工具定义 | Server 启动一次，任意 Client 可接入复用 |
| **状态管理** | 无状态，每次调用独立 | Server 可维护连接状态（如数据库连接池） |
| **标准化程度** | 各厂商格式不兼容（OpenAI vs Anthropic） | 统一开放标准，跨模型/跨厂商兼容 |
| **工具发现** | 静态，开发者手动维护 | 动态，Client 通过 `tools/list` 自动发现 |
| **适用场景** | 简单、一次性工具调用 | 复杂、可复用、需跨应用共享的工具生态 |

一句话总结：Function Calling 是"在请求里临时写工具"，MCP 是"独立部署工具服务，按标准协议接入"。

---

**Q6. 什么情况下用 MCP，什么时候用 Function Calling？**

答题要点：

用 **Function Calling** 的场景：
- 工具逻辑简单、一次性使用
- 不需要跨应用共享
- 快速原型、单一 LLM 应用

用 **MCP** 的场景：
- 工具需要在多个 AI 应用间复用（如公司内部工具平台）
- 需要访问本地文件系统、数据库等有状态资源
- 构建标准化工具生态，需要跨模型厂商兼容
- 工具本身复杂，有独立的生命周期管理需求

---

## 三、MCP 协议机制

**Q7. MCP 的连接生命周期是怎样的？**

答题要点，分三个阶段：

1. **初始化（Initialization）**：
   - Client 发送 `initialize` 请求，携带协议版本和能力声明
   - Server 回应支持的能力
   - Client 发送 `initialized` 通知完成握手

2. **操作阶段（Operation）**：
   - 正常的 Request/Response 或 Notification 交互
   - Client 可调用 `tools/list`、`tools/call`、`resources/read`、`prompts/get` 等

3. **关闭（Shutdown）**：
   - 通过 `close()` 方法或进程结束优雅终止

---

**Q8. MCP 中的 Sampling（采样）是什么？**

答题要点：
- Sampling 允许 **Server 向 Client 请求 LLM 补全**，实现 agentic 行为
- 意义：Server 不需要直接持有 LLM API Key，可借用 Host 的模型能力完成复杂推理
- 流程：`Server → sampling/createMessage → Client → Host 中的 LLM → 结果返回 Server`
- 安全考量：Human-in-the-loop，Host 可在转发前让用户审批

---

**Q9. MCP 如何保证安全性？**

答题要点：
- **最小权限原则**：Server 只暴露必要的资源和工具
- **用户确认机制**：敏感操作（文件写入、代码执行）Host 层应要求用户确认
- **传输安全**：HTTP 传输应使用 HTTPS/TLS
- **身份验证**：HTTP+SSE 模式支持标准 HTTP 认证（Bearer Token 等）
- **Prompt Injection 防护**：Server 应对通过资源注入的恶意内容做清洗，Host 不应盲目信任 Server 返回内容
- **Server 隔离**：每个 Server 运行在独立进程，互不影响

---

## 四、MCP 实战

**Q10. 如何用 Python 实现一个最简单的 MCP Server？**

答题要点（Python SDK）：

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("my-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="get_weather",
        description="获取城市天气",
        inputSchema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    )]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        return [types.TextContent(type="text", text=f"{city}今天晴，25°C")]

async def main():
    async with stdio_server() as streams:
        await app.run(
            streams[0], streams[1],
            app.create_initialization_options()
        )
```

关键点：注册 `list_tools` 和 `call_tool` 两个处理器即构成最小 Server。

---

**Q11. MCP 生态中有哪些重要的官方 Server？**

答题要点：

Anthropic 官方维护的参考实现（github.com/modelcontextprotocol/servers）：

| Server | 功能 |
|--------|------|
| `filesystem` | 文件系统读写 |
| `git` | Git 仓库操作 |
| `sqlite` | SQLite 数据库查询 |
| `fetch` | 网页抓取 |
| `memory` | KV 持久化存储 |
| `puppeteer` | 浏览器自动化 |

知名第三方：GitHub、Slack、Google Drive、PostgreSQL、Docker 等均有社区实现。

---

**Q12. MCP 与 LangChain Tools、OpenAI Plugins 的区别？**

| 方案 | 标准化 | 跨框架 | 状态管理 | 协议层 |
|------|-------|--------|---------|--------|
| LangChain Tools | 框架内标准，不跨框架 | 否 | 有限 | Python 抽象层 |
| OpenAI Plugins（已废弃）| OpenAI 私有标准 | 否 | 无 | HTTP+OpenAPI |
| **MCP** | **开放标准** | **是** | **是** | **JSON-RPC 2.0** |

MCP 优势：真正的开放协议，已获 OpenAI、Google DeepMind 等多家采用。

---

## 五、高频追问

**Q13. MCP Server 和 Client 可以在同一进程吗？**
- 可以。stdio 传输时，Host 启动 Server 子进程，Client 嵌入 Host；也可在同一进程内通过内存通道通信（适用于测试）。

**Q14. MCP 支持流式响应吗？**
- 支持。HTTP+SSE 传输天然支持流式；stdio 模式下可通过 Notifications 实现进度推送。

**Q15. 如果 MCP Server 挂了，Host 应该怎么处理？**
- 实现重连逻辑（指数退避）；对用户展示降级提示；日志记录错误；不应因单个 Server 不可用而崩溃整个 Agent。
