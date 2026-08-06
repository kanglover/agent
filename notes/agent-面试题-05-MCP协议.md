# MCP（Model Context Protocol）面试题精选（20 题）

> 涵盖 MCP 基础概念、三层架构、三类原语、传输层、代码实现、安全与生产部署，以及与 Function Calling 的本质区别。

---

## 一、基础概念（Q1–Q3）

**Q1. 什么是 MCP（Model Context Protocol）？它要解决什么核心问题？**

MCP 是 Anthropic 于 2024 年 11 月发布的开放标准协议，用于标准化 AI 应用与外部数据源、工具的连接方式。

核心问题："集成爆炸"——N 个 AI 应用 × M 个外部系统 = N×M 个独立集成，各自维护。MCP 把这个问题降为 N+M。

MCP 提供统一的"USB 接口"，任何 MCP Server 只需实现一次协议，就能被所有 MCP Host 使用。

四大核心价值：
1. 标准化工具接口
2. 安全隔离（Server 独立进程）
3. 可复用性（一套 Server 对接所有 Host）
4. 生态效应（类比 NPM）

---

**Q2. MCP 和 Function Calling 有什么本质区别？**

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 范围 | 单次 API 调用内 | 跨应用的持久连接 |
| 绑定关系 | 工具与 LLM 强绑定 | 工具与 LLM 解耦 |
| 传输层 | 无（同进程） | JSON-RPC 2.0（进程间/网络） |
| 状态 | 无状态 | 有状态（持久连接） |
| 发现机制 | 手动定义 | 动态能力发现（list_tools） |

类比：Function Calling 是"直拨电话"，MCP 是"电话总机"——统一入口，按需路由。

---

**Q3. MCP 和 REST API 相比有什么独特优势？**

- REST API 缺乏 LLM 可自动发现工具的机制；MCP 支持 `list_tools` 动态发现
- REST 文档对人类友好但不是机器可读的结构；MCP 工具用 JSON Schema 描述，LLM 直接理解
- 所有 MCP Server 使用统一消息格式（JSON-RPC 2.0），无需每个工具单独适配
- MCP 支持双向通信（Server 主动推送通知），REST 是单向请求-响应
- MCP 的 Resources 原语允许 Server 提供结构化上下文，LLM 在操作前就能了解系统状态

---

## 二、架构原理（Q4–Q6）

**Q4. MCP 的三层架构（Host、Client、Server）各自职责是什么？**

- **Host**（宿主）：运行 AI 模型的应用（如 Claude Desktop、VS Code），管理一个或多个 MCP Client，决定哪些 Server 可连接，处理用户界面
- **Client**（客户端）：内嵌在 Host 中，与单个 MCP Server 维持 1:1 连接，负责协议层消息序列化/反序列化，处理连接生命周期
- **Server**（服务端）：独立进程或远程服务，暴露 Tools/Resources/Prompts，处理工具调用请求并返回结果

数据流：用户 → Host → Client → Server → 返回结果 → Host → 用户

---

**Q5. MCP 支持哪三类原语（Resources、Tools、Prompts）？**

| 原语 | 控制方 | 是否只读 | 典型用途 |
|------|--------|---------|---------|
| Resources | AI 模型 | 是 | 读取数据、上下文注入（如文件内容、数据库记录） |
| Tools | AI 模型 | 否 | 执行操作、调用 API（有副作用，如写文件、发邮件） |
| Prompts | 用户 | - | 触发预设工作流（如代码审查模板、文档摘要模板） |

---

**Q6. MCP 的传输层支持哪几种协议？各适用于什么场景？**

消息格式统一为 JSON-RPC 2.0，传输层有三种：

| 方式 | 场景 | 特点 |
|------|------|------|
| stdio | 本地工具（命令行子进程） | 最常用，延迟极低，安全性最高（进程隔离，无网络暴露） |
| HTTP + SSE | 远程服务、多客户端共享 | 可穿越 NAT，支持 Server 主动推送，广泛基础设施支持 |
| WebSocket（新版） | 高频交互、实时协作 | 全双工，最低延迟，基础设施要求高 |

---

## 三、代码实现（Q7–Q8）

**Q7. 如何用 Python 实现一个简单的 MCP Server？**

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

app = Server("my-tool-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的当前天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        weather_data = await fetch_weather_api(city)
        return [TextContent(type="text", text=f"{city}: {weather_data}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
```

Claude Desktop 配置（`claude_desktop_config.json`）：
```json
{
  "mcpServers": {
    "my-tool": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

---

**Q8. 如何在 Multi-Agent 系统中让多个 Agent 共享同一个 MCP Server？**

- 使用 SSE 传输将 MCP Server 部署为独立服务
- 每个 Agent 创建自己的 MCP Client 实例，连接到共享的 Server URL
- 多个 Agent 并发调用时，Server 通过异步处理实现连接级并发
- 优势：工具逻辑集中管理、Agent 代码简化、Server 可独立扩容、安全边界清晰

---

## 四、安全与生产部署（Q9–Q11）

**Q9. MCP 的安全模型是什么？**

- 用户显式授权（Host 决定哪些 Server 可连接）
- 最小权限原则（Server 只暴露必要操作）
- 审计日志（所有工具调用可记录）
- stdio 传输提供进程级隔离，LLM 无法直接访问系统

---

**Q10. 生产环境如何部署 MCP Server？**

- 容器化（Docker）部署，保证环境一致性
- 添加认证中间件（如 API Key 验证）
- 集成日志监控
- MCP Server 尽量设计为无状态，需要状态时做请求级隔离并对共享资源加锁

---

**Q11. 如何测试 MCP Server？**

- 使用官方 MCP Inspector 工具进行交互测试
- 对各工具函数编写单元测试
- 先手动在命令行运行 Server 确认启动正常
- Claude Desktop 启动失败时无显眼提示，日志在 Mac 的 `~/Library/Logs/Claude/` 下

---

## 五、高频扩展题（Q12–Q17）

**Q12. MCP 支持哪些开发语言？**

官方提供 Python SDK 和 TypeScript SDK，社区有其他语言版本。

---

**Q13. MCP 和 OpenAI Tool Calling 能互通吗？**

不能直接互通，有社区适配层可以转换，但官方标准不同。

---

**Q14. 什么是 MCP 的 Resource Subscription？**

Server 主动推送资源变更（如文件监听），类似 WebSocket 通知。客户端订阅某个 Resource 后，Server 在资源变更时会主动发送通知。

---

**Q15. MCP Server 如何处理并发？**

异步处理（asyncio），连接级并发隔离。每个连接维护独立上下文，避免并发状态冲突。

---

**Q16. MCP 和 A2A（Agent-to-Agent）协议的关系？**

- MCP 是"工具协议"（Agent 调用外部工具）
- A2A 是"Agent 间通信协议"（Agent 互调）

两者互补，不冲突。MCP 解决 Agent 与外部工具的连接问题，A2A 解决 Agent 与 Agent 的协作问题。

---

**Q17. MCP 协议版本不匹配怎么办？**

- 保持 Claude Desktop 和 MCP SDK 版本同步更新
- Server 的 `initialize` 响应中声明支持的协议版本
- 客户端连接时协商双方都支持的版本

---

## 六、工程踩坑（Q18–Q20）

**Q18. 工具描述写太简单会有什么问题？如何正确写？**

模型不会用，或者调用时机不准确。

正确做法：说明适用场景、限制、参数格式要求、返回格式。

错误示例：`"读取文件"`

正确示例：`"读取本地文件系统中的文本文件内容。接受绝对路径（如 /home/user/doc.txt），返回文件的 UTF-8 文本内容。文件不存在时返回错误信息。不支持二进制文件。"`

---

**Q19. `call_tool` 没有统一错误处理会有什么后果？**

工具抛出的异常会作为 traceback 冒泡，模型收到乱码般的错误信息，无法正常处理。

正确做法：`call_tool` 需要 try/except，将异常转为 `TextContent` 返回给模型，而非让 traceback 冒泡。

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        # 工具逻辑
        result = do_something(arguments)
        return [TextContent(type="text", text=str(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"错误：{str(e)}")]
```

---

**Q20. 工具返回内容太大怎么处理？**

内容太大会撑爆 LLM 的 context window，导致后续推理失败或被截断。

解决方案：
- 工具端做截断并提示（如"结果已截断，共 X 条，显示前 20 条"）
- 提供分页功能（`page` 和 `page_size` 参数）
- 提供摘要版和详细版两个工具（模型先调摘要版，需要细节再调详细版）
