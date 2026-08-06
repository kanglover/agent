# Agent 面试题 04 — MCP 协议与工具调用

> 来源：训练知识（截止 2026 年 1 月）  
> 整理日期：2026-07-04

---

## 一、MCP 基础与架构

### Q1. 什么是 MCP（Model Context Protocol）？它解决了什么问题？

**答题要点：**

- MCP 是 Anthropic 于 2024 年 11 月开源的**标准化上下文协议**，全称 Model Context Protocol。
- **核心目标**：让 LLM 以统一、标准化的方式连接外部数据源和工具，而不需要为每个工具写一套专属集成代码。
- **类比**：MCP 相当于 AI 世界的"USB 接口标准"——以前每个设备（工具）都需要定制驱动，MCP 提供了一个通用插槽，任何符合规范的工具插上就能用。
- **解决的核心痛点**：
  1. M×N 集成爆炸问题：M 个模型 × N 个工具 = MN 套代码；MCP 将其降为 M+N。
  2. 工具能力碎片化：不同平台对 tool use 的实现不统一。
  3. 安全与权限边界模糊：缺乏统一的授权机制。

---

### Q2. 描述 MCP 的整体架构，有哪些核心角色？

**答题要点：**

MCP 采用 **Host / Client / Server 三层架构**：

| 角色 | 职责 | 举例 |
|------|------|------|
| **Host（宿主）** | 运行 LLM 的应用程序，负责编排整体交互流程 | Claude Desktop、IDE 插件、自定义 Agent |
| **Client（客户端）** | 嵌入在 Host 内部，维护与单个 MCP Server 的 1:1 连接 | Host 内部模块 |
| **Server（服务端）** | 暴露工具、资源、Prompt 能力的轻量级服务 | 文件系统 Server、GitHub Server、数据库 Server |

**关键设计原则：**
- 一个 Host 可同时连接**多个** MCP Server（通过多个 Client 实例）。
- Client 与 Server 之间是**长连接**，支持双向通信。
- Server 是无状态或轻状态的，业务逻辑集中在 Host 侧。

---

## 二、MCP vs Function Calling

### Q3. MCP 与 OpenAI Function Calling / Anthropic Tool Use 有何本质区别？

**答题要点：**

| 维度 | Function Calling / Tool Use | MCP |
|------|----------------------------|-----|
| **作用范围** | 单次请求内的工具定义 | 跨请求、跨模型的标准化协议 |
| **工具发现** | 每次请求手动传入 tools 列表 | Server 动态暴露，Client 自动发现（`tools/list`） |
| **运行位置** | 工具代码通常由调用方直接执行 | 工具运行在独立的 MCP Server 进程中 |
| **复用性** | 工具定义绑定到特定模型/SDK | MCP Server 可被任意支持 MCP 的模型复用 |
| **连接模型** | 无持久连接，每次请求独立 | 持久连接，支持订阅/通知 |
| **标准化程度** | 各厂商格式略有差异 | 统一 JSON-RPC 2.0 消息格式 |

**一句话总结**：Function Calling 是"单次对话内的工具调用机制"，MCP 是"跨系统、跨模型的工具集成标准"。

---

### Q4. 既然已有 Function Calling，为什么还需要 MCP？

**答题要点：**

1. **跨模型复用**：一个 MCP Server 写好后，Claude、GPT-4、Gemini 等任何支持 MCP 的模型都能直接使用，无需重写工具定义。
2. **动态能力发现**：MCP 支持运行时查询 Server 当前暴露了哪些工具（`tools/list`），而不是在 prompt 中静态硬编码。
3. **资源与 Prompt 模板**：MCP 不仅支持工具（Tools），还支持数据资源（Resources）和 Prompt 模板，Function Calling 只覆盖工具调用。
4. **进程隔离与安全**：MCP Server 运行在独立进程，天然隔离敏感逻辑。
5. **生态标准化**：鼓励第三方构建可共享的 MCP Server 生态（类似 npm 包）。

---

## 三、MCP 传输机制

### Q5. MCP 支持哪些传输协议？各自适用场景是什么？

**答题要点：**

MCP 定义了两种标准传输层：

**1. stdio（标准输入/输出）**
- Server 以**子进程**形式启动，Host 通过 stdin/stdout 与其通信。
- 适用场景：本地工具（文件系统、本地数据库、CLI 工具）。
- 优点：简单、无网络开销、天然进程隔离。
- 缺点：不适合远程部署，无法跨机器。

**2. HTTP + SSE（Server-Sent Events）**
- Client 通过 HTTP POST 发送请求，Server 通过 SSE 推送响应/通知。
- 适用场景：远程 MCP Server、云服务集成、多用户共享服务。
- 优点：支持远程部署，可扩展为 SaaS 形式。
- 缺点：需要处理认证、网络延迟、连接稳定性。

**底层消息格式**：两种传输均使用 **JSON-RPC 2.0** 作为消息格式，确保上层协议一致。

---

## 四、MCP 三大原语（Primitives）

### Q6. MCP 定义的三种核心原语是什么？请分别解释。

**答题要点：**

| 原语 | 英文 | 控制方 | 说明 |
|------|------|--------|------|
| **工具** | Tools | Model 控制 | LLM 可主动调用的函数，用于执行操作（写文件、调 API、查数据库） |
| **资源** | Resources | Application 控制 | 暴露给 LLM 的上下文数据，类似只读文件（日志、文档、数据库记录） |
| **提示模板** | Prompts | User 控制 | Server 提供的预定义 Prompt 模板，用户可选择触发 |

**深入区分：**
- **Tools** 是"动词"，有副作用，LLM 决定何时调用。
- **Resources** 是"名词"，是数据，应用层决定何时注入到上下文。
- **Prompts** 是"工作流模板"，由用户显式触发，封装特定任务的最佳 Prompt 写法。

---

### Q7. MCP 中 Resources 和 Tools 的边界如何划定？举例说明。

**答题要点：**

- **Resources** = 只读、无副作用的数据暴露。
  - 例：`file://project/README.md`（读取文件内容）、`db://users/recent`（查询结果）。
  - 使用方式：通过 `resources/read` 请求获取，内容注入 LLM 上下文。
- **Tools** = 有执行语义、可能有副作用的操作。
  - 例：`write_file(path, content)`、`send_email(to, subject, body)`。
  - 使用方式：LLM 生成 tool_call，Host 执行后将结果返回给 LLM。

**判断标准**：如果是"查数据"用 Resource，如果是"做操作"用 Tool。模糊情况（如搜索）可以是 Tool，因为搜索有查询副作用（计费、日志）。

---

## 五、安全考量

### Q8. MCP 在安全设计上有哪些需要重点关注的问题？

**答题要点：**

1. **Tool Poisoning（工具投毒）**
   - 攻击方式：恶意 MCP Server 在工具描述中注入隐藏指令，诱导 LLM 执行预期外的操作。
   - 防御：对第三方 MCP Server 的工具描述进行人工审查；仅使用可信来源的 Server。

2. **权限最小化原则**
   - MCP Server 只应暴露完成任务所需的最小工具集。
   - 避免将高权限操作（如删除文件、发送邮件）不加保护地暴露。

3. **用户确认机制**
   - 对于高风险 Tool（写操作、删除、外部 API 调用），Host 应在执行前向用户确认。
   - Claude Desktop 等实现中有"Allow/Deny"弹窗机制。

4. **Server 身份验证**
   - 远程 MCP Server（HTTP+SSE）必须实现认证（OAuth、API Key）。
   - 防止未授权方注册恶意 Server。

5. **提示注入（Prompt Injection）防御**
   - Resources 内容可能包含恶意指令，LLM 处理时需警惕。
   - 将 Resource 内容与系统指令在上下文中明确隔离。

---

## 六、多模型/多工具编排

### Q9. MCP 如何支持多模型、多工具的 Agent 编排场景？

**答题要点：**

1. **统一工具接口**：Agent 框架（Host）可以同时连接多个 MCP Server，每个 Server 暴露不同领域的工具，LLM 通过统一的 `tools/list` 发现所有可用能力。

2. **跨模型工具共享**：同一批 MCP Server 可以被不同的 LLM 模型调用（Claude 调文件系统 Server，同一个 Server 也可被 GPT-4 调用），无需重复开发。

3. **Agentic Loop 中的角色**：
   ```
   Host (Agent Loop)
     ├── LLM (Claude/GPT/etc.)      ← 决策层
     ├── MCP Client → Server A      ← 文件系统工具
     ├── MCP Client → Server B      ← 数据库工具
     └── MCP Client → Server C      ← 外部 API 工具
   ```

4. **Sampling 能力（高级）**：MCP Server 可以向 Host 发起 LLM 采样请求（`sampling/createMessage`），即 Server 也能触发 LLM 调用，支持更复杂的 Agent-to-Agent 通信模式。

---

### Q10. 什么是 MCP 的 Sampling 机制？它有什么意义？

**答题要点：**

- **Sampling** 是 MCP 中允许 Server 向 Host 请求执行 LLM 推理的机制。
- 请求方法：`sampling/createMessage`，Server 发起，Host 执行 LLM 调用后返回结果。
- **意义**：
  1. 使 MCP Server 本身可以包含 AI 逻辑，而不仅仅是工具执行层。
  2. 支持 Sub-Agent 模式：主 Agent 的工具调用可以触发另一个 LLM 完成子任务。
  3. 实现"工具中有 AI，AI 中有工具"的嵌套结构。
- **限制**：最终是否执行 LLM 调用的控制权在 Host，保证用户对 AI 调用有感知和控制。

---

## 七、综合对比与实战

### Q11. 在设计一个新的 Agent 系统时，你会如何决定使用 Function Calling 还是 MCP？

**答题要点：**

**选 Function Calling / Tool Use 的场景：**
- 工具数量少（< 10 个），且工具只服务于单一模型。
- 需要在单次对话中动态构建工具定义（根据用户输入变化）。
- 快速原型，不需要跨团队、跨系统复用。

**选 MCP 的场景：**
- 工具需要被多个不同模型或多个应用复用。
- 工具逻辑复杂，需要在独立进程中运行（性能隔离、语言独立）。
- 企业内部需要建立统一的工具生态（内部 MCP Server Registry）。
- 需要暴露 Resources（大量上下文数据）或 Prompt 模板。
- 安全要求高，需要明确的进程边界和权限控制。

**混合使用**：实际生产中常见"MCP Server 内部再调用 LLM tool use"的组合模式。

---

### Q12. 请描述 MCP 一次完整的工具调用流程（从 LLM 决策到结果返回）。

**答题要点：**

```
1. 初始化阶段
   Host 启动 → Client 连接 MCP Server → 发送 initialize 握手
   Server 返回能力声明（capabilities）

2. 能力发现阶段
   Client 发送 tools/list 请求
   Server 返回工具列表（name, description, inputSchema）
   Host 将工具列表注入 LLM 的系统上下文

3. LLM 决策阶段
   用户发送消息 → LLM 分析任务
   LLM 输出 tool_call（工具名 + 参数 JSON）

4. 工具执行阶段
   Host 接收 tool_call → Client 发送 tools/call 请求到 Server
   Server 执行工具逻辑 → 返回 result（content 数组）

5. 结果回注阶段
   Host 将 tool result 作为新消息追加到对话历史
   LLM 基于结果继续推理，生成最终回复

6. （可选）多轮工具调用
   LLM 可继续调用其他工具，形成 Agentic Loop
   直到 LLM 判断任务完成，输出最终答案
```

---

## 附：关键术语速查

| 术语 | 含义 |
|------|------|
| MCP Host | 运行 LLM 的宿主应用（如 Claude Desktop） |
| MCP Client | Host 内部连接单个 Server 的模块 |
| MCP Server | 暴露工具/资源/Prompt 的轻量服务 |
| stdio transport | 通过子进程 stdin/stdout 通信 |
| HTTP+SSE transport | 通过 HTTP POST + Server-Sent Events 通信 |
| Tools | LLM 可调用的函数（有副作用） |
| Resources | 只读数据，注入 LLM 上下文 |
| Prompts | 预定义 Prompt 模板，用户触发 |
| Sampling | Server 向 Host 请求 LLM 推理的机制 |
| JSON-RPC 2.0 | MCP 底层消息格式标准 |

---

> 注：以上内容基于训练知识（截止 2026 年 1 月），MCP 规范仍在快速演进，建议结合 [modelcontextprotocol.io](https://modelcontextprotocol.io) 官方文档核实最新版本细节。
