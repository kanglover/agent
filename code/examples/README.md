# Agent 开发示例代码库

面向前端转 Agent 开发工程师的完整示例集合。从"Hello Agent"到生产级多 Agent 系统，每个示例都可独立运行，配有详细注释。

---

## 📦 环境准备

### Python 环境

```bash
# 确认 Python 版本（需要 3.10+）
python3 --version

# 创建虚拟环境（隔离依赖，避免版本冲突）
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 安装核心依赖
pip install anthropic           # Claude SDK
pip install openai              # OpenAI SDK（部分示例用）
pip install langchain langchain-anthropic  # LangChain
pip install langgraph           # LangGraph（工作流）
pip install chromadb            # 向量数据库（RAG 示例）
pip install python-dotenv       # 读取 .env 配置文件
```

### Node.js 环境

```bash
# 确认 Node 版本（需要 18+）
node --version

# 进入 node 目录
cd node

# 安装依赖
npm install

# 主要依赖说明
# @anthropic-ai/sdk    → Claude 官方 SDK
# openai               → OpenAI SDK
# @langchain/core      → LangChain 核心
# @langchain/anthropic → LangChain 的 Claude 适配器
# dotenv               → 读取 .env 配置文件
```

### API Key 配置

在项目根目录创建 `.env` 文件（此文件不会被提交到 git）：

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> **获取 API Key：**
> - Anthropic（Claude）：https://console.anthropic.com/
> - OpenAI：https://platform.openai.com/api-keys

---

## 📁 目录结构

```
examples/
├── python/          ← Python 示例（25个）
│   ├── 01_hello_agent.py
│   ├── 02_tool_call_basic.py
│   └── ...
├── node/            ← Node.js 示例（20个）
│   ├── 01_hello_agent.js
│   ├── 02_tool_call_basic.js
│   └── ...
└── README.md        ← 本文件
```

---

## 🐍 Python 示例（25个）

| 文件 | 内容简介 | 难度 | 关键技术 |
|------|---------|------|---------|
| `01_hello_agent.py` | 第一个 Agent：向 Claude 发消息并打印回复 | ⭐ 入门 | `anthropic.Anthropic()`, `messages.create()` |
| `02_tool_call_basic.py` | 给 Agent 添加一个工具（加法计算器） | ⭐ 入门 | `tools` 参数, `tool_use` 解析 |
| `03_calculator_agent.py` | 四则运算计算器 Agent，支持连续对话 | ⭐ 入门 | 多工具定义, 工具结果回传 |
| `04_weather_agent.py` | 天气查询 Agent（模拟 API 调用） | ⭐⭐ 基础 | 外部 API 集成, 参数校验 |
| `05_file_reader_agent.py` | 让 Agent 读取本地文件并分析内容 | ⭐⭐ 基础 | 文件 IO, 大文本截断策略 |
| `06_search_agent.py` | 网络搜索 Agent（集成 Tavily / SerpAPI） | ⭐⭐ 基础 | 第三方搜索 API, 结果格式化 |
| `07_streaming_agent.py` | 流式输出：像 ChatGPT 一样逐字打印 | ⭐⭐ 基础 | `stream=True`, 事件流处理 |
| `08_multi_tool_agent.py` | 同时具备计算、搜索、文件读取三种能力 | ⭐⭐ 基础 | 工具路由, 能力组合 |
| `09_memory_agent.py` | 带记忆的 Agent：记住上下文，多轮对话 | ⭐⭐⭐ 进阶 | 对话历史管理, Token 窗口控制 |
| `10_react_agent.py` | ReAct 模式：思考→行动→观察→再思考 | ⭐⭐⭐ 进阶 | ReAct 框架, 推理链可视化 |
| `11_structured_output.py` | 让 Agent 输出 JSON / Pydantic 结构化数据 | ⭐⭐⭐ 进阶 | `tool_choice`, Pydantic 校验 |
| `12_error_handling_agent.py` | 工具调用失败时的重试与降级策略 | ⭐⭐⭐ 进阶 | 异常捕获, 指数退避重试 |
| `13_parallel_tools.py` | Claude 并行调用多个工具，提升效率 | ⭐⭐⭐ 进阶 | `asyncio`, 并行 Tool Use |
| `14_code_agent.py` | 代码生成+执行 Agent（在沙箱里跑代码） | ⭐⭐⭐ 进阶 | `exec()` 沙箱, 安全限制 |
| `15_todo_agent.py` | Todo 管理 Agent（增删查改本地任务列表） | ⭐⭐⭐ 进阶 | CRUD 工具设计, 状态持久化 |
| `16_rag_simple.py` | 最简 RAG：给 Agent 配上私有知识库 | ⭐⭐⭐⭐ 高级 | 文本分块, 向量检索, 上下文注入 |
| `17_vector_store_agent.py` | 基于 ChromaDB 的完整 RAG Agent | ⭐⭐⭐⭐ 高级 | ChromaDB, embedding, 相似度搜索 |
| `18_prompt_caching.py` | Prompt 缓存优化：省 Token，降延迟 | ⭐⭐⭐⭐ 高级 | `cache_control`, 缓存命中率监控 |
| `19_cost_tracking.py` | 实时追踪 Token 消耗与 API 费用 | ⭐⭐⭐⭐ 高级 | `usage` 字段, 成本计算公式 |
| `20_multi_step_agent.py` | 多步骤任务 Agent：分解目标→逐步执行 | ⭐⭐⭐⭐ 高级 | 任务规划, 步骤检查点 |
| `21_langchain_agent.py` | 用 LangChain 构建 Agent（标准化框架） | ⭐⭐⭐⭐ 高级 | `create_react_agent`, LangChain Tools |
| `22_langgraph_workflow.py` | 用 LangGraph 构建有状态工作流 | ⭐⭐⭐⭐ 高级 | StateGraph, 节点, 条件边 |
| `23_sub_agent.py` | Orchestrator + Sub-Agent 模式 | ⭐⭐⭐⭐⭐ 专家 | Agent 编排, 任务分发, 结果聚合 |
| `24_multi_agent_system.py` | 多 Agent 协作系统（分工、通信、投票） | ⭐⭐⭐⭐⭐ 专家 | Multi-Agent, 消息总线, 共识机制 |
| `25_production_agent.py` | 生产级 Agent：日志、监控、限流、降级 | ⭐⭐⭐⭐⭐ 专家 | 可观测性, 熔断器, 异步队列 |

---

## 🟨 Node.js 示例（20个）

| 文件 | 内容简介 | 难度 | 关键技术 |
|------|---------|------|---------|
| `01_hello_agent.js` | 第一个 Node Agent：调用 Claude API 打印回复 | ⭐ 入门 | `@anthropic-ai/sdk`, `async/await` |
| `02_tool_call_basic.js` | 给 Node Agent 添加工具（加法计算器） | ⭐ 入门 | `tools` 定义, `tool_use` 处理 |
| `03_calculator_agent.js` | 四则运算 Agent，支持多轮对话 | ⭐⭐ 基础 | 递归工具调用循环 |
| `04_streaming_agent.js` | 流式输出：`process.stdout.write` 逐字打印 | ⭐⭐ 基础 | `stream()`, SSE 事件处理 |
| `05_file_agent.js` | 文件读写 Agent（用 `fs` 模块操作文件） | ⭐⭐ 基础 | `fs/promises`, 路径安全校验 |
| `06_search_agent.js` | 搜索 Agent（集成 Tavily API） | ⭐⭐ 基础 | `fetch`, API Key 管理 |
| `07_multi_tool_agent.js` | 多工具 Agent（计算+搜索+文件） | ⭐⭐ 基础 | 工具分发函数, switch-case 路由 |
| `08_memory_agent.js` | 带记忆的 Agent（维护对话历史数组） | ⭐⭐⭐ 进阶 | 历史裁剪策略, 系统提示词设计 |
| `09_structured_output.js` | 结构化输出（返回 JSON Schema 格式数据） | ⭐⭐⭐ 进阶 | `tool_choice: {type:"tool"}`, Zod 校验 |
| `10_error_handling.js` | 错误处理与重试（网络超时/限流/参数错误） | ⭐⭐⭐ 进阶 | `try/catch`, 指数退避, 错误分类 |
| `11_parallel_tools.js` | 并行工具调用（`Promise.all` 加速执行） | ⭐⭐⭐ 进阶 | `Promise.all`, 并发控制 |
| `12_todo_agent.js` | Todo 管理 Agent（内存+文件双存储） | ⭐⭐⭐ 进阶 | CRUD 工具, JSON 持久化 |
| `13_rag_simple.js` | 极简 RAG：关键词检索+上下文注入 | ⭐⭐⭐⭐ 高级 | 文本分块, TF-IDF 相似度 |
| `14_express_agent.js` | 把 Agent 包装成 HTTP API（Express 服务） | ⭐⭐⭐⭐ 高级 | Express, 流式 SSE 接口, CORS |
| `15_prompt_caching.js` | Prompt 缓存：减少重复 Token 消耗 | ⭐⭐⭐⭐ 高级 | `cache_control`, 缓存统计 |
| `16_cost_tracking.js` | Token 计费追踪与日志记录 | ⭐⭐⭐⭐ 高级 | `usage` 对象, 费用累加器 |
| `17_langchain_node.js` | LangChain.js 构建 Agent | ⭐⭐⭐⭐ 高级 | `@langchain/anthropic`, AgentExecutor |
| `18_websocket_agent.js` | WebSocket 实时 Agent（双向流式通信） | ⭐⭐⭐⭐ 高级 | `ws` 库, 会话管理, 心跳检测 |
| `19_orchestrator_agent.js` | Orchestrator 模式：主 Agent 调度子任务 | ⭐⭐⭐⭐⭐ 专家 | 子任务分发, 结果聚合, 超时控制 |
| `20_production_agent.js` | 生产级 Node Agent（限流、监控、优雅退出） | ⭐⭐⭐⭐⭐ 专家 | Rate Limiter, Winston 日志, 健康检查 |

---

## 🎯 推荐学习顺序（3个月路线图）

### 第 1 个月：打基础（Week 1-4）

**目标：** 能独立运行所有入门和基础示例，理解 Agent 核心机制

```
Week 1：环境 + 第一个 Agent
  Day 1-2: 环境搭建，运行 python/01 和 node/01（Hello Agent）
  Day 3-4: 理解 Tool Use，运行 02、03（计算器 Agent）
  Day 5-7: 运行 04、05、06（天气/文件/搜索 Agent）

Week 2：工具组合 + 流式输出
  Day 1-3: 运行 07、08（多工具 + 流式输出）
  Day 4-5: 阅读并修改代码，添加一个自定义工具
  Day 6-7: 完成小作业：写一个"翻译 Agent"（调用翻译 API）

Week 3：记忆 + 多轮对话
  Day 1-3: 运行 09（Memory Agent），理解对话历史
  Day 4-5: 运行 10（ReAct Agent），理解推理链
  Day 6-7: 完成小作业：写一个能记住你名字的 Agent

Week 4：结构化输出 + 错误处理
  Day 1-3: 运行 11、12、13（结构化输出 + 错误处理 + 并行）
  Day 4-5: 运行 14、15（代码 Agent + Todo Agent）
  Day 6-7: 月度复盘 + 整理笔记
```

### 第 2 个月：进阶技术（Week 5-8）

**目标：** 掌握 RAG、LangChain、LangGraph，能构建有实用价值的 Agent

```
Week 5：RAG 知识库
  Day 1-3: 运行 16（极简 RAG），理解"检索增强生成"
  Day 4-5: 运行 17（ChromaDB），搭建真正的向量知识库
  Day 6-7: 小作业：给 Agent 上传一份 PDF，让它回答问题

Week 6：性能优化
  Day 1-2: 运行 18（Prompt 缓存），学会省 Token
  Day 3-4: 运行 19（成本追踪），监控 API 费用
  Day 5-7: 运行 20（多步骤 Agent），理解任务规划

Week 7：LangChain 生态
  Day 1-3: 运行 21（LangChain Agent），对比原生 SDK
  Day 4-7: 运行 22（LangGraph 工作流），理解有状态图

Week 8：Node.js 全套 + HTTP API
  Day 1-3: 运行 node/14（Express Agent API），用 Postman 测试
  Day 4-5: 运行 node/18（WebSocket Agent），实现实时对话
  Day 6-7: 月度复盘 + 构建一个完整的小项目
```

### 第 3 个月：专家进阶（Week 9-12）

**目标：** 掌握多 Agent 系统，能上线生产级 Agent

```
Week 9：多 Agent 架构
  Day 1-4: 运行 23（Sub-Agent），理解 Orchestrator 模式
  Day 5-7: 运行 24（Multi-Agent System），理解 Agent 协作

Week 10：生产级实践
  Day 1-3: 运行 25 和 node/20（生产级 Agent）
  Day 4-5: 学习日志、监控、限流、降级方案
  Day 6-7: 给自己的项目添加可观测性

Week 11：综合项目
  Day 1-7: 独立完成一个完整项目
  推荐选题：
    - 个人知识库问答机器人
    - 自动化代码审查 Agent
    - 多步骤研究报告生成器

Week 12：面试准备 + 总结
  Day 1-3: 刷面试题（见 notes/ 目录）
  Day 4-5: 梳理项目经验，写简历亮点
  Day 6-7: 模拟面试 + 复盘
```

---

## 🚀 快速开始（5分钟跑通第一个示例）

### Python 版（3步）

```bash
# 第1步：进入 python 目录，安装依赖
cd python
pip install anthropic python-dotenv

# 第2步：配置 API Key
echo 'ANTHROPIC_API_KEY=你的key' > .env

# 第3步：运行第一个示例
python 01_hello_agent.py
```

预期输出：
```
Agent 回复: 你好！我是 Claude，一个 AI 助手。很高兴认识你！有什么我可以帮你的吗？
```

### Node.js 版（3步）

```bash
# 第1步：进入 node 目录，安装依赖
cd node
npm install

# 第2步：配置 API Key（复用上面的 .env 或新建）
echo 'ANTHROPIC_API_KEY=你的key' > .env

# 第3步：运行第一个示例
node 01_hello_agent.js
```

预期输出：
```
Agent 回复: 你好！我是 Claude，很高兴与你交流！
```

> 遇到问题？常见解决方案：
> - `ModuleNotFoundError` → 执行 `pip install anthropic`
> - `AuthenticationError` → 检查 `.env` 文件中的 API Key 是否正确
> - `RateLimitError` → 等待 1 分钟后重试，或检查账户余额

---

## 📌 代码阅读建议

每个示例文件顶部都有这样的注释块，请先读它再看代码：

```python
"""
示例名称: 01_hello_agent.py
学习目标: 发送第一条消息给 Claude，理解最基础的 API 调用
前置知识: Python 基础语法，知道什么是函数和变量
关键概念: messages 数组结构, role 字段（user/assistant）, content 字段
运行时间: < 5秒
API 费用: 约 $0.0001（极低）
"""
```
