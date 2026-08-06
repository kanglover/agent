# 学习资料总索引

所有资料的导航，按学习顺序排列。本文件是整个知识库的"地图"，迷路时先来这里找方向。

> 最后更新：2026-07-04

---

## 📚 文档索引（按学习顺序）

按"从零到生产"的顺序排列，建议按序阅读。

### 阶段一：入门认知（先读这些）

| 文件 | 核心内容 | 预计阅读时间 |
|------|---------|------------|
| [从使用AI到建造AI工具.md](./从使用AI到建造AI工具.md) | 用户视角 → 开发者视角的思维转变，为什么要学 Agent | 15 分钟 |
| [agent-入门概念.md](./agent-入门概念.md) | Agent 是什么、和普通 AI 对话有什么区别、核心组件 | 20 分钟 |
| [agents-md-AI工作手册.md](./agents-md-AI工作手册.md) | AI 工具全景地图，各类工具的定位与选型建议 | 25 分钟 |

### 阶段二：核心技术（重点掌握）

| 文件 | 核心内容 | 预计阅读时间 |
|------|---------|------------|
| [tool-use-计算器示例.md](./tool-use-计算器示例.md) | Tool Use 机制详解，用计算器示例手把手讲解 | 30 分钟 |
| [writing-effective-tools-for-agents.md](./writing-effective-tools-for-agents.md) | 如何设计高质量的工具定义（官方最佳实践） | 40 分钟 |
| [agent-workflow-核心概念.md](./agent-workflow-核心概念.md) | Workflow vs Agent 的本质区别，何时用哪种 | 30 分钟 |
| [workflow-vs-agent-代码对比.md](./workflow-vs-agent-代码对比.md) | 同一任务用 Workflow 和 Agent 两种方式实现，代码对比 | 45 分钟 |
| [context-engineering-上下文工程.md](./context-engineering-上下文工程.md) | 上下文工程：如何精准控制 AI 看到的信息 | 35 分钟 |

### 阶段三：架构深读（进阶必读）

| 文件 | 核心内容 | 预计阅读时间 |
|------|---------|------------|
| [agent-架构解析-ClaudeCode与Hermes.md](./agent-架构解析-ClaudeCode与Hermes.md) | ClaudeCode 和 Hermes 的内部架构剖析 | 50 分钟 |
| [agent-架构解析-LangChain与LangGraph.md](./agent-架构解析-LangChain与LangGraph.md) | LangChain 和 LangGraph 框架设计原理 | 50 分钟 |
| [openspec-规格驱动开发.md](./openspec-规格驱动开发.md) | OpenSpec 方法论：用规格文件驱动 Agent 开发 | 40 分钟 |

### 阶段四：工程实践

| 文件 | 核心内容 | 预计阅读时间 |
|------|---------|------------|
| [agent-openai实践指南.md](./agent-openai实践指南.md) | 使用 OpenAI API 构建 Agent 的完整实战指南 | 60 分钟 |
| [agent-踩坑指南-100条.md](./agent-踩坑指南-100条.md) | 100 个真实踩坑记录，每条附解决方案 | 随用随查 |
| [bilibili-agent-视频资源汇总.md](./bilibili-agent-视频资源汇总.md) | B站精选 Agent 相关视频，附学习要点摘要 | 参考导航 |

### 学习计划文档

| 文件 | 核心内容 | 用途 |
|------|---------|------|
| [agent-工程师学习路径-前端转型.md](./agent-工程师学习路径-前端转型.md) | 专为前端工程师设计的转型路径，利用已有技能 | 制定个人计划 |
| [agent-完整学习计划-超详细版.md](./agent-完整学习计划-超详细版.md) | 最详细的分阶段学习计划（每天具体安排） | 日常执行参考 |
| [todo.md](./todo.md) | 当前待办事项和学习进度记录 | 每天更新 |

---

## 🐍 Python 示例索引

示例文件位置：`/code/examples/python/`

### 入门组（先跑通这5个）

| 文件 | 一句话说明 | 需要先掌握 |
|------|----------|----------|
| `01_hello_agent.py` | 最简单的 API 调用，打印 Claude 的回复 | Python 基础 |
| `02_tool_call_basic.py` | 添加第一个工具（加法器），理解工具调用流程 | 01 |
| `03_calculator_agent.py` | 完整计算器 Agent，支持四则运算 | 02 |
| `07_streaming_agent.py` | 流式输出，像 ChatGPT 一样逐字显示 | 01 |
| `09_memory_agent.py` | 多轮对话，Agent 记住上下文 | 03 |

### 工具能力组

| 文件 | 一句话说明 | 应用场景 |
|------|----------|---------|
| `04_weather_agent.py` | 对接外部 API 获取天气信息 | 第三方 API 集成 |
| `05_file_reader_agent.py` | 读取并分析本地文件 | 文档处理 |
| `06_search_agent.py` | 联网搜索最新信息 | 实时数据获取 |
| `08_multi_tool_agent.py` | 同时拥有多种工具能力 | 通用助手 |
| `14_code_agent.py` | 生成并执行 Python 代码 | 自动化脚本 |
| `15_todo_agent.py` | 增删查改本地任务列表 | 数据持久化 |

### 进阶技术组

| 文件 | 一句话说明 | 核心概念 |
|------|----------|---------|
| `10_react_agent.py` | 思考→行动→观察 的推理循环 | ReAct 框架 |
| `11_structured_output.py` | 让 Agent 输出标准 JSON 格式 | 结构化输出 |
| `12_error_handling_agent.py` | 工具失败时的自动重试 | 容错设计 |
| `13_parallel_tools.py` | 同时调用多个工具节省时间 | 并发编程 |
| `16_rag_simple.py` | 给 Agent 加上私有知识库 | RAG 基础 |
| `17_vector_store_agent.py` | 基于向量数据库的完整 RAG | 向量检索 |

### 性能优化组

| 文件 | 一句话说明 | 优化目标 |
|------|----------|---------|
| `18_prompt_caching.py` | 重复内容缓存，减少 Token 消耗 | 降低成本 |
| `19_cost_tracking.py` | 实时统计 API 费用 | 成本控制 |
| `20_multi_step_agent.py` | 大任务分解为小步骤执行 | 复杂任务 |

### 框架与系统组

| 文件 | 一句话说明 | 生产价值 |
|------|----------|---------|
| `21_langchain_agent.py` | LangChain 标准化 Agent 构建 | 团队协作 |
| `22_langgraph_workflow.py` | 有状态工作流，支持条件分支 | 复杂业务流 |
| `23_sub_agent.py` | 主 Agent 指挥多个子 Agent | 任务分工 |
| `24_multi_agent_system.py` | 多 Agent 协作与通信 | 大规模系统 |
| `25_production_agent.py` | 带监控、限流、日志的生产级 Agent | 上线部署 |

---

## 🟨 Node.js 示例索引

示例文件位置：`/code/examples/node/`

### 入门组

| 文件 | 一句话说明 | 对应 Python 版本 |
|------|----------|----------------|
| `01_hello_agent.js` | 第一个 Node Agent | `01_hello_agent.py` |
| `02_tool_call_basic.js` | 基础工具调用 | `02_tool_call_basic.py` |
| `03_calculator_agent.js` | 计算器 Agent | `03_calculator_agent.py` |
| `04_streaming_agent.js` | 流式输出 | `07_streaming_agent.py` |

### 工具与服务组

| 文件 | 一句话说明 | Node 特有优势 |
|------|----------|-------------|
| `05_file_agent.js` | 文件读写操作 | 异步 IO 性能好 |
| `06_search_agent.js` | 联网搜索 | fetch API 原生支持 |
| `07_multi_tool_agent.js` | 多工具 Agent | - |
| `12_todo_agent.js` | Todo 管理 Agent | - |
| `14_express_agent.js` | 把 Agent 包装成 HTTP API | 快速构建 Web 服务 |
| `18_websocket_agent.js` | WebSocket 实时双向对话 | 实时通信场景 |

### 进阶技术组

| 文件 | 一句话说明 | 关键知识点 |
|------|----------|----------|
| `08_memory_agent.js` | 多轮对话记忆 | 历史管理 |
| `09_structured_output.js` | JSON 结构化输出 + Zod 校验 | 类型安全 |
| `10_error_handling.js` | 完整错误处理方案 | 生产稳定性 |
| `11_parallel_tools.js` | `Promise.all` 并行工具调用 | 性能优化 |
| `13_rag_simple.js` | 极简 RAG 实现 | 知识库集成 |

### 生产与框架组

| 文件 | 一句话说明 | 应用场景 |
|------|----------|---------|
| `15_prompt_caching.js` | Prompt 缓存优化 | 高频调用场景 |
| `16_cost_tracking.js` | 费用追踪与日志 | 成本管控 |
| `17_langchain_node.js` | LangChain.js 构建 Agent | 前端团队友好 |
| `19_orchestrator_agent.js` | Orchestrator 编排模式 | 复杂任务分发 |
| `20_production_agent.js` | 生产级 Node Agent | 正式上线 |

---

## 🎯 面试题索引（1000+ 题）

面试题文件位置：`/notes/`，共 5 个专题文件。

| 文件 | 题目范围 | 题目数量 | 难度分布 |
|------|---------|---------|---------|
| [agent-面试题-01-LLM基础与Prompt工程.md](./agent-面试题-01-LLM基础与Prompt工程.md) | Transformer 原理、Attention 机制、Prompt 写法、Few-shot、Chain-of-Thought | ~200题 | 初中高级均有 |
| [agent-面试题-02-ToolUse与Agent架构.md](./agent-面试题-02-ToolUse与Agent架构.md) | Tool Use 机制、ReAct 框架、Agent 设计模式、Orchestrator 模式、错误处理 | ~200题 | 中高级为主 |
| [agent-面试题-03-RAG与向量数据库.md](./agent-面试题-03-RAG与向量数据库.md) | RAG 原理、Embedding、向量检索、ChromaDB/Pinecone、分块策略、混合检索 | ~200题 | 中高级为主 |
| [agent-面试题-04-大厂真题汇总.md](./agent-面试题-04-大厂真题汇总.md) | 阿里、字节、百度、腾讯、美团等大厂真实面试题（含答案） | ~250题 | 中高级 |
| [agent-面试题-05-工程化与生产部署.md](./agent-面试题-05-工程化与生产部署.md) | 监控、限流、缓存、成本优化、A/B 测试、安全防护、灰度发布 | ~200题 | 高级为主 |

### 面试备考策略

```
第 1 周：刷 01（LLM基础），建立理论底盘
第 2 周：刷 02（Tool Use + Agent），重点是画架构图
第 3 周：刷 03（RAG），动手搭一个向量知识库
第 4 周：刷 04（大厂真题），发现薄弱点补漏
第 5 周：刷 05（工程化），结合 25_production_agent.py 理解
第 6 周：模拟面试 + 复盘
```

---

## 🗓️ 推荐学习路径

### 3个月速成版（适合：全职学习 / 跳槽倒计时）

**节奏：** 每天 3-4 小时，周末集中攻关

```
Month 1：建立基础
  Week 1: 阶段一文档 + Python 01~08
  Week 2: 阶段二文档 + Python 09~15
  Week 3: 阶段三文档 + Python 16~20
  Week 4: Node 01~10 + 整合练习

Month 2：进阶技术
  Week 5: Python 21~22 (LangChain + LangGraph)
  Week 6: Node 11~17 + RAG 实战项目
  Week 7: Python 23~25 (多Agent + 生产级)
  Week 8: Node 18~20 + 完整项目开发

Month 3：冲刺准备
  Week 9: 面试题 01+02（重刷代码示例）
  Week 10: 面试题 03+04（RAG 项目收尾）
  Week 11: 面试题 05 + 简历优化
  Week 12: 模拟面试 + 投简历
```

**3个月里程碑：**
- 第30天：能独立实现带 Tool Use 的 Agent
- 第60天：能构建 RAG 知识库 + LangGraph 工作流
- 第90天：能在面试中讲清多 Agent 架构，有项目可展示

---

### 6个月稳健版（适合：在职学习 / 稳扎稳打）

**节奏：** 工作日每天 1-2 小时，周末 4-6 小时

```
Month 1-2：打基础
  文档：阶段一 + 阶段二全部
  代码：Python 01~15（跑通 + 能改）
  作业：写一个"个人助手 Agent"

Month 3-4：进阶技术
  文档：阶段三 + 阶段四全部
  代码：Python 16~25 + Node 全部
  作业：搭一个 RAG 知识库问答系统

Month 5：工程化实战
  深入踩坑指南 100 条
  完成一个有实用价值的项目（可上线）
  学习监控、日志、限流最佳实践

Month 6：面试冲刺
  系统刷完所有面试题
  整理项目亮点，写进简历
  参加 3-5 场面试积累经验
```

**6个月里程碑：**
- 第60天：独立完成带记忆+工具的 Agent
- 第120天：完成 RAG + 多 Agent 项目，部署上线
- 第180天：拿到目标 offer

---

### 9个月深度版（适合：零基础 / 想做得扎实）

**节奏：** 每天 1 小时，周末 2-3 小时，节奏轻松不焦虑

```
Month 1-3：夯实基础（同3个月Month 1，放慢节奏）
  重点：每个示例不只是跑通，要读懂每一行代码
  每周作业：用学到的知识解决一个实际问题

Month 4-6：技术进阶（同3个月Month 2，更深入）
  重点：理解原理，不只会用框架
  每月项目：从零搭建一个完整 Agent 应用

Month 7-8：工程化与系统设计
  深入学习：监控、可观测性、成本优化、安全
  实战：把自己的项目优化到生产级别
  学习：系统设计面试方法（Agent 系统如何设计）

Month 9：综合收尾
  完成一个有独立思考的项目（不是照抄示例）
  写技术博客沉淀知识（输出是最好的学习）
  面试准备 + 求职
```

**9个月里程碑：**
- 第90天：能流畅使用所有入门到进阶示例
- 第180天：独立搭建完整 Agent 系统
- 第270天：能做技术分享，有深度项目经验

---

## 🔍 快速查找

### 按技术点查找

| 想学这个 | 去看这里 |
|---------|---------|
| Tool Use 基础 | `tool-use-计算器示例.md` + `python/02` |
| 如何设计好工具 | `writing-effective-tools-for-agents.md` |
| RAG 知识库 | `python/16` + `python/17` |
| 流式输出 | `python/07` + `node/04` |
| 多 Agent 系统 | `python/23` + `python/24` |
| 生产级部署 | `python/25` + `node/20` + `agent-踩坑指南-100条.md` |
| LangChain | `python/21` + `agent-架构解析-LangChain与LangGraph.md` |
| LangGraph | `python/22` + `agent-架构解析-LangChain与LangGraph.md` |
| 成本优化 | `python/18` + `python/19` |
| 面试备考 | 见面试题索引章节 |

### 按场景查找

| 应用场景 | 推荐示例 |
|---------|---------|
| 个人知识库问答 | `python/16` + `python/17` |
| 自动化报告生成 | `python/20` + `python/22` |
| 代码审查 Agent | `python/14` + `python/10` |
| 客服机器人 | `python/09` + `node/14` |
| 数据分析 Agent | `python/11` + `python/13` |
| 实时聊天应用 | `node/18` + `node/04` |
| 内部工具集成 | `python/08` + `python/25` |
