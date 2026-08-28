# AI Agent 学习仓库

> 从零开始学习 AI Agent 开发的过程记录——包含阅读笔记、代码实验和动手项目。

围绕一个核心问题展开：**如何让 AI 自主使用工具、完成多步骤任务？**

从手写最小 Agent Loop，到拆解 Claude Code / Codex 的内部实现，再到用 LangChain / LangGraph 搭建完整工作流——每个阶段的理解和代码都沉淀在这里。

---

## 完整目录结构

```
agent/
├── code/                        ← 所有代码
│   ├── min_agent/               ← 最小 Agent Loop（核心项目）
│   ├── context_builder/         ← 上下文工程：五层结构 + 自动压缩
│   ├── agent_harness/           ← Agent 测评框架（出题 → 跑 → 评分）
│   ├── learn_langchain/         ← LangChain 零基础 9 篇教程
│   ├── learn_langgraph/         ← LangGraph 零基础 6 篇教程
│   ├── menu_rag_project/        ← RAG 实战项目（菜谱知识库问答）
│   ├── examples/                ← 综合示例库（Python 25 + Node 20）
│   ├── min_claude_code.py       ← Claude Code 最小实现（Edit + Hook）
│   ├── min_codex.py             ← Codex CLI 最小实现（apply_patch + Approval）
│   ├── min_hermes.py            ← Hermes Agent 最小实现（记忆 + Skills）
│   ├── multi_agent_demo.py      ← 多 Agent 协作（纯模拟，无需 API Key）
│   ├── calculator_agent.py      ← 入门：计算器工具调用
│   ├── todo_agent.py            ← 实用：读文件分析待办清单
│   ├── demo_qwen.py             ← 通义千问 + Gradio 对话界面
│   ├── qwen_embedding_demo.py   ← 通义千问 Embedding 示例
│   └── tests/                   ← 单元测试
│
├── notes/                       ← 阅读笔记与知识沉淀（38 篇）
└── docs/                        ← 规格与计划文档
```

---

## 核心代码模块

### 从零手写 Agent

| 文件 / 目录 | 说明 |
|------------|------|
| `min_agent/` | 最小 Agent Loop：observe → think → act 循环，不依赖任何框架，每步写 JSONL trace + 费用估算 |
| `min_claude_code.py` | Claude Code 最小实现：Edit 工具（old_str → new_str 精准改）+ Hook 中间件（PreHook / PostHook 拦截） |
| `min_codex.py` | Codex CLI 最小实现：apply_patch 差量修改 + suggest / auto 编辑审批模式 |
| `min_hermes.py` | Hermes Agent 最小实现：跨会话持久记忆（SQLite + MEMORY.md）+ 自动 Skill 学习 + 多平台网关 |
| `multi_agent_demo.py` | 多 Agent 协作系统：Coordinator 拆任务 → Researcher / Writer / Reviewer 执行 → 路由汇总，纯模拟无需 Key |

`min_agent/` 的核心循环：

```
用户任务
  ↓
[THINK] LLM 决策：调用工具 or 直接回答？
  ↓ tool_use
[ACT]   执行工具（search_notes / write_summary）
  ↓
[OBSERVE] 结果喂回 LLM，继续循环
  ↓ end_turn
任务完成，写入 trace.jsonl
```

关键设计：最大 5 步防无限循环、每步写 JSONL trace（含 token 费用估算）、出错不崩溃返回结构化失败原因。

### 上下文工程（`context_builder/`）

把发给 AI 的上下文拆成五层，结构化管理，长内容自动截断并生成可追溯引用：

| 层 | 名称 | 内容 |
|----|------|------|
| 1 | System | 角色定义、规则、工具列表 |
| 2 | Task | 当前任务描述与目标 |
| 3 | Memory | 跨任务持久化知识 |
| 4 | Evidence | 检索到的外部内容（自动压缩） |
| 5 | Trace | 最近的工具调用记录（自动压缩） |

```python
bundle = (
    ContextBuilder()
    .set_system(role="助手", instructions=["简洁中文"])
    .set_task(description="总结笔记", goal="输出摘要")
    .add_evidence(source="note.md", content="...很长的内容...")  # 自动压缩
    .build()
)
print(render(bundle))
```

### Agent 测评框架（`agent_harness/`）

把 Agent 当成"考生"，出同样的题、用统一标准打分排名。

- 协议层（`protocol.py`）定义 Agent 接口，任何 Agent 实现 `run()` 即可参赛
- 评估器从完成度（40%）、关键词（30%）、工具使用（20%）、效率（10%）四个维度打分
- 自带模拟 Agent，不需要 API Key 也能跑通

### 框架学习教程

| 目录 | 内容 | 篇数 |
|------|------|:---:|
| `learn_langchain/` | LangChain 零基础：调用模型 → 提示词模板 → LCEL 链 → 结构化输出 → 工具 → 记忆 → RAG → 进阶编排 → 中间件 | 9 |
| `learn_langgraph/` | LangGraph 零基础：基础图 → 条件分支循环 → 工具 Agent → 记忆持久化 → 人机协作 → 多节点工作流 | 6 |

两套教程都按顺序学，每篇开头有【学什么】【为什么】【类比】注释，循序渐进。

### RAG 实战项目（`menu_rag_project/`）

一个完整的菜谱知识库问答系统，覆盖 RAG 全链路：

数据准备 → 索引构建 → 检索优化 → 生成集成

内置 11 类菜谱数据（荤菜、素菜、汤、主食、甜品、饮品等），用向量检索 + 混合检索回答做菜问题。

### 综合示例库（`examples/`）

面向前端转 Agent 开发的完整示例集合，从 Hello Agent 到生产级系统：

- **Python（25 个）**：`01_hello_claude` → `25_prompt_caching`，覆盖工具调用、记忆、ReAct、RAG、结构化输出、LangChain / LangGraph / AutoGen / CrewAI、MCP、多模态、批量处理等
- **Node.js（20 个）**：`01_hello_claude` → `20_production_patterns`，覆盖流式输出、Express API、WebSocket、Vercel AI SDK、OpenAI Agents SDK 等

每个示例可独立运行，顶部有统一注释块（学习目标、前置知识、关键概念、运行时间、API 费用）。详见 [examples/README.md](code/examples/README.md)。

---

## 学习笔记（`notes/`）

### 入门与概念

- `agent-入门概念.md` — AI Agent 开发入门
- `agent-workflow-核心概念.md` — Workflow vs Agent 的本质区别
- `workflow-vs-agent-代码对比.md` — 用代码说明两者核心差异
- `从使用AI到建造AI工具.md` — 从"用 AI"到"造 AI 工具"的思维转变
- `tool-use-计算器示例.md` — Tool Use 实战入门
- `agents-md-AI工作手册.md` — AGENTS.md 的作用与写法

### 核心原理

- `context-engineering-上下文工程.md` — 上下文工程业界最佳实践
- `cot-思维链-完全指南.md` — CoT 思维链完全指南
- `agent-RAG-原理与实现.md` — RAG 检索增强生成原理
- `agent-ReAct-核心模板.md` — ReAct 模式核心组成与模板
- `agent-记忆机制-短期与长期.md` — Agent 记忆机制
- `agent-意图识别与槽位填充.md` — 意图识别与槽位填充
- `multi-agent-协作系统.md` — 多 Agent 协作系统
- `agent-openai实践指南.md` — OpenAI Agent 设计指南

### 架构解析

- `agent-架构解析-ClaudeCode与Hermes.md` — Claude Code 与 Hermes 架构深度对比
- `agent-架构解析-LangChain与LangGraph.md` — LangChain 与 LangGraph 架构解析
- `agent-架构解析-三agent对比.md` — 三种 Agent 架构横向对比

### 工具设计

- `writing-effective-tools-for-agents.md` — Anthropic 工具设计最佳实践

### 学习路径与资料

- `agent-学习资料索引.md` — 全部资料导航地图（迷路先看这里）
- `agent-完整学习计划-超详细版.md` — 9 个月完整学习计划
- `agent-工程师学习路径-前端转型.md` — 前端转 Agent 学习路径
- `bilibili-agent-视频资源汇总.md` — B 站 Agent 视频教程汇总

### 踩坑与实战

- `agent-踩坑指南-100条.md` — 120 条 Agent 开发血泪教训
- `openspec-规格驱动开发.md` — OpenSpec 规格驱动开发框架

### 面试题库

| 文件 | 主题 |
|------|------|
| `agent-面试题-01-LLM基础与Prompt工程.md` | LLM 基础与 Prompt 工程 |
| `agent-面试题-02-ToolUse与Agent架构.md` | Tool Use 与 Agent 架构 |
| `agent-面试题-03-RAG与向量数据库.md` | RAG 与向量数据库 |
| `agent-面试题-04-*.md` | LangChain / LangGraph、MCP 协议、大厂真题 |
| `agent-面试题-05-MCP协议.md` | MCP 协议精选 |
| `agent-面试题-06-*.md` | Multi-Agent、框架、大厂补充真题 |
| `agent-面试题-07-*.md` | MCP 协议、工程化与生产部署 |
| `agent-面试题-08-MultiAgent系统设计.md` | Multi-Agent 系统设计 |

---

## 快速开始

### 环境要求

- Python 3.12+（项目用 `uv` 管理依赖）
- Node.js 18+（运行 `examples/node/` 下的示例）

### 安装依赖

```bash
cd code
uv sync
```

### 配置 API Key

在 `code/.env` 文件中配置（此文件不会被提交到 git）：

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://your-api-base      # OpenAI 兼容接口
LLM_MODEL=your-model-name
DASHSCOPE_API_KEY=sk-xxxxxxxx              # 通义千问（demo_qwen 用）
```

### 运行

```bash
cd code

# 最小 Agent Loop
python -m min_agent.main

# Agent 测评框架（模拟模式，无需 Key）
python -m agent_harness.main

# 多 Agent 协作演示（纯模拟，无需 Key）
python multi_agent_demo.py

# LangChain 教程第一篇
uv run python learn_langchain/01_hello_chain.py

# LangGraph 教程第一篇
uv run python learn_langgraph/01_basics.py
```

### 运行测试

```bash
cd code
pytest tests/ -v
```

---

