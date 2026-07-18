# Claude Code & Hermes 架构解析

> 面向前端转 Agent 开发工程师的深度技术文档
> 日期：2026-07-03

---

## 目录

1. [Claude Code 整体架构](#1-claude-code-整体架构)
2. [Agent Loop 实现原理](#2-agent-loop-实现原理)
3. [Hermes Agent 架构](#3-hermes-agent-架构)
4. [核心数据流：端到端全链路](#4-核心数据流端到端全链路)
5. [工具定义最佳实践](#5-工具定义最佳实践)
6. [Context Engineering](#6-context-engineering)
7. [多层 Agent 架构](#7-多层-agent-架构)
8. [综合代码示例](#8-综合代码示例)

---

## 1. Claude Code 整体架构

Claude Code 是 Anthropic 官方的 CLI 工具，它不只是一个「聊天界面」，而是一个完整的 **Agent 运行时（Runtime）**。理解它的架构，能帮你建立对所有 Agent 系统的底层直觉。

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                          │
│                                                                 │
│  ┌──────────┐    ┌──────────────────────────────────────────┐  │
│  │  REPL    │    │              Agent Runtime                │  │
│  │  Layer   │───▶│                                          │  │
│  │          │    │  ┌─────────────┐   ┌──────────────────┐  │  │
│  │ 用户输入  │    │  │ Context Mgr │   │  Tool Registry   │  │  │
│  │ 命令解析  │    │  │             │   │                  │  │  │
│  │ 输出渲染  │    │  │ system.md   │   │ bash / read /    │  │  │
│  └──────────┘    │  │ CLAUDE.md   │   │ write / edit /   │  │  │
│                  │  │ history     │   │ glob / grep /    │  │  │
│                  │  │ file_tree   │   │ web_fetch ...    │  │  │
│                  │  └──────┬──────┘   └────────┬─────────┘  │  │
│                  │         │                    │            │  │
│                  │         ▼                    ▼            │  │
│                  │  ┌─────────────────────────────────────┐  │  │
│                  │  │           Agent Loop Core           │  │  │
│                  │  │                                     │  │  │
│                  │  │  build_prompt() → LLM call()        │  │  │
│                  │  │       → parse_response()            │  │  │
│                  │  │       → execute_tools()             │  │  │
│                  │  │       → update_context()            │  │  │
│                  │  └─────────────────────────────────────┘  │  │
│                  └──────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Permission Model                       │  │
│  │  settings.json → allow/deny rules → prompt on unknown   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      Hook System                         │  │
│  │  PreToolUse / PostToolUse / Stop / Notification         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Subagent Mechanism                      │  │
│  │  Task tool → spawn subagent → isolated context           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 REPL 层

REPL（Read-Eval-Print Loop）是用户与系统交互的入口，负责：

- **Read**：读取用户输入（支持多行、粘贴代码）
- **Eval**：将输入交给 Agent Runtime 处理
- **Print**：将 Agent 输出以 Markdown 格式渲染到终端

REPL 本身是 **无状态** 的，状态全部由 Context Manager 持有。这个设计让 REPL 可以随时重启而不丢失对话历史。

### 1.3 工具系统（Tool System）

工具是 Claude Code 能力的核心扩展点。每个工具本质上是一个 **结构化函数描述 + 执行器**：

```typescript
interface Tool {
  name: string;                    // 工具唯一标识
  description: string;             // 给 LLM 看的自然语言描述（极其重要）
  input_schema: JSONSchema;        // 输入参数的 JSON Schema
  execute(input: any): Promise<ToolResult>;  // 实际执行逻辑
}

interface ToolResult {
  content: string | ContentBlock[];
  is_error?: boolean;
}
```

Claude Code 内置工具集：

| 工具名 | 功能 | 危险等级 |
|--------|------|---------|
| `Bash` | 执行 shell 命令 | 高 |
| `Read` | 读取文件内容 | 低 |
| `Write` | 写入文件 | 中 |
| `Edit` | 精准替换文件内容 | 中 |
| `Glob` | 文件模式匹配 | 低 |
| `Grep` | 文件内容搜索 | 低 |
| `WebFetch` | 抓取网页内容 | 中 |
| `WebSearch` | 搜索网络 | 低 |
| `Task` | 派生子 Agent | 高 |

### 1.4 权限模型（Permission Model）

权限模型是 Claude Code 的安全护栏。它基于 **allowlist/denylist + 运行时询问** 三层机制：

```
工具调用请求
      │
      ▼
┌─────────────────────────┐
│  检查 settings.json 的   │
│  allow 规则              │──── 匹配 ──→ 直接执行
└─────────────────────────┘
      │ 不匹配
      ▼
┌─────────────────────────┐
│  检查 settings.json 的   │
│  deny 规则               │──── 匹配 ──→ 拒绝执行
└─────────────────────────┘
      │ 不匹配
      ▼
┌─────────────────────────┐
│  向用户弹出询问框         │──── 用户确认 ──→ 执行并记忆
│  "是否允许执行 X？"      │──── 用户拒绝 ──→ 拒绝执行
└─────────────────────────┘
```

`settings.json` 示例：

```json
{
  "permissions": {
    "allow": [
      "Read(*)",
      "Bash(git *)",
      "Bash(npm test)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(sudo *)"
    ]
  }
}
```

### 1.5 Hook 机制

Hook 系统让你在工具执行的关键时机插入自定义逻辑，类似于 Web 框架中的中间件：

```
用户请求
   │
   ▼
[PreToolUse Hook]  ←── 可以修改输入、阻止执行、记录日志
   │
   ▼
[工具执行]
   │
   ▼
[PostToolUse Hook] ←── 可以修改输出、触发副作用、发送通知
   │
   ▼
[Stop Hook]        ←── Agent 停止时触发，可以做清理工作
```

Hook 在 `settings.json` 中配置：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[AUDIT] Bash called: ' $CLAUDE_TOOL_INPUT | tee -a audit.log"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write $CLAUDE_TOOL_OUTPUT_FILE 2>/dev/null || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code finished'"
          }
        ]
      }
    ]
  }
}
```

### 1.6 Subagent 机制

Subagent 是 Claude Code 实现并行和隔离的核心机制。当主 Agent 调用 `Task` 工具时，会派生一个独立的子 Agent：

```
主 Agent（Orchestrator）
    │
    ├── Task("分析代码库结构") ──→ Subagent A（独立上下文、独立工具权限）
    │                                    │
    │                                    └── 返回结果给主 Agent
    │
    ├── Task("写单元测试") ──→ Subagent B（并行执行）
    │
    └── Task("更新文档") ──→ Subagent C（并行执行）
```

Subagent 的关键特性：
- **隔离上下文**：每个 subagent 有独立的对话历史，不会污染主 agent
- **独立权限**：可以为 subagent 配置不同的权限边界
- **结果汇总**：subagent 完成后，结果以文本形式返回给主 agent

---

## 2. Agent Loop 实现原理

Agent Loop 是所有 Agent 系统的核心，理解它就理解了 Agent 的本质。

### 2.1 Loop 的完整生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent Loop                             │
│                                                             │
│  ┌──────────┐                                              │
│  │  START   │                                              │
│  └────┬─────┘                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Build Context                                     │  │
│  │     - system prompt (CLAUDE.md + settings)            │  │
│  │     - conversation history                            │  │
│  │     - file tree snapshot                              │  │
│  │     - tool definitions                                │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. LLM Inference                                     │  │
│  │     POST /v1/messages                                 │  │
│  │     → streaming response                             │  │
│  │     → parse text blocks & tool_use blocks            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│            ┌────────────┴────────────┐                     │
│            │                         │                     │
│            ▼                         ▼                     │
│  ┌──────────────────┐    ┌─────────────────────────────┐  │
│  │  stop_reason =   │    │  stop_reason = "tool_use"   │  │
│  │  "end_turn"      │    │                             │  │
│  │                  │    │  3. Execute Tools           │  │
│  │  → EXIT LOOP     │    │     - check permissions     │  │
│  └──────────────────┘    │     - run PreToolUse hooks  │  │
│                           │     - execute tool          │  │
│                           │     - run PostToolUse hooks │  │
│                           │     - collect results       │  │
│                           └─────────────┬───────────────┘  │
│                                         │                   │
│                                         ▼                   │
│                           ┌─────────────────────────────┐  │
│                           │  4. Update Context          │  │
│                           │     append assistant msg    │  │
│                           │     append tool results     │  │
│                           └─────────────┬───────────────┘  │
│                                         │                   │
│                                         └──── LOOP BACK ──▶│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 消息结构（Message Structure）

Claude API 使用多轮对话格式。每轮工具调用会产生两条消息：

```python
# 完整的消息历史结构
messages = [
    # 第一轮：用户提问
    {
        "role": "user",
        "content": "帮我读取 main.py 并找出所有函数"
    },
    # 第一轮：助手决定调用工具
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "我来读取 main.py 的内容。"
            },
            {
                "type": "tool_use",
                "id": "tool_abc123",
                "name": "Read",
                "input": {"file_path": "/path/to/main.py"}
            }
        ]
    },
    # 第一轮：工具执行结果（以 user 角色返回）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_abc123",
                "content": "def hello():\n    pass\n\ndef world():\n    pass"
            }
        ]
    },
    # 第二轮：助手基于工具结果生成最终回答
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "main.py 中包含两个函数：`hello()` 和 `world()`。"
            }
        ]
    }
]
```

关键点：工具结果以 **user 角色** 发送，这是 Claude API 的设计规范。

### 2.3 并行工具调用（Parallel Tool Use）

Claude 可以在一次响应中返回多个 `tool_use` block，实现并行执行：

```python
# 助手一次返回多个工具调用
assistant_message = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "我同时读取这三个文件："},
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
        {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "b.py"}},
        {"type": "tool_use", "id": "t3", "name": "Read", "input": {"file_path": "c.py"}},
    ]
}

# 并行执行所有工具，收集结果
import asyncio

async def execute_parallel(tool_calls):
    tasks = [execute_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks)
    return results

# 所有结果打包成一条 user 消息
tool_results_message = {
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "a.py 内容..."},
        {"type": "tool_result", "tool_use_id": "t2", "content": "b.py 内容..."},
        {"type": "tool_result", "tool_use_id": "t3", "content": "c.py 内容..."},
    ]
}
```

### 2.4 上下文窗口管理

Agent Loop 面临的最大挑战之一是 **上下文窗口溢出**。Claude Code 的策略：

```
当前 token 使用量
        │
        ▼
┌───────────────────────────────────────────┐
│  < 50% 窗口容量  → 完整保留历史           │
│  50%-80% 容量    → 压缩早期工具调用结果   │
│  > 80% 容量      → 触发摘要压缩机制       │
│  > 95% 容量      → 强制截断，只保留关键帧 │
└───────────────────────────────────────────┘
```

---

## 3. Hermes Agent 架构

> Hermes Agent 是 Nous Research 于 2026 年 2 月发布的开源自主 AI 智能体。它不是一个模型系列，而是一个完整的 Agent 系统——能部署在服务器上、跨会话持续学习、越用越聪明。MIT 协议开源，无 telemetry，无云锁定。

### 3.1 Hermes Agent 简介

Hermes Agent 由 Nous Research 构建，核心亮点是「内置学习循环」——它是目前唯一一个能从经验中自动创建技能（Skill）、在使用过程中持续改进、主动提示自己持久化知识，并在会话间不断深化对用户建模的 Agent。

与 Claude Code / Codex CLI 这类「单次会话型」编程助手不同，Hermes 强调**持久化**和**自我改进**：

- **持久记忆**：跨重启记住项目、偏好和上下文，不是每次从零开始
- **Skills 系统**：Agent 自己把常用工作流保存为可复用的 Skill，下次遇到类似任务直接加载
- **多平台消息网关**：支持 Telegram、Discord、Slack、WhatsApp、钉钉、飞书、企业微信等 20+ 平台，不局限于终端
- **随处运行**：本地、Docker、SSH、Daytona、Modal、Singularity 六种终端后端，从 5 美元 VPS 到 serverless 都能部署
- **MCP 集成**：可连接任意 MCP 服务器扩展工具能力
- **MoA（Mixture of Agents）**：内置多模型聚合，让多个模型先各自分析，再由聚合器综合决策，提升困难任务质量

### 3.2 核心架构

Hermes 的顶层架构可以概括为「入口点 → AIAgent 核心 → 存储与后端」三层：

```
┌─────────────────────────────────────────────────────────────┐
│ 入口点 │
│ CLI (cli.py) │ Gateway (gateway/run.py) │ ACP / API / Batch │
└─────────────┬──────────────┬────────────────────┬────────────┘
 │ │ │
 ▼ ▼ ▼
┌─────────────────────────────────────────────────────────────┐
│ AIAgent (run_agent.py) —— 核心对话循环 │
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Prompt │ │ Provider │ │ Tool │ │
│ │ Builder │ │ Resolution │ │ Dispatch │ │
│ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
│ │ │ │ │
│ ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐ │
│ │ Compression │ │ 3 API Modes │ │ Tool Registry│ │
│ │ & Caching │ │ chat_compl. │ │ 70+ tools │ │
│ │ │ │ codex_resp. │ │ 28 toolsets │ │
│ │ │ │ anthropic │ │ │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────┴─────────────────┴─────────────────┴───────────────┘
 │ │
 ▼ ▼
┌───────────────────┐ ┌──────────────────────┐
│ Session Storage │ │ Tool Backends │
│ (SQLite + FTS5) │ │ Terminal (7 backends)│
│ hermes_state.py │ │ Browser (5 backends) │
│ gateway/session.py│ │ Web / MCP / File ... │
└───────────────────┘ └──────────────────────┘
```

关键子系统：

- **Agent 循环**（`run_agent.py`）：同步编排引擎，负责 provider 选择、prompt 构建、工具执行、重试、回退、回调、压缩和持久化
- **Prompt 系统**：从 SOUL.md（个性）、MEMORY.md/USER.md（记忆）、Skills、上下文文件（AGENTS.md、.hermes.md）等多源组装系统 prompt
- **Provider 解析**：支持 18+ 个 provider（Anthropic、OpenAI、Nous Portal、OpenRouter 等），自动映射 API 模式（chat_completions / codex_responses / anthropic_messages）
- **工具系统**：中央注册表（`tools/registry.py`），70+ 工具分 28 个 toolset，终端支持 6 种后端（local、Docker、SSH、Daytona、Modal、Singularity）
- **会话持久化**：SQLite + FTS5 全文检索，支持血缘追踪（跨压缩的父/子关系）和按平台隔离
- **消息 Gateway**：长驻进程，20 个平台适配器，统一会话路由、用户授权、斜杠命令分发

### 3.3 学习循环：为什么 Hermes「越用越聪明」

Hermes 的核心差异化能力是**闭环学习**：

1. **自动创建 Skill**：当 Agent 成功完成一个复杂任务（5+ 次工具调用）、或从错误中找到可行路径时，会调用 `skill_manage` 工具把该工作流保存为 Skill（存到 `~/.hermes/skills/`）
2. **Skill 自我改进**：下次使用 Skill 时，Agent 会根据实际执行情况不断 patch 优化它
3. **主动持久化**：Agent 会主动提示自己「该把这段知识写进 MEMORY.md 了」，而不是被动等待用户保存
4. **跨会话搜索**：基于 FTS5 的跨会话全文召回 + LLM 摘要，能搜到自己过去的对话记录
5. **用户建模**：通过 Honcho 进行辩证式用户建模，随着使用加深对用户的理解

### 3.4 Skills 系统详解

Skills 是 Hermes 的「程序性记忆」——Agent 按需加载的知识文档，遵循渐进式披露（progressive disclosure）模式以节省 token。

**Skill 加载层级：**

```
Level 0: skills_list() → [{name, description, category}, ...] (~3k tokens)
Level 1: skill_view(name) → Full content + metadata (varies)
Level 2: skill_view(name, path) → Specific reference file (varies)
```

**SKILL.md 格式：**

```yaml
---
name: my-skill
description: Brief description of what this does
version: 1.0.0
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    requires_toolsets: [terminal]
---

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes
```

**Agent 管理 Skill（skill_manage 工具）：**

| 操作 | 用途 | 关键参数 |
|------|------|---------|
| `create` | 从头创建新 skill | `name`, `content`（完整 SKILL.md） |
| `patch` | 针对性修复（首选） | `name`, `old_string`, `new_string` |
| `edit` | 重大结构性重写 | `name`, `content`（完整替换） |
| `delete` | 完全删除 | `name` |

### 3.5 Mixture of Agents（MoA）

MoA 是 Hermes 内置的多模型聚合机制，以「虚拟 provider」的形式集成在模型选择系统中：

**工作原理：**
1. 每个 turn 先并行运行配置的 reference models（advisors），它们只接收对话文本，不接收系统 prompt 和工具调用记录
2. 把各 advisor 的输出作为 private context 附加给 aggregator
3. aggregator 作为 acting model 接收正常 Hermes 工具 schema，执行工具调用
4. 下一 turn 重复同样的 MoA 流程

**配置示例：**

```yaml
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      max_tokens: 4096
```

**性能：** 在 HermesBench 上，Opus aggregator + GPT-5.5 reference 的 MoA 配置得分 0.8202，超过单独运行 Opus（0.7607）或 GPT-5.5（0.7412）。

### 3.6 与 Claude Code 的核心差异

| 维度 | Claude Code | Hermes Agent |
|------|-------------|--------------|
| 定位 | 终端编程助手 | 自主 AI 智能体（服务器常驻） |
| 持久化 | 会话结束即丢失 | 跨会话持久记忆 + SQLite 存储 |
| 学习机制 | 无 | 内置学习循环，自动创建/改进 Skill |
| 交互入口 | 终端 | CLI + 20+ 消息平台 + API + Cron |
| 部署方式 | 本地运行 | 本地 / Docker / SSH / Daytona / Modal / Singularity |
| 多模型 | 单一模型 | MoA 多模型聚合 + 18+ provider |
| 工具数量 | 约 10 个 | 70+ 工具，28 个 toolset |
| 安全机制 | Hook 中间件 | 命令审批 + 授权 + 容器隔离 |
| 开源协议 | 未开源 | MIT |

---

## 4. 核心数据流：端到端全链路

### 4.1 完整数据流图

```
用户输入
   "帮我重构 utils.py，把重复代码提取成函数"
                │
                ▼
┌──────────────────────────────────────────────────────┐
│                  Context Builder                      │
│                                                      │
│  system_prompt = load_claude_md() +                  │
│                  load_settings() +                   │
│                  build_tool_descriptions()           │
│                                                      │
│  messages = conversation_history +                   │
│             [{"role": "user", "content": input}]     │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│              Token Budget Check                       │
│                                                      │
│  total_tokens = count_tokens(system + messages)      │
│  if total_tokens > threshold:                        │
│      messages = compress_history(messages)           │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│              Claude API Call                          │
│                                                      │
│  POST https://api.anthropic.com/v1/messages          │
│  {                                                   │
│    "model": "claude-opus-4-8",                       │
│    "max_tokens": 8192,                               │
│    "system": system_prompt,                          │
│    "messages": messages,                             │
│    "tools": tool_definitions,                        │
│    "stream": true                                    │
│  }                                                   │
└─────────────────────────┬────────────────────────────┘
                          │
              ┌───────────┴──────────────┐
              │                          │
              ▼                          ▼
     stop_reason =              stop_reason =
     "end_turn"                 "tool_use"
              │                          │
              │                          ▼
              │              ┌──────────────────────┐
              │              │   Tool Dispatcher    │
              │              │                      │
              │              │  for each tool_use:  │
              │              │    1. check perms    │
              │              │    2. pre-hooks      │
              │              │    3. execute        │
              │              │    4. post-hooks     │
              │              │    5. collect result │
              │              └──────────┬───────────┘
              │                         │
              │              ┌──────────▼───────────┐
              │              │  Append to Context   │
              │              │  → loop back to API  │
              │              └──────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────┐
│              Output Renderer                          │
│                                                      │
│  render_markdown(response_text)                      │
│  update_conversation_history()                       │
│  trigger_stop_hooks()                                │
└──────────────────────────────────────────────────────┘
```

### 4.2 工具执行详细流程

```python
async def execute_tool(tool_use_block: dict) -> dict:
    """工具执行的完整流程"""
    tool_name = tool_use_block["name"]
    tool_input = tool_use_block["input"]
    tool_id = tool_use_block["id"]
    
    # Step 1: 权限检查
    permission = check_permission(tool_name, tool_input)
    if permission == "deny":
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "is_error": True,
            "content": f"Permission denied for {tool_name}"
        }
    if permission == "ask":
        confirmed = await ask_user_permission(tool_name, tool_input)
        if not confirmed:
            return error_result(tool_id, "User denied permission")
    
    # Step 2: 执行 PreToolUse Hooks
    modified_input = await run_hooks("PreToolUse", tool_name, tool_input)
    if modified_input.get("block"):
        return error_result(tool_id, modified_input["reason"])
    
    # Step 3: 执行工具
    try:
        tool = TOOL_REGISTRY[tool_name]
        result = await tool.execute(modified_input.get("input", tool_input))
    except Exception as e:
        result = ToolResult(content=str(e), is_error=True)
    
    # Step 4: 执行 PostToolUse Hooks
    await run_hooks("PostToolUse", tool_name, tool_input, result)
    
    # Step 5: 返回标准格式
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": result.content,
        "is_error": result.is_error
    }
```

---

## 5. 工具定义最佳实践

### 5.1 Schema 设计原则

工具定义是 Agent 能力的边界，直接影响 LLM 的工具选择质量。

```python
# ❌ 糟糕的工具定义
bad_tool = {
    "name": "file_op",
    "description": "文件操作",
    "input_schema": {
        "type": "object",
        "properties": {
            "op": {"type": "string"},
            "file": {"type": "string"},
            "data": {"type": "string"}
        }
    }
}

# ✅ 优秀的工具定义
good_read_tool = {
    "name": "read_file",
    "description": """读取本地文件的内容。

适用场景：
- 查看源代码文件
- 读取配置文件
- 检查日志内容

注意事项：
- 文件路径必须是绝对路径（以 / 开头）
- 二进制文件（图片、压缩包等）无法读取
- 单次最多读取 2000 行，超出时使用 offset 和 limit 参数分批读取
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要读取的文件的绝对路径，例如 /home/user/project/main.py"
            },
            "offset": {
                "type": "integer",
                "description": "从第几行开始读取（从 1 开始计数），默认从第 1 行开始",
                "minimum": 1
            },
            "limit": {
                "type": "integer",
                "description": "最多读取多少行，默认 2000 行",
                "minimum": 1,
                "maximum": 2000
            }
        },
        "required": ["file_path"]
    }
}
```

### 5.2 描述写法的核心要素

好的工具描述包含四个要素：

```
工具描述 = 一句话说明 + 适用场景 + 不适用场景 + 参数说明

示例：
"一句话说明：在文件系统中按模式搜索文件路径。

适用场景：
- 查找特定扩展名的文件（*.py, *.ts）
- 在项目中定位特定文件
- 列出某目录下的所有文件

不适用场景：
- 搜索文件内容（请使用 grep_in_files 工具）
- 需要递归搜索时（请使用 ** 通配符）

参数说明：
- pattern: glob 模式，支持 * 和 ** 通配符
- cwd: 搜索起始目录，默认为当前工作目录"
```

### 5.3 错误处理策略

```python
class RobustTool:
    """具有完善错误处理的工具基类"""
    
    async def execute(self, input: dict) -> ToolResult:
        try:
            # 1. 输入验证
            self.validate_input(input)
            
            # 2. 执行核心逻辑
            result = await self._execute_core(input)
            
            # 3. 返回成功结果（包含足够上下文）
            return ToolResult(
                content=self.format_success(result),
                is_error=False
            )
            
        except ValidationError as e:
            # 输入错误：告诉 LLM 如何修正
            return ToolResult(
                content=f"输入参数错误：{e}\n\n正确用法示例：{self.usage_example()}",
                is_error=True
            )
        except PermissionError as e:
            # 权限错误：不要重试
            return ToolResult(
                content=f"权限不足，无法执行此操作：{e}",
                is_error=True
            )
        except TimeoutError:
            # 超时错误：可以重试
            return ToolResult(
                content="操作超时。如果文件较大，请使用 offset 和 limit 参数分批处理。",
                is_error=True
            )
        except Exception as e:
            # 未知错误：提供足够信息排查
            return ToolResult(
                content=f"未知错误：{type(e).__name__}: {e}",
                is_error=True
            )
    
    def format_success(self, result) -> str:
        """成功结果格式化：包含结果 + 有用的元信息"""
        raise NotImplementedError
```

### 5.4 工具结果的信息密度

```python
# ❌ 信息不足的结果
return ToolResult(content="OK")

# ❌ 信息过多的结果（浪费 token）
return ToolResult(content=entire_10000_line_file)

# ✅ 恰当信息密度的结果
def format_file_read_result(path: str, content: str, total_lines: int) -> str:
    lines = content.split('\n')
    return f"""文件：{path}
总行数：{total_lines}
已读取：第 1-{len(lines)} 行

```
{content}
```
{"（文件已截断，使用 offset 参数读取更多内容）" if total_lines > len(lines) else "（文件已完整读取）"}
"""
```

---

## 6. Context Engineering

Context Engineering 是 Agent 开发中最被低估的技能。**传给 LLM 的信息质量，直接决定输出质量。**

### 6.1 Context 的组成部分

```
完整 Context = System Prompt + Messages History + Tool Definitions
                                                          
System Prompt 组成：
├── 角色定义（你是谁，你能做什么）
├── 项目上下文（CLAUDE.md）
├── 行为规范（如何回应，避免什么）
├── 工具使用指南（何时用哪个工具）
└── 当前任务背景（可选）

Messages History 组成：
├── 早期对话（可能被压缩）
├── 工具调用历史（包含工具入参和结果）
└── 最近几轮对话（保持完整）
```

### 6.2 System Prompt 工程技巧

```python
def build_system_prompt(
    project_root: str,
    current_date: str,
    git_status: str,
) -> str:
    
    # 1. 加载项目说明
    claude_md = load_file_safe(f"{project_root}/CLAUDE.md", default="")
    
    # 2. 构建环境信息（避免 LLM 猜测）
    env_context = f"""
## 运行环境
- 当前日期：{current_date}
- 工作目录：{project_root}
- 操作系统：{get_os_info()}
- Shell：{os.environ.get('SHELL', 'unknown')}
"""
    
    # 3. 构建 Git 状态（帮助理解代码上下文）
    git_context = f"""
## 当前 Git 状态
{git_status}
""" if git_status else ""
    
    # 4. 工具使用指南
    tool_guide = """
## 工具使用原则
- 操作文件前，先用 Read 工具确认文件内容
- 修改文件优先使用 Edit 工具（精准替换），避免 Write 整体覆盖
- 使用 Bash 工具时，优先使用无副作用的只读命令
"""
    
    return "\n\n".join(filter(None, [
        claude_md,
        env_context,
        git_context,
        tool_guide
    ]))
```

### 6.3 对话历史压缩策略

```python
class ContextCompressor:
    """智能压缩对话历史，保留关键信息"""
    
    def compress(self, messages: list, budget: int) -> list:
        current_tokens = count_tokens(messages)
        
        if current_tokens <= budget:
            return messages
        
        # 策略1：压缩工具调用结果（通常最占空间）
        compressed = self._compress_tool_results(messages)
        if count_tokens(compressed) <= budget:
            return compressed
        
        # 策略2：摘要早期对话
        compressed = self._summarize_early_messages(compressed)
        if count_tokens(compressed) <= budget:
            return compressed
        
        # 策略3：只保留最近 N 轮 + 摘要
        return self._keep_recent_with_summary(messages, budget)
    
    def _compress_tool_results(self, messages: list) -> list:
        """把大型工具结果替换为摘要"""
        result = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                compressed_content = []
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if len(content) > 500:
                            block = {
                                **block,
                                "content": f"[结果已压缩，原长度 {len(content)} 字符]\n{content[:200]}..."
                            }
                    compressed_content.append(block)
                result.append({**msg, "content": compressed_content})
            else:
                result.append(msg)
        return result
```

### 6.4 动态注入上下文

```python
# 根据用户请求的关键词，动态注入相关文件内容
async def inject_relevant_context(
    user_message: str,
    project_root: str
) -> str:
    """在 system prompt 末尾注入相关文件内容"""
    
    additions = []
    
    # 如果提到特定文件，提前读取
    file_refs = extract_file_references(user_message)
    for file_path in file_refs[:3]:  # 最多 3 个文件
        content = read_file_safe(file_path)
        if content:
            additions.append(f"## 相关文件：{file_path}\n```\n{content[:2000]}\n```")
    
    # 如果是代码任务，注入项目结构
    if is_coding_task(user_message):
        tree = get_project_tree(project_root, depth=2)
        additions.append(f"## 项目结构\n```\n{tree}\n```")
    
    return "\n\n".join(additions)
```

---

## 7. 多层 Agent 架构

### 7.1 Orchestrator + Subagent 模式

```
┌────────────────────────────────────────────────────────────┐
│                    用户请求                                 │
│         "帮我做一个完整的代码审查，包括安全检查"             │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│               Orchestrator Agent（编排层）                  │
│                                                            │
│  职责：任务分解、调度、结果汇总                              │
│  工具：Task（派生子 Agent）                                 │
│                                                            │
│  分解结果：                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Task 1   │  │ Task 2   │  │ Task 3   │  │ Task 4   │  │
│  │ 代码质量  │  │ 安全漏洞  │  │ 测试覆盖  │  │ 文档完整  │  │
│  │ 检查     │  │ 扫描     │  │ 分析     │  │ 性检查   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼─────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Subagent A│  │Subagent B│  │Subagent C│  │Subagent D│
│          │  │          │  │          │  │          │
│工具：     │  │工具：     │  │工具：     │  │工具：     │
│Read      │  │Read      │  │Read      │  │Read      │
│Bash      │  │Bash      │  │Bash      │  │Grep      │
│          │  │(安全扫描) │  │(pytest)  │  │          │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ 结果汇总      │
                   │ 生成报告      │
                   └──────────────┘
```

### 7.2 实现 Task 工具

```python
class TaskTool:
    """
    派生子 Agent 的工具
    子 Agent 在独立上下文中运行，完成后返回结果字符串
    """
    
    name = "Task"
    description = """在独立上下文中启动一个子 Agent 完成特定任务。
    
适用场景：
- 需要并行执行多个独立任务
- 子任务需要大量工具调用，避免污染主对话上下文
- 任务足够独立，可以用一段文字描述清楚

返回：子 Agent 完成任务后的最终文字输出。
"""
    
    async def execute(self, input: dict) -> ToolResult:
        task_description = input["description"]
        context = input.get("context", "")
        
        # 创建子 Agent 实例（独立的消息历史）
        subagent = AgentLoop(
            system_prompt=self.build_subagent_system(context),
            tools=self.get_subagent_tools(input.get("allowed_tools")),
            max_iterations=input.get("max_iterations", 50),
        )
        
        # 运行子 Agent
        result = await subagent.run(task_description)
        
        return ToolResult(
            content=f"子任务完成。结果如下：\n\n{result}",
            is_error=False
        )
```

### 7.3 多 Agent 通信模式

```
模式一：中心化（Hub and Spoke）
         ┌─────────────┐
         │ Orchestrator │
         └──────┬──────┘
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Agent A     Agent B     Agent C
（适合：任务依赖关系复杂，需要中心协调）

模式二：链式（Pipeline）
 Agent A → Agent B → Agent C → 结果
（适合：每步处理后传递给下一步的流水线任务）

模式三：树形（Hierarchical）
                 Root
                /    \
           Sub-1      Sub-2
          /    \      /    \
       Sub-1a  Sub-1b  Sub-2a  Sub-2b
（适合：任务可以递归分解的场景）
```

---

## 8. 综合代码示例

### 8.1 最小可运行的 Agent Loop

```python
import asyncio
import anthropic
from typing import Any

client = anthropic.Anthropic()

# 工具定义
TOOLS = [
    {
        "name": "read_file",
        "description": "读取文件内容",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件绝对路径"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file", 
        "description": "写入文件内容",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件绝对路径"},
                "content": {"type": "string", "description": "要写入的内容"}
            },
            "required": ["path", "content"]
        }
    }
]

# 工具执行器
def execute_tool(name: str, input: dict) -> str:
    if name == "read_file":
        try:
            with open(input["path"], "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"
    
    elif name == "write_file":
        try:
            with open(input["path"], "w") as f:
                f.write(input["content"])
            return f"成功写入 {len(input['content'])} 字符到 {input['path']}"
        except Exception as e:
            return f"Error: {e}"
    
    return f"Unknown tool: {name}"

# Agent Loop 核心
def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        # 调用 Claude API
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )
        
        # 将助手回复加入历史
        messages.append({"role": "assistant", "content": response.content})
        
        # 检查是否结束
        if response.stop_reason == "end_turn":
            # 提取最后一条文本回复
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""
        
        # 处理工具调用
        if response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # 将工具结果加入历史，继续循环
            messages.append({"role": "user", "content": tool_results})

# 运行
if __name__ == "__main__":
    result = run_agent("读取 /tmp/test.txt 的内容，然后告诉我有多少行")
    print(result)
```

### 8.2 带权限控制的增强版 Loop

```python
import re
from enum import Enum

class Permission(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"

class PermissionManager:
    def __init__(self, allow_patterns: list, deny_patterns: list):
        self.allow = allow_patterns
        self.deny = deny_patterns
    
    def check(self, tool_name: str, tool_input: dict) -> Permission:
        call_str = f"{tool_name}({str(tool_input)})"
        
        # 先检查 deny（黑名单优先）
        for pattern in self.deny:
            if re.match(pattern, call_str):
                return Permission.DENY
        
        # 再检查 allow（白名单）
        for pattern in self.allow:
            if re.match(pattern, call_str):
                return Permission.ALLOW
        
        # 默认询问
        return Permission.ASK

class SecureAgentLoop:
    def __init__(self, permission_manager: PermissionManager):
        self.pm = permission_manager
        self.client = anthropic.Anthropic()
    
    def run(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        
        for iteration in range(100):  # 防止无限循环
            response = self.client.messages.create(
                model="claude-opus-4-8",
                max_tokens=4096,
                tools=TOOLS,
                messages=messages,
            )
            
            messages.append({"role": "assistant", "content": response.content})
            
            if response.stop_reason == "end_turn":
                return self._extract_text(response)
            
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    
                    perm = self.pm.check(block.name, block.input)
                    
                    if perm == Permission.DENY:
                        result = f"[DENIED] 工具 {block.name} 已被禁止执行"
                    elif perm == Permission.ASK:
                        print(f"\n[权限请求] 是否允许执行：{block.name}({block.input})？(y/n)")
                        if input().strip().lower() != 'y':
                            result = "[DENIED] 用户拒绝了此操作"
                        else:
                            result = execute_tool(block.name, block.input)
                    else:
                        result = execute_tool(block.name, block.input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                
                messages.append({"role": "user", "content": tool_results})
        
        return "已达到最大迭代次数"
    
    def _extract_text(self, response) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

# 使用示例
pm = PermissionManager(
    allow_patterns=["read_file.*"],  # 读文件总是允许
    deny_patterns=["write_file.*/etc/.*"],  # 禁止写系统文件
)
agent = SecureAgentLoop(pm)
result = agent.run("分析 /home/user/project 目录下的代码结构")
```

---

## 总结

Claude Code 的架构揭示了现代 Agent 系统的核心设计哲学：

1. **工具是能力扩展点**：通过定义良好的 JSON Schema + 清晰的描述，让 LLM 知道何时如何使用工具
2. **Agent Loop 是驱动引擎**：用 `stop_reason` 驱动循环，直到任务完成
3. **权限模型是安全护栏**：在能力和安全之间找到平衡
4. **Context 是 Agent 的大脑**：传给 LLM 的信息质量决定输出质量
5. **多层架构实现复杂任务**：Orchestrator + Subagent 让复杂任务变得可管理

掌握这些原则，你就掌握了构建任何 Agent 系统的核心方法论。
