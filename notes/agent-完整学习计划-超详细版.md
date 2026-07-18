# AI Agent 开发完整学习计划（超详细版）
> 目标学员：6 年前端经验（TypeScript/React/Node.js），零 AI 工程经验  
> 预计周期：9 个月  
> 最后更新：2026-07-03

---

## 目录

1. [学习路径总览](#1-学习路径总览)
2. [每日时间分配建议](#2-每日时间分配建议)
3. [前端技能迁移对照表（详细版）](#3-前端技能迁移对照表详细版)
4. [推荐资源总表](#4-推荐资源总表)
5. [学习误区与注意事项](#5-学习误区与注意事项)
6. [9 个月详细计划](#6-9-个月详细计划)
   - [第 1 月：LLM 基础 + Prompt 工程](#第-1-月llm-基础--prompt-工程)
   - [第 2 月：Tool Use + ReAct Agent](#第-2-月tool-use--react-agent)
   - [第 3 月：记忆系统 + RAG 基础](#第-3-月记忆系统--rag-基础)
   - [第 4 月：向量数据库 + 高级 RAG](#第-4-月向量数据库--高级-rag)
   - [第 5 月：LangChain 深度](#第-5-月langchain-深度)
   - [第 6 月：LangGraph + 状态机](#第-6-月langgraph--状态机)
   - [第 7 月：Multi-Agent（AutoGen / CrewAI）](#第-7-月multi-agentautogen--crewai)
   - [第 8 月：MCP 协议 + 上下文工程 + 生产化](#第-8-月mcp-协议--上下文工程--生产化)
   - [第 9 月：可观测性 + 评测 + 毕业项目](#第-9-月可观测性--评测--毕业项目)
7. [阶段里程碑检查表](#7-阶段里程碑检查表)
8. [毕业项目建议](#8-毕业项目建议)

---

## 1. 学习路径总览

```
[基础层]                  [工具层]                  [系统层]                  [生产层]
LLM原理                   Tool Use                  Multi-Agent               可观测性
Prompt工程    ──────>     ReAct Agent   ──────>     LangGraph     ──────>     评测体系
API调用                   记忆系统                  MCP协议                   部署运维
                          RAG / 向量DB              上下文工程
```

### 技术栈演进路线

| 阶段 | 核心技术 | 语言/框架 | 类比前端概念 |
|------|---------|-----------|------------|
| 月1-2 | LLM API + Prompt | TypeScript + Fetch | REST API 调用 |
| 月3-4 | RAG + 向量搜索 | TypeScript + Pinecone | 搜索引擎 + 数据库查询 |
| 月5-6 | LangChain + LangGraph | Python/TS + LangChain | React + Redux/Zustand |
| 月7 | AutoGen + CrewAI | Python | 微服务架构 |
| 月8 | MCP + 上下文工程 | TypeScript + Python | 组件通信协议 |
| 月9 | 评测 + 部署 | 全栈 | CI/CD + 监控 |

---

## 2. 每日时间分配建议

### 工作日（2 小时/天）

```
20:00 - 20:30  理论学习（阅读文档/看视频）
20:30 - 21:30  动手编码（核心实践时间）
21:30 - 22:00  复盘 + 记笔记 + 明日计划
```

### 周末（4-6 小时/天）

```
上午 09:00 - 11:00  深度学习（新概念、难点攻坚）
下午 14:00 - 16:00  项目实践（当周综合任务）
下午 16:00 - 17:00  回顾本周 + 整理笔记
晚上（可选）         看 YouTube 教程/读英文博客
```

### 每周时间分配比例

| 类型 | 占比 | 说明 |
|------|------|------|
| 视频课程 | 25% | DeepLearning.AI 等结构化课程 |
| 官方文档阅读 | 20% | LangChain/OpenAI/Anthropic 文档 |
| 动手编码 | 40% | 最重要，跟敲 + 自己改 + 原创 |
| 项目整合 | 10% | 每月综合项目 |
| 复盘记录 | 5% | 写笔记、总结错误 |

---

## 3. 前端技能迁移对照表（详细版）

### 3.1 概念类比

| 前端概念 | AI Agent 对应概念 | 详细说明 |
|---------|-----------------|---------|
| React 组件 | Agent 节点 | 都是可复用、有输入输出的单元 |
| Props / State | Agent 上下文 (Context) | 数据在组件/节点间传递 |
| Redux/Zustand Store | Agent 记忆系统 | 全局状态管理，跨节点共享数据 |
| React Router | Agent 路由/决策 | 根据条件决定走哪条分支 |
| useEffect 副作用 | Tool 调用 | 触发外部操作（API、文件、搜索） |
| React Query / SWR | RAG 检索层 | 异步获取、缓存数据 |
| Promise / async-await | LLM 流式响应 | 异步等待、逐步获得结果 |
| TypeScript 类型系统 | Prompt 结构化输出 | 约束输出格式，类似 zod schema |
| 中间件 (Express) | LangChain Chain | 请求经过一系列处理器 |
| WebSocket | LLM Streaming | 实时接收分块数据 |
| Monorepo (Turborepo) | Multi-Agent 系统 | 多个协作的独立模块 |
| 依赖注入 | Agent 工具注入 | 运行时动态绑定能力 |
| Event Emitter | Agent 消息总线 | 节点间异步通信 |
| 单元测试 (Vitest) | LLM 评测 (Evals) | 验证输出符合预期 |
| Storybook | Prompt Playground | 隔离测试单个组件/Prompt |

### 3.2 技术技能直接复用

| 前端技能 | 在 AI Agent 中的用法 | 重要程度 |
|---------|-------------------|---------|
| TypeScript 类型定义 | 定义 Tool 输入/输出 schema，用 Zod 验证 LLM 输出 | ★★★★★ |
| async/await + Promise | LLM API 调用，流式处理 | ★★★★★ |
| Node.js 生态 | 用 LangChain.js 开发 Agent，调用各种 SDK | ★★★★★ |
| fetch / axios | 调用 OpenAI/Anthropic/Ollama REST API | ★★★★☆ |
| JSON 解析/验证 | 解析 LLM 返回的结构化输出 | ★★★★☆ |
| 环境变量管理 (.env) | API Key 安全管理 | ★★★★☆ |
| npm/pnpm 包管理 | 安装 AI 相关库 | ★★★★☆ |
| Markdown 渲染 | 展示 LLM 输出 | ★★★☆☆ |
| React 状态管理 | 理解 Agent 状态流转 | ★★★★☆ |
| REST API 设计 | 设计 Agent 服务接口 | ★★★★☆ |

### 3.3 需要新学的思维模式

| 旧思维（前端） | 新思维（AI Agent） | 为什么不同 |
|-------------|-----------------|-----------|
| 输出是确定性的 | 输出是概率性的 | LLM 每次生成结果可能不同 |
| 代码错误有明确报错 | AI 失败是"幻觉"，看起来正确 | 需要评测和验证机制 |
| 调试看 console.log | 调试看 prompt trace | 问题在 prompt 而非代码逻辑 |
| 功能测试靠断言 | AI 测试靠评测集+LLM-as-judge | 输出是文本，很难精确断言 |
| 性能优化靠缓存 | 性能优化靠 prompt 压缩 + 缓存 | Token 是钱，上下文是资源 |
| 状态存 Redux | 状态存对话历史 + 向量DB | 记忆有多种层次和形式 |

---

## 4. 推荐资源总表

### 4.1 DeepLearning.AI 课程（全部免费）

| 课程名称 | 链接 | 建议学习时间 | 对应月份 |
|---------|------|------------|---------|
| ChatGPT Prompt Engineering for Developers | deeplearning.ai | 第 1-2 天 | 月1 |
| Building Systems with the ChatGPT API | deeplearning.ai | 第 3-5 天 | 月1 |
| LangChain for LLM Application Development | deeplearning.ai | 第 2 周 | 月1-2 |
| LangChain: Chat with Your Data | deeplearning.ai | 第 3-4 周 | 月3 |
| Functions, Tools and Agents with LangChain | deeplearning.ai | 月2 | 月2 |
| Build LLM Apps with LangChain.js | deeplearning.ai | 月2-3 | 月2-3 |
| AI Agents in LangGraph | deeplearning.ai | 月6 | 月6 |
| Multi AI Agent Systems with crewAI | deeplearning.ai | 月7 | 月7 |
| AI Agentic Design Patterns with AutoGen | deeplearning.ai | 月7 | 月7 |
| Building Your Own Database Agent | deeplearning.ai | 月4 | 月4 |
| Vector Databases: from Embeddings to Applications | deeplearning.ai | 月4 | 月4 |
| Knowledge Graphs for RAG | deeplearning.ai | 月4-5 | 月4-5 |
| Evaluating and Debugging Generative AI | deeplearning.ai | 月9 | 月9 |
| LLMOps | deeplearning.ai | 月9 | 月9 |
| Automated Testing for LLMOps | deeplearning.ai | 月9 | 月9 |

### 4.2 推荐书籍

| 书名 | 作者 | 难度 | 阅读时间 | 核心收益 |
|------|------|------|---------|---------|
| Building LLM Powered Applications | Valentina Alto | ⭐⭐ | 月1-2 | LLM 应用架构全景 |
| Prompt Engineering for LLMs | John Berryman | ⭐⭐ | 月1 | Prompt 系统化方法论 |
| AI Engineering | Chip Huyen | ⭐⭐⭐ | 月3-5 | 业界最权威的AI工程指南 |
| Designing Machine Learning Systems | Chip Huyen | ⭐⭐⭐⭐ | 月6-8 | ML系统设计，含评测 |
| The Developer's Guide to LLM Agents | 各作者 | ⭐⭐⭐ | 月5-7 | Agent 架构实战 |

### 4.3 必读文档/论文

| 资料名称 | 来源 | 建议月份 |
|---------|------|---------|
| OpenAI API 官方文档 | platform.openai.com/docs | 月1 |
| Anthropic Claude API 文档 | docs.anthropic.com | 月1 |
| LangChain 官方文档 | python.langchain.com | 月5 |
| LangGraph 官方文档 | langchain-ai.github.io/langgraph | 月6 |
| ReAct: Synergizing Reasoning and Acting (论文) | arxiv.org/abs/2210.03629 | 月2 |
| Attention Is All You Need (Transformer 原论文) | arxiv.org/abs/1706.03762 | 月1（选读）|
| Retrieval-Augmented Generation (RAG 论文) | arxiv.org/abs/2005.11401 | 月3 |
| Chain-of-Thought Prompting (CoT 论文) | arxiv.org/abs/2201.11903 | 月1 |
| Tree of Thoughts (ToT 论文) | arxiv.org/abs/2305.10601 | 月2 |
| MCP 协议规范 | modelcontextprotocol.io | 月8 |
| OpenAI Cookbook | github.com/openai/openai-cookbook | 全程参考 |
| Anthropic Prompt Library | docs.anthropic.com/prompt-library | 全程参考 |

### 4.4 YouTube 频道

| 频道名称 | 内容特色 | 订阅理由 |
|---------|---------|---------|
| Andrej Karpathy | LLM 底层原理，神级讲解 | "Let's build GPT from scratch" 系列必看 |
| Sam Witteveen | LangChain/LangGraph 实战 | 更新最快，跟进最新特性 |
| Matt Williams | Ollama 本地模型 | 本地部署入门 |
| AI Jason | Agent 实战项目 | 完整项目演示 |
| David Ondrej | AI Engineering | 工程实践向 |
| 1littlecoder | Hands-on AI tutorials | 代码密度高 |
| Two Minute Papers | AI 论文速览 | 跟进前沿研究 |
| Yannic Kilcher | 论文精读 | 深度理解原理 |

### 4.5 博客与社区

| 资源 | 类型 | 更新频率 |
|------|------|---------|
| Lilian Weng's Blog (lilianweng.github.io) | 技术博客 | 月更 |
| Simon Willison's Weblog | 工程博客 | 周更 |
| Sebastian Raschka's Magazine | Newsletter | 双周更 |
| The Batch (deeplearning.ai) | Newsletter | 周更 |
| Hugging Face Blog | 平台博客 | 周更 |
| r/LocalLLaMA | Reddit 社区 | 实时 |
| LangChain Discord | 技术社区 | 实时 |

---

## 5. 学习误区与注意事项

### 5.1 常见认知误区

> **误区 1：把 LLM 当确定性函数**
>
> 错误做法：期望相同 prompt 永远返回相同格式，不做输出验证
> 正确做法：永远假设 LLM 输出可能格式错误，用 Zod/Pydantic 验证，加重试逻辑

> **误区 2：越长的 Prompt 越好**
>
> 错误做法：把所有上下文都塞进 System Prompt，导致"迷失在中间"问题
> 正确做法：精炼 prompt，关键信息放开头和结尾，善用 RAG 按需检索

> **误区 3：直接用 LangChain 跳过基础**
>
> 错误做法：第一天就学 LangChain，不理解底层 API
> 正确做法：先用原始 API 手写 2 周，再引入框架，才能真正理解框架做了什么

> **误区 4：忽视 Token 成本**
>
> 错误做法：对话历史无限增长，不做裁剪
> 正确做法：从第一天就关注 token 用量，用 tiktoken 计算，设计上下文管理策略

> **误区 5：认为 Python 是唯一选择**
>
> 真相：TypeScript/LangChain.js 生态已经很成熟，前 3 个月可以全用 TypeScript
> 建议：月 5 开始学 Python，因为 LangChain/LangGraph Python 版功能更完整

> **误区 6：RAG = 把文档扔进去就行**
>
> 错误做法：直接向量化整个文档，期望魔法般工作
> 正确做法：学习 chunking 策略、embedding 选择、hybrid search、reranking 等优化手段

### 5.2 学习节奏建议

- **不要跳步**：每周任务是递进的，跳过基础会在后期卡壳
- **先跑通再深入**：每个概念先做出能运行的 demo，再深入理解原理
- **错误是财富**：遇到 LLM 行为诡异时，仔细分析 prompt trace，这比看教程更有价值
- **定期回顾**：每月最后一周要回顾整月内容，整理 "陌生词汇 → 熟悉概念" 的变化
- **不要追完美**：Agent 系统没有完美，先做出能用的 v1，再迭代优化

---

## 6. 9 个月详细计划

---

### 第 1 月：LLM 基础 + Prompt 工程

**月度目标**：能用 TypeScript 调用 LLM API，理解 Prompt 工程核心技巧，能写出质量稳定的 Prompt

---

#### 第 1 周：LLM 是什么，怎么调用

**理论学习**

- [ ] 看 Andrej Karpathy《Intro to Large Language Models》（YouTube，1小时）
- [ ] 阅读：什么是 Token、什么是 Embedding、Temperature/Top-P 参数含义
- [ ] 阅读 OpenAI API 文档：Chat Completions API
- [ ] 阅读 Anthropic 文档：Messages API

**代码练习**

```typescript
// 练习 1：最基础的 API 调用
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function chat(userMessage: string): Promise<string> {
  const response = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: '你是一个有帮助的助手。' },
      { role: 'user', content: userMessage },
    ],
    temperature: 0.7,
    max_tokens: 500,
  });
  return response.choices[0].message.content ?? '';
}
```

```typescript
// 练习 2：流式响应（对比 WebSocket，你会很熟悉这种模式）
async function streamChat(userMessage: string): Promise<void> {
  const stream = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: userMessage }],
    stream: true,
  });

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    process.stdout.write(delta);
  }
}
```

```typescript
// 练习 3：多轮对话（注意：历史要手动维护，这是关键！）
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

class SimpleChat {
  private history: Message[] = [];

  constructor(systemPrompt: string) {
    this.history.push({ role: 'system', content: systemPrompt });
  }

  async send(userMessage: string): Promise<string> {
    this.history.push({ role: 'user', content: userMessage });
    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: this.history,
    });
    const assistantMessage = response.choices[0].message.content ?? '';
    this.history.push({ role: 'assistant', content: assistantMessage });
    return assistantMessage;
  }
}
```

**本周作业**：写一个命令行对话 bot，能记住对话历史，且在历史超过 10 轮时自动截断旧消息

---

#### 第 2 周：Prompt 工程核心技术

**理论学习**

- [ ] DeepLearning.AI：《ChatGPT Prompt Engineering for Developers》全部（约 2 小时）
- [ ] 阅读 Anthropic Prompt Engineering Guide
- [ ] 学习以下技术：
  - **Zero-shot prompting**：直接问，不给例子
  - **Few-shot prompting**：给 2-5 个例子再问
  - **Chain-of-Thought (CoT)**：让模型"先思考再回答"
  - **Role prompting**：给模型设定角色
  - **Structured output**：要求 JSON 格式输出

**关键 Prompt 模板**

```
# Zero-shot
将以下英文翻译成中文：{text}

# Few-shot
将以下句子分类为正面/负面情感：
例子1：今天天气真好！→ 正面
例子2：这个产品太差了。→ 负面
现在分类：{sentence}

# Chain-of-Thought
解决下面的数学问题，请一步一步思考：
问题：{problem}
让我们逐步分析：

# Structured Output (JSON)
分析以下产品评价，返回 JSON 格式：
{"sentiment": "positive/negative/neutral", "score": 1-5, "key_points": []}
评价：{review}
```

**代码练习**

```typescript
// Zod 验证 LLM 结构化输出（这是你作为 TS 工程师的优势！）
import { z } from 'zod';

const ReviewSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  score: z.number().min(1).max(5),
  keyPoints: z.array(z.string()),
});

async function analyzeReview(review: string) {
  const prompt = `分析以下评价，只返回 JSON，不要任何其他文字：
  {"sentiment": "positive/negative/neutral", "score": 1-5, "keyPoints": ["..."]}
  
  评价：${review}`;

  const response = await chat(prompt);

  try {
    const parsed = JSON.parse(response);
    return ReviewSchema.parse(parsed); // 类型安全！
  } catch (e) {
    // LLM 输出不符合格式时的处理
    throw new Error(`LLM 输出格式错误: ${response}`);
  }
}
```

**本周作业**：写一个"产品评价分析器"，批量分析 20 条评价，输出结构化报告

---

#### 第 3 周：高级 Prompt 技术

**理论学习**

- [ ] 阅读：Tree of Thoughts 论文摘要
- [ ] 学习：Self-consistency（多次采样取最优）
- [ ] 学习：System Prompt 设计最佳实践
- [ ] 学习：Prompt 安全（Prompt Injection 攻击与防御）

**代码练习**

```typescript
// Self-consistency：同一问题问 5 次，取最多答案（多数投票）
async function selfConsistency(question: string, n = 5): Promise<string> {
  const answers = await Promise.all(
    Array.from({ length: n }, () => chat(question))
  );

  // 统计频率，取最多的答案
  const freq = new Map<string, number>();
  for (const ans of answers) {
    freq.set(ans, (freq.get(ans) ?? 0) + 1);
  }
  return [...freq.entries()].sort((a, b) => b[1] - a[1])[0][0];
}
```

**本周作业**：实现一个带"置信度"的问答系统，使用 self-consistency，返回答案和置信度百分比

---

#### 第 4 周：Token 管理 + 本月综合项目

**理论学习**

- [ ] 理解 Token 计数（tiktoken 库）
- [ ] 学习 Context Window 限制及应对策略
- [ ] 阅读：模型选择策略（GPT-4o vs Claude 3.5 Sonnet vs Gemini）

**代码练习**

```typescript
// Token 计数工具
import Tiktoken from 'tiktoken';

function countTokens(text: string, model = 'gpt-4o'): number {
  const enc = Tiktoken.encoding_for_model(model as any);
  const tokens = enc.encode(text);
  enc.free();
  return tokens.length;
}

// 对话历史修剪（保留最近 N 个 token）
function trimHistory(messages: Message[], maxTokens = 3000): Message[] {
  let total = 0;
  const result: Message[] = [];
  // 从最新消息开始倒序累加
  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = countTokens(messages[i].content);
    if (total + tokens > maxTokens) break;
    result.unshift(messages[i]);
    total += tokens;
  }
  return result;
}
```

**月度综合项目：智能简历分析助手**

需求：
1. 用户输入职位描述和简历内容
2. 系统分析匹配度（0-100分）
3. 指出优势和不足
4. 给出改进建议
5. 以 JSON 格式返回结果，用 Zod 验证

**月度里程碑检查**

- [ ] 能独立调用 OpenAI/Anthropic API
- [ ] 能维护多轮对话历史
- [ ] 能写出 Few-shot + CoT 组合 Prompt
- [ ] 能用 Zod 验证并处理 LLM 输出异常
- [ ] 理解 Token 计数，实现历史修剪
- [ ] 完成简历分析助手项目

---

### 第 2 月：Tool Use + ReAct Agent

**月度目标**：理解 Function Calling/Tool Use 机制，实现第一个能使用工具的 ReAct Agent

---

#### 第 1 周：Function Calling 基础

**理论学习**

- [ ] OpenAI Function Calling 官方文档
- [ ] 阅读 ReAct 论文（arxiv 2210.03629）
- [ ] DeepLearning.AI：《Functions, Tools and Agents with LangChain》

**代码练习**

```typescript
// Tool 定义（类比 TypeScript 接口定义）
const tools = [
  {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: '获取指定城市的天气信息',
      parameters: {
        type: 'object',
        properties: {
          city: {
            type: 'string',
            description: '城市名称，如：北京、上海',
          },
          unit: {
            type: 'string',
            enum: ['celsius', 'fahrenheit'],
            description: '温度单位',
          },
        },
        required: ['city'],
      },
    },
  },
];

// Tool 执行器（类比 Redux middleware）
async function executeTool(
  name: string,
  args: Record<string, unknown>
): Promise<string> {
  switch (name) {
    case 'get_weather':
      // 实际调用天气 API
      return JSON.stringify({ city: args.city, temp: 25, condition: '晴' });
    default:
      throw new Error(`未知 Tool: ${name}`);
  }
}
```

---

#### 第 2-3 周：ReAct Agent 实现

ReAct = **Re**asoning + **Act**ing，核心循环：

```
思考(Thought) → 行动(Action) → 观察(Observation) → 思考 → ...
```

**代码练习**

```typescript
// ReAct Agent 核心循环（这是整个 Agent 开发的基石）
interface AgentStep {
  thought: string;
  action?: { tool: string; args: Record<string, unknown> };
  observation?: string;
}

async function reactAgent(question: string): Promise<string> {
  const steps: AgentStep[] = [];
  const maxSteps = 10;

  let messages: Message[] = [
    {
      role: 'system',
      content: `你是一个会使用工具的 AI 助手。
      当你需要信息时，使用提供的工具。
      思考格式：先说明你的思考过程，再决定是否使用工具。`,
    },
    { role: 'user', content: question },
  ];

  for (let step = 0; step < maxSteps; step++) {
    const response = await client.chat.completions.create({
      model: 'gpt-4o',
      messages,
      tools,
      tool_choice: 'auto',
    });

    const choice = response.choices[0];

    // 如果没有 tool call，说明 Agent 有了最终答案
    if (!choice.message.tool_calls || choice.message.tool_calls.length === 0) {
      return choice.message.content ?? '无法回答';
    }

    // 执行 tool
    messages.push(choice.message);

    for (const toolCall of choice.message.tool_calls) {
      const toolArgs = JSON.parse(toolCall.function.arguments);
      const observation = await executeTool(toolCall.function.name, toolArgs);

      messages.push({
        role: 'tool',
        // @ts-expect-error tool_call_id is required
        tool_call_id: toolCall.id,
        content: observation,
      });

      console.log(`[Step ${step + 1}] Tool: ${toolCall.function.name}`);
      console.log(`  Args: ${JSON.stringify(toolArgs)}`);
      console.log(`  Result: ${observation}`);
    }
  }

  return '超出最大步数，未能完成任务';
}
```

**本月综合项目：个人研究助手 Agent**

功能：
1. 工具：网页搜索、计算器、日期时间、单位换算
2. 能回答"明天北京天气怎么样，要不要带伞？"
3. 能计算"如果我每月存 5000 元，年化 5%，10 年后有多少钱？"
4. 打印每一步的 Thought/Action/Observation

**月度里程碑检查**

- [ ] 理解 Function Calling 的完整流程（定义→调用→返回结果）
- [ ] 能实现至少 5 个实用工具函数
- [ ] 实现完整的 ReAct Agent 循环
- [ ] Agent 能正确处理多步推理（3步以上）
- [ ] 处理 Tool 执行失败的重试逻辑

---

### 第 3 月：记忆系统 + RAG 基础

**月度目标**：理解 Agent 记忆的四种类型，实现基础 RAG 系统

---

#### 记忆系统四层架构

| 记忆类型 | 类比 | 实现方式 | 存储介质 |
|---------|------|---------|---------|
| Sensory Memory | 感官输入（当前 prompt） | 当前消息 | 内存 |
| Working Memory | 短期工作台（对话历史） | messages 数组 | 内存 |
| Episodic Memory | 情景记忆（历史对话摘要） | 摘要 + 数据库 | 数据库 |
| Semantic Memory | 知识记忆（外部知识库） | RAG / 向量搜索 | 向量数据库 |

```typescript
// 对话摘要（压缩 Working Memory 到 Episodic Memory）
async function summarizeHistory(messages: Message[]): Promise<string> {
  const historyText = messages
    .map(m => `${m.role}: ${m.content}`)
    .join('\n');

  return await chat(
    `请用 3-5 句话总结以下对话的关键信息：\n${historyText}`
  );
}
```

#### RAG 基础实现

RAG = Retrieval-Augmented Generation（检索增强生成）

```
用户问题 → [检索] → 找到相关文档片段 → [生成] → 结合文档回答
```

```typescript
// 最简单的 RAG：基于关键词搜索
class SimpleRAG {
  private documents: { id: string; content: string; embedding?: number[] }[] = [];

  async addDocument(content: string): Promise<void> {
    const embedding = await this.embed(content);
    this.documents.push({
      id: crypto.randomUUID(),
      content,
      embedding,
    });
  }

  async embed(text: string): Promise<number[]> {
    const response = await client.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
    });
    return response.data[0].embedding;
  }

  cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dot / (normA * normB);
  }

  async search(query: string, topK = 3): Promise<string[]> {
    const queryEmbedding = await this.embed(query);
    return this.documents
      .map(doc => ({
        ...doc,
        score: this.cosineSimilarity(queryEmbedding, doc.embedding!),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(doc => doc.content);
  }

  async answer(question: string): Promise<string> {
    const contexts = await this.search(question);
    const prompt = `根据以下背景资料回答问题。如果资料中没有相关信息，请说"我不知道"。

背景资料：
${contexts.map((c, i) => `[${i + 1}] ${c}`).join('\n\n')}

问题：${question}`;

    return await chat(prompt);
  }
}
```

**月度综合项目：公司知识库问答系统**

功能：
1. 导入 PDF/TXT 文档
2. 分块（每块 500 字，50 字重叠）
3. 向量化存储（内存版）
4. 用户提问 → 检索相关片段 → 生成回答，并标注来源

**月度里程碑检查**

- [ ] 理解并实现四层记忆架构
- [ ] 能用 OpenAI Embeddings API 计算向量
- [ ] 实现余弦相似度搜索
- [ ] 完成基础 RAG 问答，回答带来源标注
- [ ] 文档分块策略：Fixed-size / Sentence / Recursive

---

### 第 4 月：向量数据库 + 高级 RAG

**月度目标**：掌握主流向量数据库，实现生产级 RAG 系统

#### 向量数据库对比

| 数据库 | 托管/自托管 | 特点 | 适用场景 |
|-------|-----------|------|---------|
| Pinecone | 托管 | 最易上手，管理界面好 | 快速原型 |
| Weaviate | 两者皆有 | 支持混合搜索，图谱 | 复杂场景 |
| Qdrant | 两者皆有 | 高性能，Rust 实现 | 生产部署 |
| Chroma | 自托管 | 开发友好，Python 优先 | 本地开发 |
| pgvector | PostgreSQL 扩展 | 已有 PG 时首选 | 集成现有系统 |

#### 高级 RAG 技术

```typescript
// 1. 语义分块（比固定大小分块更智能）
// 2. 父子分块（检索小块，返回大块上下文）
// 3. Hybrid Search（向量 + 关键词 BM25）
// 4. Reranking（用 Cohere Reranker 重排序）
// 5. HyDE（假设文档嵌入）

// HyDE: 先生成假设答案，再用假设答案去检索
async function hydeSearch(question: string, rag: SimpleRAG): Promise<string> {
  // 先生成一个假设的答案
  const hypotheticalAnswer = await chat(
    `假设你已经知道答案，请写一个回答以下问题的段落（可以不准确，只是假设）：
    问题：${question}`
  );

  // 用假设答案的向量去检索（比用问题直接检索效果更好）
  const contexts = await rag.search(hypotheticalAnswer);

  const prompt = `根据以下背景资料准确回答问题：
  ${contexts.join('\n\n')}
  问题：${question}`;

  return await chat(prompt);
}
```

**月度里程碑检查**

- [ ] 成功接入 Pinecone 或 Qdrant
- [ ] 实现 Hybrid Search（向量 + 关键词）
- [ ] 实现 Reranking 提升检索精度
- [ ] RAG 评测：Context Recall、Answer Faithfulness 指标
- [ ] 完成 DeepLearning.AI 向量数据库课程

---

### 第 5 月：LangChain 深度

**月度目标**：系统掌握 LangChain，理解 LCEL（LangChain Expression Language）

#### LangChain 核心概念对照

| LangChain 概念 | 类比前端 | 说明 |
|--------------|---------|------|
| Chain | Promise.then() 链 | 顺序处理管道 |
| LCEL (pipe) | RxJS pipe | 函数式组合 |
| Runnable | Array.prototype | 统一接口 |
| Memory | React Context | 跨组件状态 |
| OutputParser | zod.parse() | 输出解析验证 |
| PromptTemplate | Template Literal | 动态 Prompt 生成 |

```python
# Python 版 LangChain（月5开始引入Python）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LCEL 管道（就像 RxJS 的 pipe）
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题：\n{context}\n\n问题：{question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 调用
answer = chain.invoke("什么是 RAG？")
```

**月度里程碑检查**

- [ ] 理解并使用 LCEL 构建 Chain
- [ ] 实现 ConversationBufferMemory 和 ConversationSummaryMemory
- [ ] 用 LangChain 重构前 4 月的 RAG 系统
- [ ] 完成 DeepLearning.AI LangChain 全系列课程

---

### 第 6 月：LangGraph + 状态机

**月度目标**：用 LangGraph 构建复杂的有状态 Agent 工作流

#### LangGraph 核心概念

LangGraph = 有向图 + 状态管理，是专门为 Agent 设计的状态机框架

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Agent 状态（类比 Redux Store）
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str
    research_results: list[str]

# 定义节点（类比 Redux Reducer）
def research_node(state: AgentState) -> AgentState:
    # 执行研究
    ...
    return {"research_results": [...]}

def write_node(state: AgentState) -> AgentState:
    # 根据研究结果写作
    ...

def should_continue(state: AgentState) -> str:
    """路由函数（类比 React Router）"""
    if state["next_action"] == "research":
        return "research"
    elif state["next_action"] == "write":
        return "write"
    return END

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_conditional_edges("start", should_continue)
workflow.set_entry_point("start")

app = workflow.compile()
```

**月度综合项目：研究报告生成 Agent**

使用 LangGraph 构建：
1. 搜索节点：搜索多个来源
2. 评估节点：评估信息是否足够
3. 写作节点：生成报告
4. 审核节点：检查报告质量
5. 条件路由：信息不足时回到搜索

**月度里程碑检查**

- [ ] 理解 StateGraph 的工作原理
- [ ] 能实现条件路由（条件边）
- [ ] 实现 Human-in-the-loop（暂停等待人工输入）
- [ ] 实现 Agent 执行过程的可视化
- [ ] 完成 DeepLearning.AI LangGraph 课程

---

### 第 7 月：Multi-Agent（AutoGen / CrewAI）

**月度目标**：理解多 Agent 协作范式，用 AutoGen 和 CrewAI 构建 Agent 团队

#### Multi-Agent 模式对比

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| Supervisor 模式 | 一个管理者 Agent 分配任务给工人 Agent | 需要协调的复杂任务 |
| Peer-to-peer 模式 | Agent 之间直接通信 | 需要协商的场景 |
| Hierarchical 模式 | 多层级 Agent 树 | 超大规模任务分解 |
| Blackboard 模式 | 所有 Agent 共享一个黑板（状态） | 知识共享型任务 |

```python
# CrewAI 示例：技术写作团队
from crewai import Agent, Task, Crew

researcher = Agent(
    role='技术研究员',
    goal='深度研究技术主题，找到最新最准确的信息',
    backstory='你是一位严谨的技术研究员，擅长从各种来源收集和验证信息',
    tools=[search_tool, web_scraper_tool],
)

writer = Agent(
    role='技术写作专家',
    goal='将复杂技术内容转化为清晰易懂的文章',
    backstory='你是一位经验丰富的技术作家，善于向非专业读者解释复杂概念',
)

editor = Agent(
    role='内容编辑',
    goal='确保内容准确、清晰、有价值',
    backstory='你是挑剔的编辑，会仔细核查事实并优化文章结构',
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "LangGraph vs AutoGen 对比分析"})
```

**月度里程碑检查**

- [ ] 完成 DeepLearning.AI AutoGen 课程
- [ ] 完成 DeepLearning.AI CrewAI 课程
- [ ] 实现一个 3-Agent 协作系统（研究→写作→审核）
- [ ] 理解 Agent 间通信协议设计
- [ ] 处理 Agent 死锁和无限循环问题

---

### 第 8 月：MCP 协议 + 上下文工程 + 生产化

**月度目标**：掌握 MCP 协议，理解上下文工程，将 Agent 服务化

#### MCP（Model Context Protocol）

MCP = Anthropic 主导的开放协议，让 LLM 能安全地访问外部工具和数据源

```typescript
// MCP Server 实现（你的 TS 背景在这里大放异彩）
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server(
  { name: 'my-mcp-server', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

// 注册工具
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: 'query_database',
    description: '查询公司数据库',
    inputSchema: {
      type: 'object',
      properties: {
        sql: { type: 'string', description: 'SQL 查询语句' }
      },
      required: ['sql']
    }
  }]
}));

// 处理工具调用
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === 'query_database') {
    const { sql } = request.params.arguments as { sql: string };
    const result = await db.query(sql);
    return { content: [{ type: 'text', text: JSON.stringify(result) }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

#### 上下文工程（Context Engineering）

上下文工程 = 精心设计输入给 LLM 的所有内容（不仅是 prompt）

**上下文窗口管理策略**

```
[System Prompt]     ← 角色定义、规则、工具描述
[Long-term Memory]  ← 从向量DB检索的相关记忆
[Short-term Memory] ← 最近 N 轮对话
[Retrieved Context] ← RAG 检索结果
[Working Memory]    ← 当前任务状态
[User Message]      ← 用户最新输入
```

**生产化 Agent 服务**

```typescript
// Express + Agent 服务化
import express from 'express';
import { z } from 'zod';

const app = express();

const AgentRequestSchema = z.object({
  sessionId: z.string(),
  message: z.string(),
  userId: z.string(),
});

app.post('/api/agent/chat', async (req, res) => {
  const { sessionId, message, userId } = AgentRequestSchema.parse(req.body);

  // 1. 加载会话历史
  const history = await sessionStore.get(sessionId);
  // 2. 运行 Agent
  const result = await reactAgent({ history, message, userId });
  // 3. 保存历史
  await sessionStore.set(sessionId, result.updatedHistory);
  // 4. 返回结果（流式或一次性）
  res.json({ answer: result.answer, steps: result.steps });
});
```

**月度里程碑检查**

- [ ] 实现至少 2 个 MCP Server（如：数据库查询、文件操作）
- [ ] 理解上下文窗口 4 区域分配策略
- [ ] Agent 服务化：REST API + 会话管理
- [ ] 实现 Rate Limiting 和 Token 配额管理
- [ ] 并发 Agent 请求处理（队列 + 并发控制）

---

### 第 9 月：可观测性 + 评测 + 毕业项目

**月度目标**：建立 Agent 系统的可观测性，掌握评测方法论，完成毕业项目

#### 可观测性三支柱

| 支柱 | 工具 | 收集内容 |
|------|------|---------|
| Logs（日志） | LangSmith, Langfuse | 每次 LLM 调用的输入/输出/Token 用量 |
| Traces（追踪） | LangSmith, Phoenix | Agent 完整执行链路，每步耗时 |
| Metrics（指标） | Prometheus + Grafana | 响应时间、成功率、Token 成本 |

```python
# LangSmith 集成（一行代码接入 Tracing）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# 之后所有 LangChain 调用自动被追踪
```

#### LLM 评测体系

```python
# 评测四维度
EVAL_DIMENSIONS = {
    "faithfulness": "回答是否忠实于检索到的文档",
    "answer_relevancy": "回答是否相关于问题",
    "context_recall": "检索到的文档是否包含回答所需信息",
    "context_precision": "检索结果的精确度",
}

# LLM-as-Judge（用 LLM 评估 LLM 的输出）
async def evaluate_answer(question: str, answer: str, ground_truth: str) -> float:
    prompt = f"""评估以下回答的质量（0-10分）：
    
问题：{question}
标准答案：{ground_truth}
模型回答：{answer}

评分标准：
- 10分：完全正确，逻辑清晰
- 7-9分：基本正确，有小瑕疵
- 4-6分：部分正确
- 0-3分：基本错误

只返回数字评分，不要解释。"""
    
    score_str = await chat(prompt)
    return float(score_str.strip())
```

**毕业项目：企业级 RAG + Agent 系统**

要求：
1. 支持多格式文档（PDF/Word/CSV）
2. 向量数据库：Qdrant
3. Hybrid Search + Reranking
4. LangGraph 多步推理 Agent
5. MCP 协议扩展工具
6. LangSmith 全链路追踪
7. 评测集 + 自动化评测
8. REST API 服务化
9. Docker 部署
10. 基础 Web UI（React + TypeScript，你的强项！）

**月度里程碑检查**

- [ ] LangSmith Traces 接入，能看到完整执行链路
- [ ] 建立评测集（50条问答对），自动化评测跑通
- [ ] 毕业项目全部功能完成并部署
- [ ] 写一篇技术博客：你从前端到 AI 工程的转型历程
- [ ] 完成 DeepLearning.AI 评测相关课程

---

## 7. 阶段里程碑检查表

### 第一阶段（月1-3）：基础能力

- [ ] 独立调用 3 个以上 LLM API（OpenAI / Anthropic / Gemini）
- [ ] 能写出 Zero-shot / Few-shot / CoT 各种 Prompt 变体
- [ ] 用 TypeScript 实现完整的多轮对话系统
- [ ] 实现 ReAct Agent，能使用 5 个以上工具
- [ ] 实现基础 RAG，回答准确率 > 70%
- [ ] 理解 Token 计数，能设计上下文管理策略

### 第二阶段（月4-6）：工具能力

- [ ] 接入向量数据库（Pinecone 或 Qdrant）
- [ ] 实现 Hybrid Search
- [ ] 用 LangChain 重构已有系统
- [ ] 用 LangGraph 实现有状态的工作流 Agent
- [ ] Human-in-the-loop 流程实现
- [ ] 开始用 Python 开发

### 第三阶段（月7-9）：系统能力

- [ ] Multi-Agent 系统：3 Agent 协作完成复杂任务
- [ ] MCP Server 实现并接入 Claude Desktop
- [ ] 上下文工程：优化 Token 利用效率 > 20%
- [ ] 全链路可观测性（LangSmith / Langfuse）
- [ ] 自动化评测流水线
- [ ] 毕业项目上线部署

---

## 8. 毕业项目建议

### 方向一：企业知识助手

技术点：RAG + LangGraph + 多格式文档 + 权限控制
商业价值：内部知识库问答、文档检索、新员工培训

### 方向二：代码审查 Agent

技术点：ReAct + 代码分析工具 + GitHub API
商业价值：代码质量自动化，你最懂前端代码！

### 方向三：数据分析 Agent

技术点：Text-to-SQL + 可视化 + Multi-Agent
商业价值：业务人员自助数据分析

### 方向四：客服自动化系统

技术点：意图识别 + RAG + 人工兜底 + 多轮对话
商业价值：降低客服成本，24小时服务

---

> 学习是一场马拉松，不是短跑。
> 每天进步 1%，9 个月后你就是一名合格的 AI 工程师。
> 你的前端背景是宝贵的优势——很多 AI 工程师不懂工程化，而你懂。

---

*文档版本：v1.0 | 创建日期：2026-07-03*
