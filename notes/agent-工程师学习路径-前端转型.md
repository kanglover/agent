# 前端开发转 Agent 工程师 —— 完整学习路径

> 沉淀时间：2026-07-01  
> 适用对象：有前端开发基础（JS/TS/React），零 AI 工程经验，目标转型 Agent 开发岗位

---

## 一句话总结

前端转 Agent 工程师，你最大的优势是**已经懂 TypeScript + API 调用**，缺的是 LLM 原理知识 + Python 后端经验 + Agent 系统设计能力，分阶段补齐即可。

---

## 岗位核心要求拆解（来自 4 份 JD 分析）

| 能力模块 | 出现频率 | 你的现状 |
|---|---|---|
| Prompt 工程 | ⭐⭐⭐⭐⭐ | 待学 |
| Tool Use / Function Calling | ⭐⭐⭐⭐⭐ | 待学（但懂 API，上手快）|
| RAG（检索增强生成）| ⭐⭐⭐⭐⭐ | 待学 |
| ReAct / Agent 框架 | ⭐⭐⭐⭐⭐ | 待学 |
| 上下文管理 / Token Budget | ⭐⭐⭐⭐ | 待学 |
| Multi-Agent 多智能体 | ⭐⭐⭐⭐ | 待学 |
| Python 后端 | ⭐⭐⭐⭐ | 待学 |
| MCP 协议 | ⭐⭐⭐ | 待学 |
| LangChain / AutoGen 框架 | ⭐⭐⭐ | 待学 |
| LLM 原理（Transformer）| ⭐⭐⭐ | 待学 |
| 评测体系 / 可观测性 | ⭐⭐⭐ | 待学 |

---

## 学习路径总览（预计 6-9 个月）

```
第 0 阶段：准备期（2 周）
      ↓
第 1 阶段：LLM 基础 + Prompt 工程（4-6 周）
      ↓
第 2 阶段：Agent 核心技术（6-8 周）
      ↓
第 3 阶段：Agent 框架 + RAG（6-8 周）
      ↓
第 4 阶段：Multi-Agent + 工程化（6-8 周）
      ↓
第 5 阶段：项目实战 + 求职准备（持续）
```

---

## 第 0 阶段：准备期（2 周）

> 目标：搭好环境，用最快速度"看到 AI 在工作"

### 你要做的事

**1. 注册必要账号**
- [Anthropic Console](https://console.anthropic.com) — 获取 Claude API Key
- [OpenAI Platform](https://platform.openai.com) — 获取 GPT API Key（两个都要，方便对比）
- [GitHub](https://github.com) — 代码托管

**2. 安装 Python（你已经有 Node.js，Python 也要装）**
```bash
# Mac 推荐用 pyenv 管理版本
brew install pyenv
pyenv install 3.11
pyenv global 3.11

# 验证
python --version  # 应该显示 3.11.x
pip --version
```

**3. 安装 Claude Code CLI（你将大量使用）**
```bash
npm install -g @anthropic-ai/claude-code
```

**4. 跑通第一个 API 调用**
```python
import anthropic

client = anthropic.Anthropic(api_key="你的API_KEY")

message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
)

print(message.content[0].text)
```

### 里程碑检查
- [ ] API 调用成功，看到 AI 回复
- [ ] 知道什么是 Token（费用单位，类似"字符数"）

---

## 第 1 阶段：LLM 基础 + Prompt 工程（4-6 周）

> 目标：理解 AI 是怎么工作的，会写高质量 Prompt

### 1.1 必须理解的 LLM 核心概念

**概念清单（每个都要能用大白话解释）：**

| 概念 | 简单理解 |
|---|---|
| Token | AI 处理文字的最小单位，1 个 token ≈ 0.75 个英文词 ≈ 0.5 个汉字 |
| Context Window | AI 的"工作记忆"，一次能看多少文字（如 200K tokens ≈ 15 万汉字）|
| Temperature | 创造力旋钮，0=严谨死板，1=天马行空 |
| System Prompt | 给 AI 下的"基础指令"，相当于岗位说明书 |
| Prompt | 你发给 AI 的问题/指令 |
| Completion | AI 的回答 |
| Fine-tuning | 用你自己的数据重新"训练"AI |
| Embedding | 把文字变成数字向量，让 AI 能做"相似度搜索" |

**推荐学习资源：**
- 🎥 [3Blue1Brown: Neural Networks 系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — 10 小时，最好的 AI 原理科普，不需要数学基础
- 📖 [Transformer 论文原文](https://arxiv.org/abs/1706.03762) — 先不用看懂，感受一下
- 🎥 [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 跟着写一个微型 GPT，真正理解原理

### 1.2 Prompt 工程（最高投资回报的技能）

**核心技巧：**

```
1. 角色设定（Role）
   "你是一名资深 Python 工程师，擅长 API 设计..."
   
2. 任务描述（Task）
   "帮我 review 这段代码，重点检查错误处理是否完善"
   
3. 输出格式（Format）
   "用 JSON 格式输出，包含 issue、severity、suggestion 三个字段"
   
4. 示例（Few-shot）
   "参考以下例子：[示例1] → [期望输出1]"
   
5. 约束条件（Constraints）
   "不要修改函数签名，只修改函数体内部"
```

**实战练习（每天 30 分钟）：**
- 用 Claude 做代码 Review，观察不同 Prompt 的输出差异
- 尝试让 AI 生成结构化 JSON 数据
- 练习写 System Prompt，让 AI 扮演不同角色

**推荐课程：**
- 📚 [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)（官方，必读）
- 🎓 [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)（免费，2小时）

### 里程碑检查
- [ ] 能解释 Token、Context Window、Temperature 是什么
- [ ] 写过至少 10 个不同风格的 Prompt，观察差异
- [ ] 能让 AI 稳定输出结构化 JSON

---

## 第 2 阶段：Agent 核心技术（6-8 周）

> 目标：理解 Agent 的本质，写出第一个能"自主完成任务"的 AI

### 2.1 什么是 Agent？

> **类比**：普通 AI 对话 = 问路人（只会回答问题）  
> Agent = 司机（能接收目的地，自己规划路线，开车，遇堵绕行，最终把你送到）

Agent 的本质是一个**循环**：
```
观察环境 → 思考下一步 → 执行动作 → 观察结果 → 思考下一步 → ...
直到任务完成
```

### 2.2 ReAct 框架（最重要的 Agent 模式）

ReAct = **Re**asoning + **Act**ing（思考 + 行动）

```
用户："帮我查一下明天上海的天气，如果下雨就帮我加一条提醒"

AI 思考（Thought）：我需要查天气，我有查天气工具
AI 行动（Action）：调用 get_weather(city="上海", date="明天")
观察结果（Observation）：{"weather": "雨", "temp": "18-24°C"}
AI 思考：天气是雨，需要添加提醒
AI 行动（Action）：调用 add_reminder(text="明天带伞", time="明天早上8点")
观察结果（Observation）：{"status": "success"}
AI 输出（Final Answer）："已查到明天上海有雨，已为你添加带伞提醒"
```

### 2.3 Tool Use / Function Calling（Agent 的"手"）

Tool Use 让 AI 能调用外部函数，是 Agent 能力的基础。

**最简单的 Tool Use 示例：**
```python
import anthropic

client = anthropic.Anthropic()

# 定义工具（告诉 AI 有什么工具可以用）
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：上海、北京"
                }
            },
            "required": ["city"]
        }
    }
]

# 第一轮：发送问题，AI 决定用哪个工具
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "上海明天天气怎么样？"}]
)

# AI 返回工具调用请求
if response.stop_reason == "tool_use":
    tool_use = response.content[1]
    print(f"AI 想调用：{tool_use.name}")      # get_weather
    print(f"参数：{tool_use.input}")           # {"city": "上海"}
    
    # 你在这里真正调用天气 API
    weather_result = {"weather": "晴", "temp": "25°C"}
    
    # 第二轮：把工具结果告诉 AI，让它给出最终回答
    final_response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "上海明天天气怎么样？"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(weather_result)
                }]
            }
        ]
    )
    print(final_response.content[0].text)
```

### 2.4 Memory（记忆系统）

Agent 的记忆分四层：

| 记忆类型 | 类比 | 技术实现 |
|---|---|---|
| 短期记忆 | 对话中的临时记忆 | Context Window（messages 列表）|
| 工作记忆 | 桌面上的便利贴 | 变量、运行时状态 |
| 长期记忆 | 笔记本 | 数据库 + Embedding 检索 |
| 外部记忆 | 图书馆 | RAG 系统（下一阶段讲）|

**实现简单的对话记忆：**
```python
# 核心就是维护一个 messages 列表
conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=conversation_history  # 每次都传完整历史
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message
```

### 2.5 Context 管理（Token Budget）

Context Window 是有限的，当对话太长时需要压缩：

```
策略一：滑动窗口 — 只保留最近 N 条消息
策略二：摘要压缩 — 让 AI 把旧消息总结成摘要
策略三：重要性筛选 — 标记重要消息，只压缩不重要的
```

**推荐学习资源：**
- 📚 [Anthropic: Tool Use 官方文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)（必读）
- 🎓 [DeepLearning.AI: AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)（免费，推荐）
- 📖 本项目 `code/min_agent/` — 已经有一个最小 Agent 实现，认真读懂它

### 里程碑检查
- [ ] 独立写出一个 ReAct Agent（能循环调用工具）
- [ ] 实现带记忆的多轮对话
- [ ] 理解 Token Budget，写出对话压缩逻辑

---

## 第 3 阶段：Agent 框架 + RAG（6-8 周）

> 目标：用主流框架提速开发，掌握知识库检索核心技术

### 3.1 主流 Agent 框架对比

| 框架 | 语言 | 适合场景 | 学习优先级 |
|---|---|---|---|
| LangChain | Python | 通用，生态最大 | ⭐⭐⭐⭐⭐ 必学 |
| LangGraph | Python | 复杂工作流，State Machine | ⭐⭐⭐⭐⭐ 必学 |
| AutoGen | Python | 多智能体对话 | ⭐⭐⭐⭐ 重要 |
| CrewAI | Python | 角色扮演式多 Agent | ⭐⭐⭐ 了解 |
| Vercel AI SDK | TypeScript | 前端友好，你的优势 | ⭐⭐⭐⭐ 推荐 |

**学习顺序建议：**
1. 先用原生 API 写会（第 2 阶段做的事）
2. 再学 LangChain，理解框架做了什么抽象
3. 然后学 LangGraph，理解复杂流程控制
4. 最后看 AutoGen，理解多 Agent 协作

**LangChain 快速上手：**
```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天晴，25°C"

# 2. 创建 Agent
llm = ChatAnthropic(model="claude-opus-4-8")
agent = create_react_agent(llm, [get_weather], prompt=...)
executor = AgentExecutor(agent=agent, tools=[get_weather])

# 3. 运行
result = executor.invoke({"input": "上海今天天气怎么样？"})
print(result["output"])
```

### 3.2 RAG（检索增强生成）— 给 AI 一个"知识库"

**RAG 解决什么问题？**
AI 只知道训练时学过的东西，不知道你公司内部文档、最新资讯、私有数据。  
RAG = 让 AI 先去"图书馆"查相关资料，再回答你。

**RAG 完整流程：**
```
建库阶段（离线）：
文档 → 分段（Chunking）→ 向量化（Embedding）→ 存入向量数据库

检索阶段（在线）：
用户问题 → 向量化 → 在数据库中找最相似的段落 → 拼入 Prompt → AI 回答
```

**最简单的 RAG 实现：**
```python
import anthropic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = anthropic.Anthropic()

# 1. 你的知识库（实际项目中这些来自文档）
documents = [
    "Claude 是 Anthropic 开发的 AI 助手，擅长分析和写作",
    "Claude Code 是面向开发者的 AI 编程工具",
    "MCP 是 Model Context Protocol，让 AI 能连接外部工具"
]

# 2. 把文档转成向量（Embedding）
def embed(text):
    response = client.embeddings.create(
        model="voyage-3",
        input=[text]
    )
    return response.embeddings[0]

doc_embeddings = [embed(doc) for doc in documents]

# 3. 检索最相关的文档
def retrieve(query, top_k=2):
    query_embedding = embed(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# 4. 用检索结果回答问题
def rag_answer(question):
    relevant_docs = retrieve(question)
    context = "\n".join(relevant_docs)
    
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""根据以下资料回答问题：

资料：
{context}

问题：{question}"""
        }]
    )
    return response.content[0].text

print(rag_answer("Claude Code 是什么？"))
```

### 3.3 MCP 协议（Model Context Protocol）

**MCP 是什么？**  
类比：USB 接口标准。以前每个设备都有自己的接口，很混乱。USB 统一了接口标准。  
MCP 是让 AI 连接各种工具的**统一标准协议**，你写一次工具，所有支持 MCP 的 AI 都能用。

**学习资源：**
- 📚 [MCP 官方文档](https://modelcontextprotocol.io/docs)
- 🎓 [Anthropic: MCP Quickstart](https://docs.anthropic.com/en/docs/build-with-claude/mcp)

**推荐学习资源：**
- 🎓 [DeepLearning.AI: Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)（免费）
- 🎓 [DeepLearning.AI: LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)（免费）
- 📚 [LangChain 官方文档](https://python.langchain.com/docs/introduction)

### 里程碑检查
- [ ] 用 LangChain 重写第 2 阶段的 Agent
- [ ] 实现一个完整 RAG：上传自己的文档，让 AI 基于文档回答
- [ ] 理解 MCP 协议，调通一个 MCP Server

---

## 第 4 阶段：Multi-Agent + 工程化（6-8 周）

> 目标：设计多智能体系统，掌握生产级工程实践

### 4.1 Multi-Agent（多智能体）

**为什么需要多个 Agent？**  
一个人什么都做 vs 一个团队分工合作 — 复杂任务需要分工。

**常见 Multi-Agent 模式：**

```
1. 流水线模式（Pipeline）
   用户 → Agent A（收集信息）→ Agent B（分析）→ Agent C（生成报告）→ 输出

2. 主从模式（Orchestrator + Workers）
   用户 → 主 Agent（分析任务，分配工作）
              ├── Worker A（搜索网络）
              ├── Worker B（查询数据库）
              └── Worker C（生成代码）
   主 Agent 汇总 → 输出

3. 辩论模式（Debate）
   同一问题 → Agent A（正方）+ Agent B（反方）→ 裁判 Agent → 最优答案
```

**用 AutoGen 实现多 Agent 对话：**
```python
import autogen

# 创建两个 Agent
user_proxy = autogen.UserProxyAgent(
    name="用户代理",
    human_input_mode="NEVER"
)

assistant = autogen.AssistantAgent(
    name="AI助手",
    llm_config={"model": "claude-opus-4-8"}
)

code_reviewer = autogen.AssistantAgent(
    name="代码审查员",
    system_message="你是一名严格的代码审查员，找出所有潜在问题",
    llm_config={"model": "claude-opus-4-8"}
)

# 发起群聊
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, code_reviewer],
    messages=[],
    max_round=10
)
```

### 4.2 Harness Engineering（工程基础设施）

这是 JD 里高频出现的词，指 Agent 系统的"工程底座"：

| 组件 | 作用 | 关键点 |
|---|---|---|
| 上下文管理 | 控制传给 AI 的信息量 | 不能太少（AI 不知道背景）不能太多（Token 超限）|
| Token Budget | 控制 API 费用和速度 | 动态分配，重要对话多给 Token |
| 错误处理 | AI 调用失败时的兜底 | 重试、降级、回退策略 |
| 可观测性 | 知道 Agent 在干什么 | 日志、追踪、监控 |
| 评测体系 | 衡量 Agent 好不好 | 任务完成率、工具调用准确率、延迟 |

### 4.3 Python 后端技能补充（前端转型必学）

```python
# 你需要掌握这些 Python 知识：

# 1. async/await（和 JS 很像！）
import asyncio

async def call_ai(prompt):
    # 类似 JS 的 await fetch(...)
    response = await client.messages.create(...)
    return response

# 2. FastAPI（Python 版 Express）
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    response = await call_ai(message)
    return {"reply": response}

# 3. 环境变量管理
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
```

**推荐学习资源：**
- 🎓 [DeepLearning.AI: Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)（免费）
- 🎓 [DeepLearning.AI: AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)（免费）
- 📚 [FastAPI 官方教程](https://fastapi.tiangolo.com/tutorial/)（Python 后端必学）

### 里程碑检查
- [ ] 用 AutoGen 或 LangGraph 实现主从式多 Agent 系统
- [ ] 给 Agent 加上日志和错误处理
- [ ] 设计并实现一个简单的评测指标

---

## 第 5 阶段：项目实战（持续）

> 目标：用真实项目证明你的能力，建立作品集

### 推荐项目（按复杂度排序）

**初级项目（第 2 阶段后可做）**
- [ ] **代码 Review Bot**：接收 Git Diff，AI 自动做 Code Review
- [ ] **个人知识库 AI**：把自己的笔记上传，让 AI 帮你查资料
- [ ] **TODO 管理 Agent**：自然语言管理任务，能自动拆解、提醒

**中级项目（第 3 阶段后可做）**
- [ ] **RAG 问答系统**：上传公司文档，搭建企业内部知识库
- [ ] **AI 数据分析师**：给数据文件，AI 自动分析并出报告
- [ ] **多工具调度 Agent**：能用计算器、查天气、搜网络、操作文件

**高级项目（第 4 阶段后可做）**
- [ ] **Auto-debugging Agent**：给代码和错误信息，AI 自动找原因改代码
- [ ] **Multi-Agent 研究助手**：多个 Agent 协作完成复杂调研报告
- [ ] **Claude Code 插件**：开发 MCP Server，给 Claude Code 添加自定义能力

---

## 精选课程总表

### 免费课程（DeepLearning.AI — 全部推荐！）
| 课程名 | 时长 | 学习阶段 |
|---|---|---|
| [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | 2h | 第 1 阶段 |
| [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) | 3h | 第 3 阶段 |
| [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | 3h | 第 3 阶段 |
| [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | 4h | 第 2-3 阶段 |
| [Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) | 3h | 第 4 阶段 |
| [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) | 3h | 第 4 阶段 |

### 视频教程（YouTube）
| 频道/视频 | 内容 | 学习阶段 |
|---|---|---|
| [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | LLM 原理最佳入门 | 第 1 阶段 |
| [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 从零写 GPT，深度理解 | 第 1 阶段 |
| [AI Jason](https://www.youtube.com/@AIJasonZ) | Agent 实战项目 | 第 2-3 阶段 |

### 必读文档
| 文档 | 内容 | 重要性 |
|---|---|---|
| [Anthropic Docs](https://docs.anthropic.com) | Claude API 完整文档 | ⭐⭐⭐⭐⭐ |
| [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | 官方 Prompt 指南 | ⭐⭐⭐⭐⭐ |
| [LangChain Docs](https://python.langchain.com/docs/introduction) | LangChain 官方文档 | ⭐⭐⭐⭐ |
| [MCP Protocol](https://modelcontextprotocol.io/docs) | MCP 协议文档 | ⭐⭐⭐⭐ |

### 书籍
| 书名 | 适合阶段 |
|---|---|
| 《AI Engineering》by Chip Huyen | 第 3-4 阶段（英文，2024年出版）|
| 《Building LLM Powered Applications》 | 第 2-3 阶段 |

---

## 你的前端优势，如何快速迁移

| 前端技能 | 对应 Agent 技能 | 迁移难度 |
|---|---|---|
| TypeScript | 用 Vercel AI SDK 直接写 Agent | 🟢 非常简单 |
| React + 状态管理 | 理解 Agent 的状态机思维 | 🟢 概念相通 |
| API 调用（fetch）| 调用 LLM API | 🟢 几乎一样 |
| async/await | Python 也有 async/await | 🟢 直接复用 |
| 组件化思维 | 工具函数模块化 | 🟡 思路相通，换语言 |
| Node.js 后端 | Python FastAPI | 🟡 需要学 Python |
| 无 | LLM 原理 | 🔴 需要从头学 |
| 无 | 向量数据库、RAG | 🔴 需要从头学 |

**建议：一开始用 TypeScript + Vercel AI SDK 写 Agent，能快速看到效果，建立信心，再逐渐补 Python。**

---

## 每周学习安排（参考）

```
周一：看视频/读文档（1-2 小时）
周二：写代码练习（1-2 小时）
周三：做小项目（2 小时）
周四：复习 + 整理笔记（1 小时）
周五：挑战新东西 / 看开源项目代码（1-2 小时）
周末：做一个完整的小项目（4-6 小时）
```

---

## 面试高频考题（提前准备）

1. **什么是 ReAct Agent？它和普通对话有什么区别？**
2. **Tool Use 的完整流程是什么？**
3. **RAG 和 Fine-tuning 分别适合什么场景？**
4. **Context Window 满了怎么办？有哪些处理策略？**
5. **如何评估一个 Agent 的质量？**
6. **Multi-Agent 系统中如何处理 Agent 之间的冲突？**
7. **Token Budget 如何优化？**
8. **MCP 协议解决什么问题？**

---

## 进度检查表（打印出来贴墙上）

- [ ] **第 0 阶段**：跑通第一个 API 调用
- [ ] **第 1 阶段**：写过 10+ Prompt，AI 能稳定输出结构化 JSON
- [ ] **第 2 阶段**：独立写出 ReAct Agent，有工具调用 + 记忆
- [ ] **第 3 阶段**：实现完整 RAG，用 LangChain 重写 Agent
- [ ] **第 4 阶段**：Multi-Agent 系统可以运行，有日志和错误处理
- [ ] **项目**：GitHub 上有 2-3 个 Agent 项目，有 README 和演示
- [ ] **求职**：能流畅讲清楚 8 道面试题

---

*相关概念：agent、llm、prompt、rag、langchain、tool-use、multi-agent、context-engineering、前端转型*
