# LangChain & LangGraph 架构解析

> 面向前端转 Agent 开发工程师的深度技术文档
> 日期：2026-07-03

---

## 目录

1. [LangChain 核心架构](#1-langchain-核心架构)
2. [LangChain 工具、记忆与向量存储](#2-langchain-工具记忆与向量存储)
3. [LangGraph 核心概念](#3-langgraph-核心概念)
4. [LangGraph vs LangChain](#4-langgraph-vs-langchain)
5. [LangGraph 实现 ReAct Agent](#5-langgraph-实现-react-agent)
6. [LangGraph 实现 Multi-Agent](#6-langgraph-实现-multi-agent)
7. [LangGraph 状态管理与持久化](#7-langgraph-状态管理与持久化)
8. [框架横向对比：AutoGen、CrewAI](#8-框架横向对比autogencrewai)
9. [架构图与流程图汇总](#9-架构图与流程图汇总)
10. [关键源码分析](#10-关键源码分析)

---

## 1. LangChain 核心架构

### 1.1 LangChain 的设计哲学

LangChain 诞生于 2022 年底，是 LLM 应用开发领域最早的综合性框架。它的核心理念是：**把 LLM 应用中所有可复用的模式抽象成可组合的积木**。

```
┌────────────────────────────────────────────────────────────────┐
│                    LangChain 生态系统                           │
│                                                                │
│  langchain-core     ← 核心抽象（Runnable、BaseMessage 等）      │
│  langchain          ← 高级组件（Agent、Chain、Memory 等）        │
│  langchain-community← 第三方集成（数百种 LLM、工具、存储）       │
│  langgraph          ← 有状态的图式 Agent 编排（独立包）          │
│  langsmith          ← 观测与评估平台（SaaS）                    │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Runnable 协议——LangChain 的统一接口

`Runnable` 是 LangChain 中最重要的抽象。任何组件，只要实现了这个接口，就可以被链接和组合：

```python
from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncIterator

class Runnable(ABC):
    """LangChain 中一切组件的基类"""
    
    @abstractmethod
    def invoke(self, input: Any, config=None) -> Any:
        """同步执行"""
        pass
    
    async def ainvoke(self, input: Any, config=None) -> Any:
        """异步执行（默认在线程池中运行 invoke）"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.invoke, input, config
        )
    
    def stream(self, input: Any, config=None) -> Iterator[Any]:
        """流式输出（默认返回单个结果）"""
        yield self.invoke(input, config)
    
    async def astream(self, input: Any, config=None) -> AsyncIterator[Any]:
        """异步流式输出"""
        yield await self.ainvoke(input, config)
    
    def batch(self, inputs: list, config=None) -> list:
        """批量执行（支持并行）"""
        return [self.invoke(i, config) for i in inputs]
    
    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """支持 | 操作符组合"""
        return RunnableSequence(first=self, last=other)
```

**关键点**：`|` 操作符让组合变得极其优雅，这是 LCEL 的基础。

### 1.3 LCEL（LangChain Expression Language）

LCEL 是 LangChain 的声明式组合语言，用 `|` 把组件串联成管道：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义各个组件
prompt = ChatPromptTemplate.from_template(
    "用一句话解释什么是{concept}，要用大白话"
)
model = ChatAnthropic(model="claude-opus-4-8")
parser = StrOutputParser()

# 用 LCEL 组合成 Chain
chain = prompt | model | parser

# 执行
result = chain.invoke({"concept": "向量数据库"})
print(result)
# 输出：向量数据库就是专门用来存储和搜索"意思相近的内容"的数据库，
#       你搜索"猫"，它能找到关于"猫咪"、"喵星人"的内容。
```

LCEL 的内部执行流程：

```
{"concept": "向量数据库"}
        │
        ▼
[ChatPromptTemplate]
        │ 生成 ChatPromptValue
        ▼
[ChatAnthropic]
        │ 发送 API 请求，返回 AIMessage
        ▼
[StrOutputParser]
        │ 提取 AIMessage.content
        ▼
"向量数据库就是..."（纯字符串）
```

### 1.4 Chain 的演进历史

```
LangChain v0.1 时代（2022-2023）：
  LLMChain → SequentialChain → ConversationalChain
  （继承式 API，难以组合）

LangChain v0.2+ 时代（2024+）：
  LCEL（prompt | model | parser）
  （组合式 API，极度灵活）
```

旧式 Chain vs LCEL 对比：

```python
# 旧式（已不推荐）
from langchain.chains import LLMChain
chain = LLMChain(llm=model, prompt=prompt)
result = chain.run("向量数据库")

# LCEL（推荐）
chain = prompt | model | StrOutputParser()
result = chain.invoke({"concept": "向量数据库"})
```

### 1.5 核心组件详解

```python
# 1. PromptTemplate - 提示词模板
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 多角色对话模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}专家，用{style}风格回答问题"),
    MessagesPlaceholder(variable_name="history"),  # 对话历史占位符
    ("human", "{question}"),
])

# 2. Output Parser - 输出解析器
from langchain_core.output_parsers import (
    StrOutputParser,       # 纯文本
    JsonOutputParser,      # JSON
    PydanticOutputParser,  # Pydantic 模型
    CommaSeparatedListOutputParser,  # 逗号分隔列表
)

from pydantic import BaseModel, Field

class BookInfo(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    year: int = Field(description="出版年份")

parser = PydanticOutputParser(pydantic_object=BookInfo)
chain = prompt | model | parser
book = chain.invoke({"book_name": "三体"})
print(book.title, book.author)  # 类型安全的访问

# 3. RunnableParallel - 并行执行
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    summary=prompt_summary | model | StrOutputParser(),
    keywords=prompt_keywords | model | StrOutputParser(),
    sentiment=prompt_sentiment | model | StrOutputParser(),
)
# 三个子链并行执行
result = parallel_chain.invoke({"text": "待分析的文本"})
# result = {"summary": "...", "keywords": "...", "sentiment": "..."}

# 4. RunnableLambda - 包装普通函数
from langchain_core.runnables import RunnableLambda

def preprocess(text: str) -> str:
    return text.strip().lower()

chain = RunnableLambda(preprocess) | prompt | model | parser
```

---

## 2. LangChain 工具、记忆与向量存储

### 2.1 工具系统（Tool System）

```python
from langchain_core.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Optional

# 方式一：装饰器（最简单）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气。
    
    Args:
        city: 城市名称，例如 '北京'、'上海'
    """
    # 实际调用天气 API
    return f"{city}今天晴天，气温 28°C"

# 方式二：继承 BaseTool（最灵活）
class SearchTool(BaseTool):
    name = "web_search"
    description = "搜索互联网获取最新信息"
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """同步执行搜索"""
        results = self._call_search_api(query)
        return self._format_results(results)
    
    async def _arun(self, query: str, **kwargs) -> str:
        """异步执行搜索"""
        results = await self._async_call_search_api(query)
        return self._format_results(results)

# 工具绑定到模型（让模型知道可以调用哪些工具）
model_with_tools = model.bind_tools([get_weather, SearchTool()])

# 执行带工具的对话
response = model_with_tools.invoke("北京今天天气怎么样？")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_xxx'}]
```

### 2.2 记忆系统（Memory System）

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import (
    InMemoryChatMessageHistory,
    RedisChatMessageHistory,
    SQLChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建带历史记忆的 Chain
chain = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
]) | model | StrOutputParser()

# 用 RunnableWithMessageHistory 包装，自动管理历史
store = {}  # 实际应用中用 Redis/DB

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 多轮对话，session_id 用于隔离不同用户
config = {"configurable": {"session_id": "user_123"}}

reply1 = chain_with_history.invoke({"input": "我叫小明"}, config=config)
reply2 = chain_with_history.invoke({"input": "你还记得我叫什么吗？"}, config=config)
# reply2 会正确回答 "小明"
```

### 2.3 向量存储集成（Vector Store Integration）

```python
from langchain_anthropic import AnthropicEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 文档分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块 1000 字符
    chunk_overlap=200,  # 相邻块重叠 200 字符（保证上下文连续性）
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分割优先级
)

docs = splitter.split_documents(raw_documents)

# 生成向量并存入向量库
embeddings = AnthropicEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 相似度检索
retriever = vectorstore.as_retriever(
    search_type="mmr",           # Maximum Marginal Relevance（多样性检索）
    search_kwargs={
        "k": 5,                  # 返回 5 个最相关的文档
        "fetch_k": 20,           # 先取 20 个，再用 MMR 筛选
        "lambda_mult": 0.7,      # 相关性 vs 多样性的平衡（0-1）
    }
)

# RAG Chain（检索增强生成）
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | ChatPromptTemplate.from_template(
        "根据以下上下文回答问题：\n\n{context}\n\n问题：{question}"
    )
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("什么是 Agent Loop？")
```

---

## 3. LangGraph 核心概念

### 3.1 为什么需要 LangGraph

LangChain 的 Chain/LCEL 适合 **线性流程**，但真实的 Agent 往往需要：

- **循环**（Loop）：Agent 需要反复思考-行动直到任务完成
- **条件分支**（Conditional）：根据不同情况走不同路径
- **并行**（Parallel）：多个子任务同时执行
- **持久化**（Persistence）：任务中断后能恢复
- **人工介入**（Human-in-the-loop）：在关键决策点等待人类确认

这些需求都指向 **图（Graph）** 这种数据结构。LangGraph 将 Agent 的执行流程建模为一个 **有状态的有向图**。

### 3.2 核心概念图谱

```
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph 核心概念                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  State（状态）                                        │   │
│  │                                                      │   │
│  │  TypedDict 或 Pydantic 模型，保存整个图的运行状态     │   │
│  │  每个 Node 读取 State，返回 State 的更新部分          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node（节点）                                         │   │
│  │                                                      │   │
│  │  Python 函数：接收 State → 返回 State 的更新          │   │
│  │  类型：普通函数、LLM 调用、工具执行、人工审批...       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Edge（边）                                           │   │
│  │                                                      │   │
│  │  普通边：无条件转到下一节点                            │   │
│  │  条件边：根据 State 内容选择下一节点                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Graph（图）                                          │   │
│  │                                                      │   │
│  │  StateGraph：有状态图（最常用）                       │   │
│  │  MessageGraph：专为对话 Agent 优化的图                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 State 的设计

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 基础 State（用于简单 Agent）
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages 是一个 reducer：新消息追加到列表，而不是替换

# 复杂 State（用于业务 Agent）
class ResearchState(TypedDict):
    # 输入
    user_query: str
    
    # 中间状态
    search_queries: list[str]      # 搜索关键词
    search_results: list[dict]     # 搜索结果
    relevant_docs: list[str]       # 筛选后的相关文档
    
    # 输出
    final_answer: str
    citations: list[str]
    
    # 控制流
    iteration_count: int
    should_continue: bool
    error_message: str
```

`Annotated[list, add_messages]` 的工作原理：

```python
# LangGraph 中的 Reducer 机制
# 没有 Reducer：直接替换（覆盖）
class SimpleState(TypedDict):
    counter: int  # 每次更新都会覆盖

# 有 Reducer：按规则合并
def add_reducer(existing: list, new: list) -> list:
    return existing + new  # 追加而不是替换

class ListState(TypedDict):
    items: Annotated[list, add_reducer]
```

### 3.4 Node 的实现

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

model = ChatAnthropic(model="claude-opus-4-8")

# 节点函数：输入 State → 输出 State 的更新部分（字典）
def call_model(state: AgentState) -> dict:
    """调用 LLM 节点"""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}  # 只返回需要更新的字段

def execute_tools(state: AgentState) -> dict:
    """执行工具节点"""
    last_message = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool = TOOL_MAP[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": tool_results}

# 条件边函数：输入 State → 返回下一节点的名称
def should_continue(state: AgentState) -> str:
    """决定下一步走哪条路"""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"      # 有工具调用 → 执行工具
    else:
        return END          # 没有工具调用 → 结束
```

### 3.5 Graph 的构建与编译

```python
from langgraph.graph import StateGraph, END, START

# 1. 创建图
graph_builder = StateGraph(AgentState)

# 2. 添加节点
graph_builder.add_node("llm", call_model)
graph_builder.add_node("tools", execute_tools)

# 3. 添加边
graph_builder.add_edge(START, "llm")               # 开始 → LLM
graph_builder.add_conditional_edges(
    "llm",                      # 从哪个节点出发
    should_continue,            # 条件函数
    {
        "tools": "tools",       # 条件函数返回 "tools" → 走到 tools 节点
        END: END,               # 条件函数返回 END → 结束
    }
)
graph_builder.add_edge("tools", "llm")             # tools → 回到 LLM

# 4. 编译
graph = graph_builder.compile()

# 5. 运行
result = graph.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？")]
})
```

---

## 4. LangGraph vs LangChain

### 4.1 核心区别

```
                LangChain LCEL          LangGraph
─────────────────────────────────────────────────────
执行模式         线性管道（DAG）          有环图（支持循环）
状态管理         无内建状态管理           中心化 State 管理
循环支持         不支持                  原生支持
条件分支         有限支持                完整支持
持久化           需要外部实现             内建 Checkpointer
并发             RunnableParallel        原生并行节点
调试             LangSmith 追踪          图可视化 + 步骤追踪
适用场景         简单链式任务             复杂 Agent、工作流
```

### 4.2 选择指南

```
你的任务是什么？
│
├── 简单的"输入→处理→输出"（RAG、摘要、分类）
│   └── 用 LCEL（更简单，更快）
│
├── 需要工具调用，但流程固定
│   └── 用 LCEL + Tool Calling
│
├── 需要 Agent 自主决策（循环、条件分支）
│   └── 用 LangGraph
│
├── 多个 Agent 协作
│   └── 用 LangGraph Multi-Agent
│
└── 需要人工审批步骤
    └── 用 LangGraph + Human-in-the-loop
```

---

## 5. LangGraph 实现 ReAct Agent

### 5.1 ReAct 模式原理

ReAct（Reason + Act）是目前最主流的 Agent 模式：

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                         │
│                                                             │
│  用户输入                                                    │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Thought（思考）                                      │  │
│  │  "我需要先查询天气 API 获取北京的天气数据"              │  │
│  └─────────────────────────┬────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Action（行动）                                       │  │
│  │  调用 get_weather(city="北京")                        │  │
│  └─────────────────────────┬────────────────────────────┘  │
│                            │                               │
│                            ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Observation（观察）                                  │  │
│  │  "北京今天晴天，气温 28°C，湿度 45%"                   │  │
│  └─────────────────────────┬────────────────────────────┘  │
│                            │                               │
│            ┌───────────────┤                               │
│            │               │                               │
│            ▼               ▼                               │
│    需要更多信息？         可以回答了？                       │
│    → 继续循环             → 生成最终答案                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 用 LangGraph 实现 ReAct

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. 定义状态
class ReActState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义工具
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    # 实际实现
    return f"搜索结果：关于'{query}'的信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2 + 3 * 4'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

tools = [search_web, calculator]

# 3. 创建绑定了工具的模型
model = ChatAnthropic(model="claude-opus-4-8").bind_tools(tools)

# 4. 定义节点
def agent_node(state: ReActState) -> dict:
    """Agent 思考节点：调用 LLM 决定下一步"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode 是 LangGraph 内置的工具执行节点
tool_node = ToolNode(tools)

# 5. 定义条件边
def route_after_agent(state: ReActState) -> str:
    """根据 LLM 的输出决定下一步"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 6. 构建图
workflow = StateGraph(ReActState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "agent")  # 工具执行后回到 agent

# 7. 编译并运行
react_agent = workflow.compile()

# 使用 LangGraph 内置的快捷方式（等价于上面的完整实现）
from langgraph.prebuilt import create_react_agent
quick_agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),
    tools=tools,
    state_modifier="你是一个有用的助手，擅长搜索和计算"
)

# 运行 Agent
result = react_agent.invoke({
    "messages": [HumanMessage(content="上海今天气温是多少？加上10度是多少？")]
})

for message in result["messages"]:
    print(f"{message.__class__.__name__}: {message.content[:100]}")
```

### 5.3 ReAct 的图结构可视化

```
START
  │
  ▼
[agent]  ←────────────────────────┐
  │                               │
  ├── tool_calls 存在 ──→ [tools] ─┘
  │
  └── 无 tool_calls ──→ END
```

---

## 6. LangGraph 实现 Multi-Agent

### 6.1 Supervisor 模式（最常用）

```
                用户请求
                   │
                   ▼
         ┌─────────────────┐
         │   Supervisor    │  ← 决策节点：分配任务给哪个 Agent
         │  （主 LLM）      │
         └────────┬────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
 [Researcher]  [Coder]   [Writer]
  搜索信息      写代码     写文档
      │           │           │
      └───────────┴───────────┘
                  │
                  ▼
         [Supervisor 再次评估]
                  │
          任务完成 │ 需要继续
                  ▼
               结束/继续
```

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal
import json

# 1. 定义多 Agent 状态
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str  # 下一个执行的 Agent

# 2. 创建专门化的 Agent
def create_specialist_agent(role: str, tools: list):
    model = ChatAnthropic(model="claude-opus-4-8").bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def agent_fn(state):
        # 在 messages 前面注入角色说明
        messages = [SystemMessage(content=f"你是{role}，专注于{role}相关任务")] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    
    return agent_fn

researcher_agent = create_specialist_agent("研究员", [search_web])
coder_agent = create_specialist_agent("程序员", [run_code, read_file])
writer_agent = create_specialist_agent("文档作者", [read_file, write_file])

# 3. Supervisor 节点
AGENTS = ["researcher", "coder", "writer"]

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor：决定下一步由谁执行"""
    
    system = f"""你是任务分配器。基于对话历史，决定下一步由哪个 Agent 执行。
    
可用 Agent：{', '.join(AGENTS)}
如果任务已完成，返回 FINISH。

返回 JSON 格式：{{"next": "researcher" | "coder" | "writer" | "FINISH"}}"""
    
    messages = [SystemMessage(content=system)] + state["messages"]
    response = ChatAnthropic(model="claude-opus-4-8").invoke(messages)
    
    # 解析决策
    decision = json.loads(response.content)
    return {"next": decision["next"]}

# 4. 路由函数
def route_supervisor(state: SupervisorState) -> str:
    return state["next"]

# 5. 构建 Multi-Agent 图
workflow = StateGraph(SupervisorState)

# 添加所有节点
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("writer", writer_agent)

# 从 START 到 supervisor
workflow.add_edge(START, "supervisor")

# supervisor 根据决策路由到不同 agent
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "FINISH": END,
    }
)

# 所有 agent 执行完回到 supervisor
for agent in ["researcher", "coder", "writer"]:
    workflow.add_edge(agent, "supervisor")

# 编译
multi_agent = workflow.compile()
```

### 6.2 Hierarchical 模式（分层）

```python
# 嵌套 Agent：子图作为父图的一个节点

# 子图1：研究子系统
research_subgraph = StateGraph(ResearchState)
research_subgraph.add_node("search", search_node)
research_subgraph.add_node("summarize", summarize_node)
research_subgraph.add_edge(START, "search")
research_subgraph.add_edge("search", "summarize")
research_subgraph.add_edge("summarize", END)
compiled_research = research_subgraph.compile()

# 子图2：写作子系统
writing_subgraph = StateGraph(WritingState)
# ... 类似结构
compiled_writing = writing_subgraph.compile()

# 父图：将子图作为节点使用
main_graph = StateGraph(MainState)
main_graph.add_node("research", compiled_research)  # 子图作为节点
main_graph.add_node("writing", compiled_writing)
```

---

## 7. LangGraph 状态管理与持久化

### 7.1 Checkpointer（检查点）

Checkpointer 是 LangGraph 持久化能力的核心，它在每个节点执行后自动保存状态快照：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# 内存检查点（开发测试用）
memory_checkpointer = MemorySaver()

# SQLite 检查点（本地持久化）
sqlite_checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

# PostgreSQL 检查点（生产级持久化）
postgres_checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/dbname"
)

# 编译时传入 Checkpointer
graph = workflow.compile(checkpointer=sqlite_checkpointer)

# 运行时需要指定 thread_id（同一 thread_id 共享状态）
config = {"configurable": {"thread_id": "conversation_001"}}

# 第一次调用
result1 = graph.invoke(
    {"messages": [HumanMessage(content="我叫小明")]},
    config=config
)

# 第二次调用（同一 thread_id，自动加载之前的状态）
result2 = graph.invoke(
    {"messages": [HumanMessage(content="你记得我叫什么吗？")]},
    config=config
)
# result2 中的 Agent 会知道用户叫小明
```

### 7.2 Human-in-the-Loop（人工介入）

```python
from langgraph.graph import interrupt

# 在节点中插入中断点
def review_node(state: AgentState) -> dict:
    """需要人工审核的节点"""
    plan = state["plan"]
    
    # interrupt 会暂停图的执行，等待人工输入
    human_feedback = interrupt({
        "type": "review_request",
        "plan": plan,
        "message": "请审核此计划，是否继续执行？"
    })
    
    if human_feedback["approved"]:
        return {"plan": plan, "status": "approved"}
    else:
        return {
            "plan": human_feedback.get("revised_plan", plan),
            "status": "revised"
        }

# 编译时必须指定 checkpointer（中断需要持久化状态）
graph = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_code"],  # 在执行代码前自动中断
)

# 运行直到第一个中断点
config = {"configurable": {"thread_id": "task_001"}}
result = graph.invoke({"messages": [...]}, config=config)
# 此时 result 包含中断信息

# 人工审核后，用 Command 恢复执行
from langgraph.types import Command

resumed = graph.invoke(
    Command(resume={"approved": True}),  # 传入人工决策
    config=config
)
```

### 7.3 时间旅行（Time Travel）

```python
# 查看某个 thread 的所有历史状态快照
history = list(graph.get_state_history(config))

for snapshot in history:
    print(f"Step {snapshot.metadata.get('step')}: {snapshot.values['messages'][-1].content[:50]}")

# 回滚到某个历史状态重新执行（"时间旅行"）
target_snapshot = history[3]  # 第 4 个状态
graph.update_state(config, target_snapshot.values, as_node="agent")

# 从该历史点重新执行（可以用于调试和修复）
new_result = graph.invoke(None, config=config)
```

---

## 8. 框架横向对比：AutoGen、CrewAI

### 8.1 全面对比表

| 维度 | LangGraph | AutoGen | CrewAI |
|------|-----------|---------|--------|
| 核心抽象 | 状态图（Graph） | 对话式 Agent | 角色（Role）+ 任务（Task） |
| 编程范式 | 声明式图构建 | 对话驱动 | 高层 API |
| 学习曲线 | 中等 | 低 | 低 |
| 灵活性 | 极高 | 中等 | 中等 |
| 状态管理 | 中心化 State | 分散在对话历史 | 任务属性 |
| 持久化 | 内建 Checkpointer | 需要自行实现 | 需要自行实现 |
| 人工介入 | 原生支持 | 支持（对话暂停） | 有限支持 |
| 可观测性 | LangSmith | AutoGen Studio | 有限 |
| 生产就绪度 | 高 | 中 | 中 |
| 适合场景 | 复杂工作流 | 代码生成任务 | 角色扮演协作 |

### 8.2 AutoGen 架构

```python
# AutoGen 的核心概念：两个 Agent 互相对话
import autogen

# 配置 LLM
config_list = [{"model": "claude-opus-4-8", "api_key": "..."}]

# 助手 Agent：负责生成代码
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="你是一个 Python 编程专家。"
)

# 用户代理：代表用户，负责执行代码和反馈
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",   # NEVER/TERMINATE/ALWAYS
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "./workspace"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
)

# 启动对话：两个 Agent 互相交流直到任务完成
user_proxy.initiate_chat(
    assistant,
    message="写一个计算斐波那契数列的 Python 函数，并运行测试"
)
```

AutoGen 的核心循环：

```
UserProxy 发送消息
       │
       ▼
Assistant 生成回复（可能包含代码块）
       │
       ▼
UserProxy 检测到代码块 → 在沙盒中执行
       │
       ▼
UserProxy 把执行结果发回 Assistant
       │
       ▼
Assistant 基于结果继续对话 → 循环直到 TERMINATE
```

### 8.3 CrewAI 架构

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# 定义工具
search_tool = SerperDevTool()

# 定义 Agent（角色驱动）
researcher = Agent(
    role="市场研究员",
    goal="找出{topic}领域的最新趋势和关键数据",
    backstory="""你是一位经验丰富的市场研究员，擅长从海量信息中
    提取有价值的洞察，你的分析报告以准确和深度著称。""",
    tools=[search_tool],
    verbose=True,
)

writer = Agent(
    role="内容撰写师",
    goal="将研究数据转化为清晰易读的报告",
    backstory="你是一位专业的商业内容撰写师，擅长把复杂数据讲成故事。",
    verbose=True,
)

# 定义任务（任务依赖关系）
research_task = Task(
    description="深入研究 {topic} 的市场现状、主要玩家和未来趋势",
    expected_output="一份包含数据支撑的详细研究报告",
    agent=researcher,
)

writing_task = Task(
    description="基于研究报告，撰写一篇 1000 字的市场分析文章",
    expected_output="一篇结构清晰、数据充分的市场分析文章",
    agent=writer,
    context=[research_task],  # 依赖研究任务的输出
)

# 创建 Crew（小队）
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # sequential 或 hierarchical
    verbose=True,
)

# 执行
result = crew.kickoff(inputs={"topic": "AI Agent 框架"})
```

---

## 9. 架构图与流程图汇总

### 9.1 LangChain LCEL 执行流程

```
输入数据
   │
   ▼
┌─────────────┐
│  Runnable 1  │  (e.g., PromptTemplate)
│  invoke(x)   │
└──────┬──────┘
       │ output_1
       ▼
┌─────────────┐
│  Runnable 2  │  (e.g., ChatAnthropic)
│  invoke(x)   │
└──────┬──────┘
       │ output_2
       ▼
┌─────────────┐
│  Runnable 3  │  (e.g., StrOutputParser)
│  invoke(x)   │
└──────┬──────┘
       │
       ▼
    最终输出
```

### 9.2 LangGraph StateGraph 执行模型

```
┌─────────────────────────────────────────────────────────────┐
│                    StateGraph 执行模型                        │
│                                                             │
│  编译时：                                                    │
│  ┌───────┐   ┌───────┐   ┌───────┐                         │
│  │ Node1 │──▶│ Node2 │──▶│ Node3 │──▶ END                  │
│  └───────┘   └───────┘   └───┬───┘                         │
│                               │ 条件边                      │
│                               ▼                             │
│                          ┌────────┐                         │
│                          │ Node4  │──────────────────────▶  │
│                          └────────┘                         │
│                                                             │
│  运行时：                                                    │
│  State_0 → Node1(State_0) → State_1                        │
│         → Node2(State_1) → State_2                        │
│         → route(State_2) → Node3 or Node4                 │
│         → Node3(State_2) → State_3                        │
│         → END                                              │
│                                                             │
│  每步执行后 Checkpointer 保存 State 快照                     │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 RAG 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 架构                               │
│                                                             │
│  离线阶段（索引构建）：                                       │
│  原始文档 → 文本切分 → 向量化 → 存入向量库                   │
│                                                             │
│  在线阶段（查询服务）：                                       │
│                                                             │
│  用户问题                                                    │
│      │                                                      │
│      ▼                                                      │
│  [Query Embedding]  ← 将问题也向量化                         │
│      │                                                      │
│      ▼                                                      │
│  [Vector Search]    ← 在向量库中找最相似的文档块              │
│      │                                                      │
│      ▼                                                      │
│  Top-K 相关文档                                              │
│      │                                                      │
│      ▼                                                      │
│  [Prompt 构建]      ← 问题 + 相关文档 → 完整 Prompt          │
│      │                                                      │
│      ▼                                                      │
│  [LLM 推理]         ← 基于上下文生成回答                      │
│      │                                                      │
│      ▼                                                      │
│  [引用追踪]         ← 标注答案来源                            │
│      │                                                      │
│      ▼                                                      │
│  最终答案（含引用）                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 关键源码分析

### 10.1 RunnableSequence（LCEL 核心）

```python
# 简化版 RunnableSequence 实现原理
class RunnableSequence(Runnable):
    """
    | 操作符背后的实现
    chain = A | B | C 等价于 RunnableSequence([A, B, C])
    """
    
    def __init__(self, steps: list[Runnable]):
        self.steps = steps
    
    def invoke(self, input: Any, config=None) -> Any:
        current = input
        for step in self.steps:
            current = step.invoke(current, config)
        return current
    
    async def ainvoke(self, input: Any, config=None) -> Any:
        current = input
        for step in self.steps:
            current = await step.ainvoke(current, config)
        return current
    
    def stream(self, input: Any, config=None) -> Iterator:
        # 只有最后一个 step 需要 stream
        current = input
        for step in self.steps[:-1]:
            current = step.invoke(current, config)
        yield from self.steps[-1].stream(current, config)
    
    def __or__(self, other: Runnable) -> "RunnableSequence":
        # 支持链式 | 操作，优化为扁平结构
        if isinstance(other, RunnableSequence):
            return RunnableSequence(self.steps + other.steps)
        return RunnableSequence(self.steps + [other])
```

### 10.2 LangGraph 状态更新机制

```python
# 简化版 StateGraph 核心逻辑
class StateGraph:
    def __init__(self, state_schema: Type[TypedDict]):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        # 提取每个字段的 reducer（来自 Annotated 类型注解）
        self.reducers = self._extract_reducers(state_schema)
    
    def _apply_updates(self, state: dict, updates: dict) -> dict:
        """将节点返回的更新合并到当前状态"""
        new_state = dict(state)
        
        for key, value in updates.items():
            if key in self.reducers:
                # 有 reducer：使用 reducer 合并（如 add_messages）
                reducer = self.reducers[key]
                new_state[key] = reducer(state.get(key), value)
            else:
                # 无 reducer：直接覆盖
                new_state[key] = value
        
        return new_state
    
    def _run_node(self, node_name: str, state: dict) -> dict:
        """执行单个节点"""
        node_fn = self.nodes[node_name]
        updates = node_fn(state)  # 节点函数返回 State 的更新部分
        return self._apply_updates(state, updates)
    
    def _get_next_node(self, current_node: str, state: dict) -> str:
        """确定下一个执行的节点"""
        if current_node in self.conditional_edges:
            condition_fn, mapping = self.conditional_edges[current_node]
            result = condition_fn(state)
            return mapping[result]
        elif current_node in self.edges:
            return self.edges[current_node]
        return END
```

### 10.3 Checkpointer 的工作原理

```python
# 检查点机制的简化实现
class MemorySaver:
    """内存检查点：每次节点执行后保存状态快照"""
    
    def __init__(self):
        # {thread_id: [(step, state, metadata), ...]}
        self.storage = {}
    
    def put(self, config: dict, checkpoint: dict, metadata: dict):
        """保存一个状态快照"""
        thread_id = config["configurable"]["thread_id"]
        if thread_id not in self.storage:
            self.storage[thread_id] = []
        
        self.storage[thread_id].append({
            "step": metadata.get("step"),
            "checkpoint": checkpoint,
            "metadata": metadata,
            "timestamp": time.time()
        })
    
    def get(self, config: dict) -> dict | None:
        """获取最新的状态快照"""
        thread_id = config["configurable"]["thread_id"]
        snapshots = self.storage.get(thread_id, [])
        if not snapshots:
            return None
        return snapshots[-1]["checkpoint"]
    
    def list(self, config: dict) -> list[dict]:
        """列出所有历史快照（用于时间旅行）"""
        thread_id = config["configurable"]["thread_id"]
        return list(reversed(self.storage.get(thread_id, [])))
```

### 10.4 完整的生产级 LangGraph Agent

```python
"""
生产级 LangGraph Agent 示例
包含：工具调用、错误处理、检查点、流式输出
"""
from typing import TypedDict, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 状态定义
class ProductionAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error_count: int
    task_completed: bool

# 工具定义
@tool
def search_docs(query: str, max_results: int = 5) -> str:
    """搜索内部文档库。
    
    Args:
        query: 搜索关键词
        max_results: 最多返回多少条结果（1-10）
    """
    # 实际实现：调用向量数据库
    return f"找到 {max_results} 条关于 '{query}' 的结果..."

@tool
def execute_python(code: str) -> str:
    """在沙箱中执行 Python 代码。
    
    Args:
        code: 要执行的 Python 代码
    """
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__})
        return output.getvalue() or "代码执行成功（无输出）"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"

tools = [search_docs, execute_python]
model = ChatAnthropic(model="claude-opus-4-8").bind_tools(tools)
tool_node = ToolNode(tools)

# 节点实现
def agent_node(state: ProductionAgentState) -> dict:
    system = SystemMessage(content="""你是一个专业的技术助手。
    - 需要查找信息时，使用 search_docs 工具
    - 需要验证计算或代码时，使用 execute_python 工具
    - 完成任务后，清晰总结结果""")
    
    messages = [system] + state["messages"]
    
    try:
        response = model.invoke(messages)
        return {
            "messages": [response],
            "error_count": 0  # 成功后重置错误计数
        }
    except Exception as e:
        error_msg = f"LLM 调用失败：{e}"
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=error_msg)],
            "error_count": state.get("error_count", 0) + 1
        }

def should_continue(state: ProductionAgentState) -> Literal["tools", "end"]:
    last_msg = state["messages"][-1]
    
    # 超过错误阈值，强制结束
    if state.get("error_count", 0) >= 3:
        return "end"
    
    # 有工具调用，继续执行
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    return "end"

# 构建图
workflow = StateGraph(ProductionAgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

# 带检查点编译
checkpointer = MemorySaver()
production_agent = workflow.compile(checkpointer=checkpointer)

# 使用示例（支持流式输出）
config = {"configurable": {"thread_id": "session_001"}}

print("=== 流式输出 ===")
for chunk in production_agent.stream(
    {"messages": [HumanMessage(content="用 Python 计算 1 到 100 的所有质数")],
     "error_count": 0,
     "task_completed": False},
    config=config,
    stream_mode="values"  # 或 "updates"、"messages"
):
    last_msg = chunk["messages"][-1]
    if hasattr(last_msg, "content") and last_msg.content:
        print(f"\r{last_msg.content[:80]}...", end="", flush=True)

print("\n=== 完成 ===")
```

---

## 总结

LangChain 和 LangGraph 代表了 Agent 框架的两个层次：

**LangChain/LCEL** 解决的是 **组合问题**：如何优雅地把 LLM 调用、工具调用、格式转换串联起来，适合 80% 的"一次性"任务。

**LangGraph** 解决的是 **编排问题**：如何管理有状态的、循环的、需要人工介入的复杂工作流，适合真正的 Agent 系统。

与 AutoGen 和 CrewAI 相比，LangGraph 的优势在于：
1. **灵活性**：图结构可以表达任何工作流模式
2. **可观测性**：每个状态快照都可以追踪和回放
3. **生产就绪**：内建持久化、中断恢复、时间旅行
4. **可控性**：精确控制 Agent 的决策边界

对于前端工程师来说，LangGraph 的图模型很类似于前端状态机（XState），一旦建立起这个类比，学习曲线会大幅降低。
