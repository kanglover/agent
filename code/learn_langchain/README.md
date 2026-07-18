# LangChain 零基础学习示例

这是为零基础用户准备的 LangChain 学习合集，**从最简单的调用讲起，循序渐进**，每个示例都能直接运行。

所有示例统一使用**"LLM**（通过 OpenAI 兼容接口），所以你只需要配一个 Key 就能跑通全部 8 篇，不用再申请别家的。

---

## 🚀 5 分钟跑通第一篇

### 1. 配置 API Key

在 `code/.env` 文件里写好（如果还没有这个文件就新建）：

```
OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=https://xxxx
LLM_MODEL=xxxxx
```

### 2. 运行第一篇

```bash
cd code
uv run python learn_langchain/01_hello_chain.py
```

看到模型回复就说明成功了。后续每篇都是同样的运行方式，把文件名换掉即可。

---

## 📚 学习路线（按顺序学，每篇约 10-15 分钟）

| 顺序 | 文件 | 学什么 | 关键概念 |
|:---:|------|--------|---------|
| 01 | `01_hello_chain.py` | 第一次调用模型 | `ChatOpenAI`、`invoke`、`AIMessage` |
| 02 | `02_prompt_template.py` | 用模板复用提示词 | `PromptTemplate`、`ChatPromptTemplate` |
| 03 | `03_output_parser.py` | LCEL 管道串成链 | `\|` 管道、`StrOutputParser` |
| 04 | `04_structured_output.py` | 让模型吐结构化数据 | `with_structured_output`、Pydantic |
| 05 | `05_tools.py` | 让模型调用工具 | `@tool`、`bind_tools`、工具调用循环 |
| 06 | `06_memory.py` | 多轮对话的记忆 | `RunnableWithMessageHistory`、`trim_messages` |
| 07 | `07_rag.py` | 检索增强生成全流程 | 切块、Embedding、向量库、检索 |
| 08 | `08_lcel_advanced.py` | LCEL 进阶编排 | `RunnableLambda`、路由、降级、并行 |
| 09 | `09_middleware.py` | 中间件（在 Agent 每步插钩子） | `AgentMiddleware`、`create_agent`、`wrap_model_call` |

### 学习建议

- **严格按 01 → 09 顺序**：每篇都建立在上一篇的基础上，跳着看容易卡住
- **每篇先读顶部注释**：开头有【这章学什么】【为什么学它】【类比】三段，帮你建立直觉
- **跑一遍再改着玩**：把示例里的问题、变量换成你自己的，观察输出变化，理解更深
- **不必一次学完**：01-04 是核心基础（链 + 结构化），先吃透；05-08 是进阶应用

---

## 🗺️ 整体认知地图

学完这 9 篇，你就掌握了 LangChain 的核心。它们之间的关系：

```
01 调用模型（最底层，一切的基础）
   ↓
02 提示词模板（让调用可复用）
   ↓
03 LCEL 链（模板|模型|解析器，串成流水线）── 核心骨架
   ↓
   ├─→ 04 结构化输出（让链吐对象）
   ├─→ 05 工具调用（让链会办事）── Agent 的基础
   ├─→ 06 记忆（让链记得上下文）
   └─→ 07 RAG（让链查资料再答）

08 LCEL 进阶（路由/降级/并行，把链拼成复杂工作流）
09 Middleware（在 Agent 每步插钩子：日志/重试/改行为）
```

**学完之后**：
- 把 05（工具）+ 06（记忆）组合起来，就是一个完整的 **Agent（智能体）**
- 09 的 `create_agent` 是官方推荐的 Agent 快捷构建方式（自带 middleware 系统）
- 想学更复杂的工作流（分支、循环、并行）？继续看 [`learn_langgraph/`](../learn_langgraph/README.md)

---


## 📦 依赖

依赖已写进 `code/pyproject.toml` 并锁定在 `code/uv.lock`。第一次运行前执行一次同步即可：

```bash
cd code
uv sync
```

主要用到的库：`langchain`、`langchain-openai`、`langchain-chroma`、`langchain-text-splitters`、`python-dotenv`、`pydantic`。

---

## ❓ 遇到问题？

| 报错 | 原因 | 解决 |
|------|------|------|
| `AuthenticationError` | `.env` 里 Key 没填或填错 | 检查 `OPENAI_API_KEY` 是否正确 |
| `externally-managed-environment` | 没用虚拟环境/uv | 必须用 `uv run` 而非裸 `python` |
| `No module named xxx` | 依赖没装 | 跑 `uv sync` 同步依赖 |
| `RateLimitError` | 调用太频繁或额度用完 | 等一会再试，或检查百炼平台额度 |
