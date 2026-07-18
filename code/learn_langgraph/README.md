# LangGraph 零基础学习示例

LangGraph 是用来构建「有状态、多步骤」AI 工作流的框架——把它想象成一张流程图：每个处理步骤是一个节点，步骤之间用边连接，所有节点共享一个状态。

这是为零基础用户准备的 6 篇循序渐进教程，建议学完 [`learn_langchain/`](../learn_langchain/README.md) 后再来（需要先懂链、工具调用、记忆等基础概念）。

---

## 🚀 5 分钟跑通第一篇

### 1. 配置 API Key

在 `code/.env` 文件里写好（跟 learn_langchain 共用同一个）：

```
OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=https://xxxx
LLM_MODEL=xxxxx
```

### 2. 运行第一篇

```bash
cd code
uv run python learn_langgraph/01_basics.py
```

---

## 📚 学习路线（按顺序学，每篇约 15-20 分钟）

| 顺序 | 文件 | 学什么 | 关键概念 |
|:---:|------|--------|---------|
| 01 | `01_basics.py` | 画第一张图 | `State`、`Node`、`Edge`、`StateGraph` |
| 02 | `02_conditional.py` | 条件分支与循环 | `add_conditional_edges`、循环 |
| 03 | `03_tool_agent.py` | 会调工具的 Agent | `ToolNode`、`tools_condition`、ReAct |
| 04 | `04_memory_persistence.py` | 记忆与持久化 | `Checkpointer`、`MemorySaver`、`thread_id` |
| 05 | `05_human_in_loop.py` | 人机协作 | `interrupt_before`、`interrupt()`、`Command` |
| 06 | `06_workflow.py` | 多节点工作流 | 串行流水线、并行 Map-Reduce、审校循环 |

### 学习建议

- **先学 learn_langchain 的 01-05**：LangGraph 假设你已懂模型调用、工具、记忆
- **严格按 01 → 06 顺序**：每篇都建立在前面之上
- **每篇先读顶部注释**：开头有【这章学什么】【为什么学它】【类比】
- **用 stream() 看执行过程**：调试图的最佳方式，能看到每个节点的流转

---

## 🗺️ 整体认知地图

```
01 基础（State + 节点 + 边 = 一张图）── 骨架
   ↓
02 条件边 + 循环 ── 让图能分叉、能打回
   ↓
   ├─→ 03 工具 Agent（条件边 + ToolNode = ReAct 智能体）
   ├─→ 04 记忆（Checkpointer 让图记住对话）
   └─→ 05 人机协作（interrupt 在关键步骤暂停等人确认）

06 工作流（综合：串行 + 并行 + 循环 = 完整生产线）
```

**核心心智模型**：LangGraph 一切皆"图"。你只需要回答三个问题：
1. **State 有哪些字段？**（白板上记什么）
2. **有哪些节点？每个节点做什么？**（处理步骤）
3. **节点之间怎么连？哪里分叉、哪里循环？**（流程）

把这三点想清楚，代码就是照着翻译成 `add_node` / `add_edge` / `add_conditional_edges`。

---

## 🔄 LangChain vs LangGraph 什么时候用哪个？

| 场景 | 用什么 |
|------|--------|
| 简单的「输入→处理→输出」一条线 | LangChain 链（LCEL `\|`） |
| 需要分支、循环、多步骤协作 | LangGraph 图 |
| 标准的"会调工具的 Agent" | 都行（LangGraph 的 03 或 LangChain 的 create_agent） |
| 复杂工作流（并行、人工介入、多阶段） | LangGraph |

经验法则：**先想任务是不是"一条直线"，是就用 LangChain，有分叉/循环就用 LangGraph。**

---

## ⚠️ 注意事项

- **`create_react_agent` 已废弃**：LangGraph 官方推荐用 `langchain.agents.create_agent`（带 middleware，见 `learn_langchain/09_middleware.py`）。本教程的 03 篇用手搓 StateGraph，是为了让你理解 Agent 的内部原理。
- **interrupt 必须配 checkpointer**：人机协作的暂停/恢复依赖记忆，没 checkpointer 会报错。
- **循环要加最大次数保护**：防止条件边形成的循环无限转下去。

---

## 📦 依赖

依赖已写进 `code/pyproject.toml` 并锁定在 `code/uv.lock`。首次运行前：

```bash
cd code
uv sync
```

主要用到：`langgraph`、`langchain`、`langchain-openai`、`langgraph-checkpoint-sqlite`、`python-dotenv`。

---

## ❓ 遇到问题？

| 报错 | 原因 | 解决 |
|------|------|------|
| `AuthenticationError` | `.env` 里 Key 没填或填错 | 检查 `OPENAI_API_KEY` |
| `No module named xxx` | 依赖没装 | 跑 `uv sync` |
| `GraphRecursionError` | 循环转太多圈 | 加最大次数保护，或调大 `recursion_limit` |
| interrupt 报错 | 没配 checkpointer | `compile(checkpointer=MemorySaver())` |
