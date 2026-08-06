# LangGraph 面试题精选

> 来源：模型训练知识（截止 2026 年 1 月），涵盖 LangGraph 官方文档及工程实践

---

## 一、StateGraph 基础

**Q1. LangGraph 的 StateGraph 是什么？和普通 LangChain AgentExecutor 有什么区别？**

答题要点：
- StateGraph 是 LangGraph 的核心抽象：把 Agent 逻辑建模为**有向图**（可含环），Node = 处理步骤，Edge = 条件路由
- AgentExecutor 是线性 ReAct 循环：无分支、无环、无并行
- LangGraph 优势：
  1. 支持环（Agent 可反复重试）
  2. 支持条件分支（如置信度低则转人工）
  3. 内置状态持久化（Checkpointing，支持断点续跑）
  4. 可视化和调试更友好（图结构清晰）

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    step_count: int

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

**Q2. LangGraph 中 State 的 Reducer 是什么？`add_messages` 做了什么？**

答题要点：
- 每个 Node 返回一个 dict，LangGraph 用 **Reducer** 把这个 dict merge 进全局 State
- 默认 Reducer：**覆盖**（直接替换字段值）
- `add_messages` Reducer：**追加**消息列表而不是覆盖
  - 使用 `Annotated[list, add_messages]` 声明字段类型
  - 还会自动去重（根据 message id），防止重放时重复追加
- 自定义 Reducer：可传任意函数 `Annotated[T, your_reducer_fn]`

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]   # 追加模式
    user_id: str                               # 覆盖模式（默认）
```

---

**Q3. 什么是条件边（Conditional Edge）？如何实现动态路由？**

答题要点：
- `add_conditional_edges(source, condition_fn, path_map)` 根据函数返回值决定走哪条边
- `condition_fn` 接收当前 State，返回一个字符串 key
- `path_map` 把 key 映射到目标 Node 名称
- 常见用法：判断是否需要调用工具、是否触发人工审核、置信度路由

```python
def should_continue(state: State) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
```

---

## 二、Checkpointing 与持久化

**Q4. LangGraph 的 Checkpointer 是什么？支持哪些后端？**

答题要点：
- Checkpointer 是 LangGraph 持久化的核心：每个 Node 执行完后**自动保存状态快照**
- 支持后端：
  - `MemorySaver`：内存，仅用于开发/测试，重启丢失
  - `SqliteSaver`：SQLite 文件，本地持久化
  - `PostgresSaver`：PostgreSQL，生产级别
- 使用方式：在 `compile()` 时传入，通过 `config={"configurable": {"thread_id": "xxx"}}` 区分会话

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 开发
checkpointer = MemorySaver()

# 生产（SQLite）
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

# 执行时指定 thread_id
config = {"configurable": {"thread_id": "user_session_42"}}
result = app.invoke({"messages": [...]}, config=config)
```

---

**Q5. 什么是 LangGraph 的"Time Travel"功能？**

答题要点：
- 允许查看和回滚到任意历史状态快照，用于调试和修复
- 依赖 Checkpointer（每步都有快照）
- `graph.get_state_history(config)` → 返回所有历史快照列表
- `graph.update_state(config, values, as_node=...)` → 恢复到指定状态并可修改
- 典型用途：发现 Agent 某步走错了，回滚到该步，修正后重新执行

---

**Q6. Human-in-the-Loop 如何在 LangGraph 中实现？**

答题要点（两种方式）：

**方式一：`interrupt()` 动态中断**
- 在 Node 函数内部调用 `interrupt(payload)` 动态暂停
- `payload` 是展示给人类的信息（如"是否确认执行此操作？"）
- 恢复：`graph.invoke(Command(resume=human_response), config=config)`

**方式二：`interrupt_before` 静态中断**
- 在 `compile(interrupt_before=["node_name"])` 时指定
- 每次进入该 Node 前自动暂停

```python
from langgraph.types import interrupt, Command

def approval_node(state: State):
    decision = interrupt({
        "question": "是否执行删除操作？",
        "affected_files": state["files_to_delete"]
    })
    if decision == "yes":
        return {"status": "approved"}
    return {"status": "rejected"}
```

**注意：** 两种方式都需要 Checkpointer 才能暂停和恢复。

---

## 三、Multi-Agent with LangGraph

**Q7. LangGraph 中如何实现 Supervisor 多 Agent 模式？**

答题要点：
- Supervisor 本身是一个 Node（通常是一个 LLM），负责决定调用哪个 Worker Agent
- Worker 以 Node 形式存在，Supervisor 通过条件边路由
- `Command` 对象：Worker 返回 `Command(goto="supervisor", update={...})` 实现受控跳转

```python
from langgraph.types import Command

def worker_agent(state: State) -> Command:
    result = llm.invoke(...)
    return Command(
        goto="supervisor",          # 完成后回到 Supervisor
        update={"messages": result} # 更新状态
    )
```

---

**Q8. LangGraph 中 subgraph（子图）是什么？什么时候用？**

答题要点：
- 子图是一个独立编译的 `StateGraph`，可以作为父图的一个 Node 嵌入
- **私有状态**：子图有自己的 State schema，与父图隔离
- **共享状态**：通过 overlapping key 实现父子图数据传递
- 适用场景：
  - 封装可复用的 Agent 逻辑（如"研究 Agent"可在多个流程中复用）
  - 复杂层次化多 Agent 系统
  - 不同团队负责不同子图，独立开发和测试

---

**Q9. `graph.stream()` 的 `stream_mode` 有哪些？**

答题要点：

| stream_mode | 含义 | 适用场景 |
|-------------|------|---------|
| `"values"` | 每个 Node 执行完后输出**完整当前 State** | 最常用，展示最终消息流 |
| `"updates"` | 每个 Node 执行完后只输出**该 Node 的增量 dict** | 调试，分析每步变化 |
| `"messages"` | 流式输出 LLM **token 级别消息增量** | 实时打字机效果 |

```python
# 流式展示 token
for chunk in app.stream(input, config, stream_mode="messages"):
    if chunk[1]["langgraph_node"] == "agent":
        print(chunk[0].content, end="", flush=True)
```

---

**Q10. LangGraph 如何处理并行执行？**

答题要点：
- 使用 `send()` API 向同一 Node 并发派发多个任务
- State 中用 `Annotated[list, operator.add]` 收集并行结果
- 并行 Node 应**无共享可变状态**（避免竞争条件）

```python
import operator
from langgraph.types import Send

class State(TypedDict):
    tasks: list[str]
    results: Annotated[list, operator.add]  # 并行结果自动合并

def dispatcher(state: State):
    return [Send("worker", {"task": t}) for t in state["tasks"]]

graph.add_conditional_edges("dispatcher", dispatcher)
```

---

## 四、高频追问

**Q11. LangGraph 中 `END` 和 `interrupt()` 有什么区别？**
- `END`：正常结束，图执行完毕，状态保存到 Checkpointer（若有）
- `interrupt()`：**暂停**，等待外部输入后**恢复**继续，图并未结束，状态挂起在 Checkpointer 中

**Q12. 如何给 LangGraph 的 Agent 设置最大迭代次数防止无限循环？**
- 在 `compile(recursion_limit=25)` 设置最大递归深度（默认 25）
- 超过限制抛出 `GraphRecursionError`，可在外层 try-catch 处理
- 也可在 State 中维护 `step_count`，条件边检查是否超限

**Q13. LangGraph 与 AutoGen、CrewAI 的核心区别？**

| 框架 | 核心抽象 | 优势 | 劣势 |
|------|---------|------|------|
| LangGraph | StateGraph（有状态图） | 高灵活性、内置持久化与中断恢复、生产就绪 | 学习曲线较陡 |
| AutoGen | 对话式 Agent | 入门容易、代码生成强 | 状态散落在对话历史，流程精确控制难 |
| CrewAI | 角色 + 任务 | 高层 API 简洁、角色扮演直觉化 | 持久化/中断需自行实现，灵活性低 |

---

> 参考来源（网络受限无法访问，供后续核实）：
> - https://langchain-ai.github.io/langgraph/
> - https://langchain-ai.github.io/langgraph/concepts/
> - https://github.com/langchain-ai/langgraph
