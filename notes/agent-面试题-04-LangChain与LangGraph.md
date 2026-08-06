# LangChain & LangGraph 面试题精选（25 题）

> 涵盖 LCEL、RAG 链构建、LangGraph StateGraph、Multi-Agent、ReAct、Human-in-the-Loop 等核心考点。

---

## 一、LangChain 基础与 LCEL（Q1–Q9）

**Q1. LangChain 的包结构分哪几层？各层的职责是什么？**

共五层：
- `langchain-core`：核心抽象（Runnable、BaseMessage 等），几乎无外部依赖
- `langchain`：高级组件（Agent、Chain、Memory 等）
- `langchain-community`：第三方集成（数百种 LLM、工具、向量库）
- `langgraph`：有状态图式 Agent 编排（独立包）
- `langsmith`：观测与评估 SaaS 平台

---

**Q2. 什么是 Runnable 协议？它定义了哪几个核心方法？**

`Runnable` 是 LangChain 中所有组件的统一基类，任何实现它的组件都可以被组合。核心方法：
- `invoke(input)` — 同步执行
- `ainvoke(input)` — 异步执行
- `stream(input)` — 同步流式输出
- `astream(input)` — 异步流式输出
- `batch(inputs)` — 批量并行执行
- `__or__(other)` — 支持 `|` 操作符，返回 `RunnableSequence`

---

**Q3. 什么是 LCEL？核心语法是什么？相比旧版 LLMChain 有何优势？**

LCEL（LangChain Expression Language）是声明式组合语言，用 `|` 把 Runnable 组件串联成管道：

```python
chain = prompt | model | StrOutputParser()
result = chain.invoke({"concept": "向量数据库"})
```

相比旧版 `LLMChain`：
- 旧版继承式 API，难以组合；新版组合式，极度灵活
- LCEL 自动支持流式、异步、批处理，无需额外代码
- 所有子链天然可观测（LangSmith 追踪）

---

**Q4. `RunnableParallel` 和 `RunnableLambda` 分别用于什么场景？**

- `RunnableParallel`：让多个子链并行执行，结果合并为字典。适用于"同一输入需要多个不同分析"（如同时做摘要、提关键词、情感分析）
- `RunnableLambda`：将普通 Python 函数包装为 Runnable，嵌入 LCEL 管道，适用于自定义预处理或后处理

---

**Q5. `RunnableSequence` 内部如何实现 `|` 链式调用？流式输出有什么特殊处理？**

`|` 调用 `__or__` 返回 `RunnableSequence(steps=[A, B, C])`。执行时：
- `invoke`：依次调用每个 step 的 `invoke`，上一个输出作为下一个的输入
- `stream`：只有最后一个 step 调用 `stream`，前面 step 仍用 `invoke`（确保中间步骤完成）
- 链式 `|` 会扁平化为单个 RunnableSequence，避免嵌套

---

**Q6. LangChain 的记忆系统如何工作？`RunnableWithMessageHistory` 的作用是什么？**

通过 `RunnableWithMessageHistory` 将历史管理与 Chain 解耦：
- 传入 `get_session_history(session_id)` 函数，按 session_id 隔离不同用户历史
- 每次 invoke 时自动读取历史、注入 prompt 的 `MessagesPlaceholder`，结束后自动追加新消息
- 支持 `InMemoryChatMessageHistory`（开发用）、`RedisChatMessageHistory`、`SQLChatMessageHistory`（生产用）

---

**Q7. 如何用 LCEL 构建一个 RAG 链？`RunnablePassthrough` 的作用是什么？**

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

`RunnablePassthrough` 把输入原样传递，不做任何变换。在 RAG 中它将用户问题"直接透传"到 prompt 的 `{question}` 位置，同时 retriever 并行检索相关文档填入 `{context}`。

---

**Q8. RecursiveCharacterTextSplitter 的 `chunk_overlap` 参数有什么用？**

相邻文本块之间的重叠字符数（如 200）。作用是保证跨块的语义连续性——如果一个句子或概念横跨两块边界，重叠区域确保两块都包含这段上下文，避免关键信息因分块而丢失。

---

**Q9. 向量检索的 `search_type="mmr"` 是什么意思？相比默认相似度检索有何优势？**

MMR（Maximum Marginal Relevance，最大边际相关性）在检索时同时考虑**相关性**和**多样性**。先取 `fetch_k` 个候选，再用 MMR 从中选 `k` 个，避免返回的文档过于相似（如同一段话被分成多块）。`lambda_mult` 参数调节权重（0 偏多样性，1 偏相关性）。

---

## 二、LangGraph 核心概念（Q10–Q18）

**Q10. LangGraph 与 LangChain LCEL 的核心区别是什么？什么时候应该选 LangGraph？**

| 维度 | LangChain LCEL | LangGraph |
|------|---------------|-----------|
| 执行模式 | 线性 DAG，不支持循环 | 有环图，支持循环 |
| 状态管理 | 无内建状态 | 中心化 State 对象 |
| 条件分支 | 有限 | 完整支持 |
| 持久化 | 需外部实现 | 内建 Checkpointer |
| 人工介入 | 不支持 | 原生支持 interrupt |

选 LangGraph 的时机：Agent 需要循环决策、需要持久化、需要 Human-in-the-loop、多 Agent 协作。

---

**Q11. LangGraph 中 State 是什么？为什么用 `Annotated[list, add_messages]` 而不是普通 `list`？**

State 是一个 `TypedDict`（或 Pydantic 模型），是整个图的共享数据容器，每个节点读取 State、返回 State 的局部更新。

`Annotated[list[BaseMessage], add_messages]` 中的 `add_messages` 是一个 **Reducer 函数**。没有 Reducer 时，节点的返回值会**直接覆盖**对应字段；有 Reducer 时，LangGraph 调用 Reducer 将新值**合并**到现有值（`add_messages` 的行为是追加，而非替换）。这样历史消息不会丢失。

---

**Q12. LangGraph 节点（Node）函数的输入和输出格式是什么？**

节点函数签名为 `(state: StateType) -> dict`：
- 输入：完整的当前 State 对象
- 输出：仅包含**需要更新的字段**的字典（不需要返回完整 State）
- LangGraph 内部用 `_apply_updates` 将返回字典合并（或通过 Reducer 规约）到当前 State

---

**Q13. `add_conditional_edges` 的第二个参数（条件函数）返回值类型是什么？第三个参数 mapping 的作用？**

- 条件函数接收 State，返回一个**字符串**（表示下一步的路由）
- mapping 是 `{条件函数返回值: 下一节点名}` 的字典，将字符串映射到实际节点名（或 `END`）

```python
graph.add_conditional_edges(
    "llm",
    should_continue,
    {"tools": "tools", END: END}
)
```

mapping 可省略（直接用条件函数返回值作为节点名），但显式 mapping 更清晰可维护。

---

**Q14. `StateGraph.compile()` 做了什么？编译后的 graph 和 workflow 有何区别？**

`compile()` 主要做三件事：
1. 验证图的合法性（检查节点、边是否完整，是否有孤立节点）
2. 将构建好的图结构转化为可执行的 `CompiledGraph` 对象
3. 注入 Checkpointer（如果指定）

编译后得到 `CompiledGraph`，实现了 `Runnable` 协议，可调用 `invoke`、`stream`、`ainvoke`。`workflow`（`StateGraph` 实例）只是构建器，不能直接运行。

---

**Q15. 什么是 LangGraph 的 Checkpointer？它支持哪几种后端？**

Checkpointer 是 LangGraph 持久化能力的核心，在每个节点执行后**自动保存状态快照**。支持：
- `MemorySaver`：内存存储，开发测试用，进程重启后丢失
- `SqliteSaver`：SQLite 文件，本地持久化
- `PostgresSaver`：PostgreSQL，生产级持久化

使用时在 `compile(checkpointer=xxx)` 传入，运行时通过 `config={"configurable": {"thread_id": "xxx"}}` 指定会话 ID，同一 thread_id 的多次调用共享状态。

---

**Q16. LangGraph 的"时间旅行"（Time Travel）功能是什么？如何实现？**

时间旅行允许查看和回滚到历史任意状态快照，用于调试和修复：

```python
# 查看所有历史快照
history = list(graph.get_state_history(config))

# 回滚到某个历史点重新执行
target = history[3]
graph.update_state(config, target.values, as_node="agent")
new_result = graph.invoke(None, config=config)
```

前提：必须有 Checkpointer，每步快照才会被保存。

---

**Q17. Human-in-the-Loop 在 LangGraph 中如何实现？`interrupt` 和 `interrupt_before` 有什么区别？**

- `interrupt(payload)`：在节点函数内部调用，动态暂停执行，`payload` 是传给人类的信息
- `interrupt_before=["node_name"]`：在 `compile()` 时指定，在进入该节点**之前**自动暂停

恢复执行：`graph.invoke(Command(resume=human_response), config=config)`，必须配合 Checkpointer 使用。

---

**Q18. LangGraph 中的 `ToolNode` 是什么？相比手动实现工具执行节点有何优势？**

`ToolNode`（`from langgraph.prebuilt import ToolNode`）接受工具列表，自动：
- 遍历 state 中最后一条消息的 `tool_calls`
- 逐一（或并行）执行对应工具
- 将 `ToolMessage` 结果追加到 messages

相比手动实现，避免了样板代码，且对错误处理、工具 ID 匹配等边界情况已有内置处理。

---

## 三、ReAct 与 Multi-Agent（Q19–Q21）

**Q19. 什么是 ReAct 模式？在 LangGraph 中其图结构如何？**

ReAct（Reason + Act）是最主流的 Agent 模式，循环执行"思考 → 行动 → 观察"直到任务完成。LangGraph 图结构：

```
START → [agent]
[agent] —— 有 tool_calls ——→ [tools] → [agent]（循环）
[agent] —— 无 tool_calls ——→ END
```

`create_react_agent` 是 LangGraph 提供的快捷方式，一行代码等价于完整手动实现。

---

**Q20. Supervisor 多 Agent 模式的核心思想是什么？如何用 LangGraph 实现？**

Supervisor 是"主 LLM 决策 + 多专员执行"的协作模式：
- Supervisor 节点：接收当前状态，输出下一步由哪个 Agent 执行（或 FINISH）
- 各专员节点：执行特定领域任务后，结果返回 Supervisor 再次评估
- 路由逻辑：`add_conditional_edges("supervisor", route_fn, {"researcher": "researcher", ..., "FINISH": END})`
- 所有专员节点执行完后，边指回 supervisor（`add_edge(agent, "supervisor")`）

---

**Q21. LangGraph 如何支持嵌套 Agent（子图）？**

编译好的子图（`compiled_subgraph`）可以直接作为父图的一个节点：

```python
main_graph.add_node("research", compiled_research_subgraph)
```

子图作为节点时，父图的 State 字段必须和子图的 State 字段兼容（或通过 input/output mapping 转换）。

---

## 四、框架对比与生产实践（Q22–Q25）

**Q22. LangGraph、AutoGen、CrewAI 三个框架的核心抽象和优劣势对比？**

| 框架 | 核心抽象 | 优势 | 劣势 |
|------|---------|------|------|
| LangGraph | 有状态图（StateGraph） | 极高灵活性、内建持久化与中断恢复、生产就绪 | 学习曲线较陡 |
| AutoGen | 对话式 Agent | 上手容易，擅长代码生成任务 | 状态分散于对话历史，难以精确控制流 |
| CrewAI | 角色（Role）+ 任务（Task） | 高层 API 简洁，角色扮演协作直观 | 持久化/中断需自行实现，灵活性低 |

---

**Q23. 什么情况下应该用 LCEL，什么情况下应该用 LangGraph？**

- **用 LCEL**：简单线性流程（RAG、摘要、分类、格式转换）；流程固定、不需要循环；无状态任务
- **用 LangGraph**：Agent 需要自主循环决策；需要条件分支；需要任务中断/恢复；多 Agent 协作；需要历史状态回放（时间旅行）

---

**Q24. LangSmith 的作用是什么？如何在 LangChain 项目中启用追踪？**

LangSmith 是可观测性平台，记录每次 LLM 调用、工具调用、chain 执行的详细信息（输入输出、耗时、token 用量）。启用方式：

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
```

设置环境变量后，所有 LangChain/LangGraph 的调用自动上报，无需修改代码。

---

**Q25. `stream_mode` 参数在 LangGraph `graph.stream()` 中有哪几种取值？各自的输出格式是什么？**

- `"values"`：每次节点执行后，输出**完整的当前 State**（最常用，适合展示最终消息流）
- `"updates"`：每次节点执行后，只输出该节点**返回的更新部分**（字典）
- `"messages"`：流式输出 LLM 的 **token 级别消息增量**（适合实时打字效果）
