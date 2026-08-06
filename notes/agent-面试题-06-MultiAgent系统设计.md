# Multi-Agent 系统设计面试题精选（20 题）

> 涵盖基础概念、架构设计、工程实践、框架原理（AutoGen/CrewAI/LangGraph）、高级系统设计题。

---

## 一、基础概念（Q1–Q5）

**Q1. 什么是 Multi-Agent 系统？与单 Agent 有什么本质区别？**

Multi-Agent 系统是由多个具有独立推理能力的 Agent 组成的系统，每个 Agent 有自己的角色（role）、工具（tools）、记忆（memory）和目标（goal）。

单 Agent 串行执行所有步骤，上下文窗口容易撑爆；Multi-Agent 可以并行分工、隔离上下文、专业化处理。

核心区别：**分工协作 vs 单点全包**。

类比：一个全科医生 vs 专科医生团队会诊。

---

**Q2. Multi-Agent 系统的核心组件有哪些？**

1. **Orchestrator（编排器）**：负责任务分解、分配和汇总，不直接执行具体任务
2. **Sub-Agents（子 Agent）**：负责具体子任务执行，如搜索、代码执行、写作等
3. **Memory（记忆）**：分为短期（context window）、长期（向量数据库）、Entity Memory（实体记忆）
4. **Tools（工具）**：函数调用、API、数据库查询等
5. **Communication Channel（通信通道）**：Agent 间消息传递机制
6. **State（状态）**：记录任务执行的全局状态，如 LangGraph 中的 StateGraph

---

**Q3. 什么是 Orchestrator-Worker 模式？适用场景是什么？**

Orchestrator（主控 Agent）接收用户任务，拆解为子任务，动态分发给 Worker Agents，收集结果后汇总。

适用场景：任务可拆解为并行子任务，如：数据分析流水线、多步骤研究报告生成、软件开发中的需求→设计→编码→测试。

关键实现细节：
- Orchestrator 本身不执行工具，只做路由和聚合
- 通过结构化 JSON 或自然语言传递上下文

---

**Q4. 什么是 Supervisor 模式？与 Orchestrator 有何不同？**

Supervisor 模式中，Supervisor Agent 不仅分配任务，还**持续监控**每个 Worker 的状态，并在失败时介入重试或重新分配。

- Orchestrator 更像"项目经理"
- Supervisor 更像"质检主管"——关注执行质量和异常恢复

LangGraph 官方文档中明确区分了这两种模式，Supervisor 适合需要高可靠性的生产场景。

---

**Q5. Agent 的 ReAct 框架是什么？**

ReAct = **Reason + Act**，Agent 交替进行：
1. **Thought**（思考）：分析当前状态，决定下一步
2. **Action**（行动）：调用工具或执行操作
3. **Observation**（观察）：获取工具返回结果
4. 循环直到任务完成或达到最大步数

优点：可解释性强，便于调试；缺点：串行执行，效率较低。

---

## 二、架构设计（Q6–Q10）

**Q6. 如何设计一个能处理复杂研究任务的 Multi-Agent 系统？**

```
用户输入
   ↓
[Planner Agent] → 任务分解为 N 个子任务
   ↓ (并行)
[Search Agent] [Summarizer Agent] [Fact-Check Agent]
   ↓ (汇总)
[Writer Agent] → 生成最终报告
   ↓
[Critic Agent] → 质量审查 → 若不通过，回退给 Writer
```

关键设计决策：
- 是否并行执行取决于子任务是否有依赖关系（DAG 调度）
- Critic Agent 引入反馈循环（Reflection Loop），防止低质量输出
- 使用 checkpointing 保存中间状态，防止长任务失败后从头开始

---

**Q7. Multi-Agent 系统中如何处理 Agent 之间的状态共享？**

三种方案对比：

| 方案 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| 共享内存（如 Redis） | 实时一致性 | 并发写冲突 | 强一致性场景 |
| 消息队列（如 Kafka） | 解耦、异步 | 调试复杂 | 高吞吐异步任务 |
| 中心 State Object（LangGraph） | 简单、可追溯 | 单点瓶颈 | 单机编排 |

LangGraph 的做法：所有 Agent 操作同一个 `State` 字典，每个节点只读取需要的字段，只更新自己负责的字段，避免冲突。

---

**Q8. 如何设计 Agent 的记忆系统（Memory）？**

四层记忆架构：

1. **Sensory Memory（感知记忆）**：当前 prompt 的原始输入，极短暂
2. **Working Memory（工作记忆）**：context window 内的对话历史，当前任务状态
3. **Episodic Memory（情节记忆）**：历史交互摘要，存向量数据库（如 Chroma、Pinecone）
4. **Semantic Memory（语义记忆）**：领域知识库，RAG 检索增强

关键工程问题：
- 如何处理 context 超限？→ 滑动窗口 + 摘要压缩（Summary Buffer Memory）
- 如何避免记忆污染？→ Agent 隔离各自的 working memory，只通过 shared state 交互

---

**Q9. CrewAI 和 AutoGen 的核心架构差异是什么？**

| 维度 | CrewAI | AutoGen（微软） |
|------|--------|----------------|
| 核心抽象 | Crew + Agent + Task + Tool | ConversableAgent + GroupChat |
| 协作模式 | 基于角色和任务流 | 基于对话（conversational） |
| 任务分配 | 显式指定 Agent 负责哪个 Task | 由 GroupChatManager 动态决定 |
| 代码执行 | 需配置 Code Tool | 内置代码执行沙箱 |
| 适用场景 | 流程化、角色明确的任务 | 探索性、对话驱动的协作 |

两者都支持 Human-in-the-loop（人工介入），但实现方式不同。

---

**Q10. 什么是 Agentic Loop？如何防止无限循环？**

Agentic Loop 指 Agent 不断执行 Think-Act-Observe 循环直到完成任务。

防止无限循环的机制：
1. **Max iterations**：硬限制最大步数（如 AutoGen 默认 10 次）
2. **Task completion signal**：要求 Agent 输出特定 `TERMINATE` 信号
3. **Convergence check**：检测 Agent 是否在原地打转（连续 N 次 action 相同）
4. **Cost budget**：按 token 消耗设置预算上限
5. **Human interrupt**：设置 checkpoint，允许人工介入

---

## 三、工程实践（Q11–Q15）

**Q11. 如何设计 Multi-Agent 系统的错误处理和容错机制？**

- **Agent 级别**：每个 Agent 的工具调用加 try-catch，失败时返回结构化错误信息而非抛异常
- **Orchestrator 级别**：
  - 子任务失败 → 自动重试 3 次 → 降级（使用备用 Agent）→ 上报人工
  - 关键路径失败 → 整体任务标记失败并保存 checkpoint
- **幂等性设计**：同一子任务多次执行结果一致，防止重试产生副作用
- **Dead Letter Queue**：无法处理的消息进入死信队列，供事后分析

---

**Q12. 如何评估 Multi-Agent 系统的质量？**

四个维度：

1. **Task Completion Rate（任务完成率）**：最终任务成功率
2. **Latency（延迟）**：端到端耗时，需区分并行 vs 串行任务
3. **Token Cost（成本）**：每次任务消耗的 token 总量，Multi-Agent 因多次 LLM 调用成本较高
4. **Accuracy / Quality**：引入 Evaluator Agent 或人工标注，对输出质量打分

评估方法：
- 构建 Golden Dataset：准备标准问答对，自动化 regression testing
- LLM-as-Judge：使用更强的模型评估较弱模型的输出
- Trajectory Evaluation：不只评估最终结果，还评估 Agent 的决策过程是否合理

---

**Q13. Multi-Agent 系统的安全性如何保障？（Prompt Injection 防御）**

威胁模型：恶意输入试图操控 Agent 调用危险工具（如删除文件、发送敏感数据）。

防御措施：
1. **工具权限最小化**：每个 Agent 只赋予完成任务所需的最小工具集
2. **输入验证**：对 Agent 接收的 tool 参数做 schema 校验
3. **Sandboxing**：代码执行 Agent 在隔离容器中运行（如 Docker）
4. **Human confirmation**：危险操作（写文件、调 API）需人工确认
5. **输出过滤**：Orchestrator 对 sub-agent 返回内容做敏感词/结构检查

Anthropic 官方建议：对于权限边界，应假设"任何 Agent 都可能被 compromise"，做 defense in depth。

---

**Q14. 如何实现 Agent 的并行执行？**

- **LangGraph**：使用 `send()` API 将同一节点并行 dispatch，结果通过 `Annotated[list, operator.add]` 聚合
- **AutoGen**：`GroupChat` 支持多 Agent 并发发言，但默认是串行轮询
- **Python 原生**：`asyncio.gather()` 并发调用多个 async Agent

关键注意点：
- 并行 Agent 不应共享可变状态（race condition）
- 合并结果时需定义清晰的 merge 策略（取最佳 / 投票 / 全部保留）
- 注意 LLM Provider 的 rate limit，并行请求过多会触发限流

---

**Q15. 什么是 Human-in-the-Loop（HITL）？如何在系统中实现？**

HITL 是指在 Agent 执行过程中，允许人类在特定节点介入、审核或修改决策。

实现方式：
1. **LangGraph interrupt**：在 graph 节点前设置 `interrupt_before=["node_name"]`，执行暂停等待人工输入
2. **AutoGen Human Proxy Agent**：配置 `human_input_mode="ALWAYS"` 或 `"TERMINATE"`
3. **Webhook/callback**：Agent 执行到关键步骤时调用外部 webhook，等待回调信号

适用场景：审批流程、危险操作确认、创意内容人工审核。

---

## 四、框架原理（Q16–Q18）

**Q16. LangGraph 相比 LangChain 的 AgentExecutor 有何优势？**

LangChain AgentExecutor 是线性的 ReAct 循环，不支持分支、循环、并行；LangGraph 将 Agent 逻辑建模为**有向图（DAG 或含环）**：
- 节点（Node）= 处理步骤（LLM 调用 / 工具调用）
- 边（Edge）= 条件路由（根据输出决定下一步走哪条路）

优势：
1. 支持**循环**（Agent 可以反复尝试）
2. 支持**条件分支**（根据置信度决定是否人工介入）
3. 内置**状态持久化**（checkpointing，支持断点续跑）
4. 便于**调试和可视化**（图结构清晰）

---

**Q17. AutoGen 的 GroupChat 是如何决定下一个发言 Agent 的？**

默认策略：`auto`，由 `GroupChatManager`（也是一个 LLM Agent）根据对话历史决定谁应该发言。

其他策略：
- `round_robin`：轮流发言
- `random`：随机选择
- 自定义 `speaker_selection_func`：完全自定义选择逻辑

注意点：`auto` 模式下 GroupChatManager 本身也消耗 token，且选择逻辑不可控；生产中建议使用自定义函数。

---

**Q18. 什么是 Reflection Agent？如何实现自我改进？**

Reflection Agent 在完成任务后，会对自己的输出进行批评和改进，形成迭代循环。

经典实现（Reflexion 论文，2023）：
```
初始输出 → Critic（批评，指出错误） → Reviser（修改） → 再次 Critic → ... → 满足质量标准
```

在 Multi-Agent 中：Critic 和 Reviser 是两个独立 Agent，避免"自己批评自己"时的偏见。

关键超参：最大反思轮数（防止无限循环）、质量阈值（何时停止反思）。

---

## 五、高级系统设计题（Q19–Q20）

**Q19. 设计一个能自主完成软件需求到代码部署的 Multi-Agent 系统**

参考架构（Devin/SWE-Agent 类型）：

```
[BA Agent]        ← 解析需求，产出 PRD
    ↓
[Architect Agent] ← 设计架构，产出技术方案
    ↓
[Coder Agent]     ← 编写代码（可并行多个模块）
    ↓
[Tester Agent]    ← 运行测试，产出 test report
    ↓
[Debugger Agent]  ← 修复失败的测试（循环直到通过）
    ↓
[DevOps Agent]    ← 部署到环境，监控结果
    ↓
[Reviewer Agent]  ← 代码审查，可回退到 Coder
```

关键考点：
- 每个 Agent 拥有最小化工具集
- 使用 git 作为共享工作区（Agent 之间通过 commit 传递代码）
- 全程记录 execution trace，便于失败后回溯

---

**Q20. 如何解决 Multi-Agent 系统中的"上下文爆炸"问题？**

问题根源：每个 Agent 的对话历史不断增长，最终超过 context window。

解决方案：
1. **摘要压缩**：每 N 轮对话压缩一次，只保留摘要
2. **滚动窗口**：只保留最近 K 轮对话
3. **Agent 隔离**：每个 sub-agent 有独立 context，Orchestrator 只传递结构化的任务和结果，不传递完整历史
4. **外部记忆卸载**：将中间结果存入向量数据库，Agent 按需检索而非全量载入
5. **任务分解细化**：将大任务拆成更小单元，每个子任务 context 更少
