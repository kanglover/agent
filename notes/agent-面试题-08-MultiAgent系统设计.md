# Multi-Agent 系统设计面试题精选

> 来源：模型训练知识（截止 2026 年 1 月）+ 仓库已有笔记
> 参考：AutoGen、CrewAI、LangGraph 官方文档，Reflexion 论文（2023）

---

## 一、基础概念

**Q1. 什么是 Multi-Agent 系统？和单 Agent 的本质区别是什么？**

答题要点：
- Multi-Agent System (MAS)：多个自主 Agent 各有独立角色、工具、记忆和目标，协作完成单个 Agent 无法独立完成的任务
- 单 Agent 串行执行、受限于单个上下文窗口
- Multi-Agent 支持：**并行工作、上下文隔离、专业化分工**
- 类比：全科医生（单 Agent）vs 专科医生会诊团队（Multi-Agent）

---

**Q2. Multi-Agent 系统面临哪些核心挑战？**

答题要点：
- **错误传播**：上游错误被下游放大 → 在关键节点加入 Validation Agent
- **延迟累积**：每次 LLM 调用 1-10s → 最大化并行执行，路由用更快的模型
- **调试困难**：使用 LangSmith/Langfuse 可观测平台，每个 Agent 加 trace ID
- **成本控制**：简单任务用更便宜的模型，智能缓存，设置 Token 预算上限
- **死锁/循环**：任务依赖设计为严格 DAG，设置超时，监控心跳信号

---

**Q3. 什么时候用 Multi-Agent，什么时候用单 Agent？**

答题要点：

用**单 Agent** 的场景：
- 任务在单个上下文窗口内可完成
- 工作流线性无分支
- 延迟预算紧张（每多一个 Agent 增加 1-10s）
- 成本约束严格

用 **Multi-Agent** 的场景（必要条件，不是"nice to have"）：
1. 任务有多个**可并行**的独立子任务
2. 不同子任务需要**根本不同的专业知识**（不同系统提示、工具、甚至不同底层模型）
3. 需要内置的**制衡机制**——一个 Agent 的输出需要另一个独立审核
4. 任务的总信息量（文档、代码、数据）**超出单一上下文窗口**
5. 可靠性要求需要冗余——子 Agent 失败可以重试或替换，无需重启整个任务

黄金法则：如果能用一句话描述任务、有清晰输出格式、塞得进上下文，就是单 Agent 问题。

---

**Q4. 如何判断任务分解粒度是否合适？**

答题要点：
- **过粗信号**：单个子任务仍超出合理 LLM 处理范围；需要多个专业知识领域
- **过细信号**：协调开销超过计算开销；依赖链消除并行收益；Orchestrator 管理 10+ Agent
- **黄金标准**：每个子任务可用一句话描述清楚输入、输出和成功标准

---

## 二、架构模式

**Q5. Orchestrator-Worker 模式是什么？**

答题要点：
- Orchestrator 接收用户任务 → 拆分子任务 → 动态派发给 Worker Agent → 收集聚合结果
- **Orchestrator 自己不执行工具**，只路由和聚合
- 通过结构化 JSON 或自然语言传递上下文
- 适用：可并行分解的任务（数据分析流水线、多步骤研究、软件开发流水线）

---

**Q6. Supervisor 模式与 Orchestrator 有何不同？**

答题要点：
- Supervisor 持续监控每个 Worker 的状态，遇到失败时介入重试或重新分配
- Orchestrator 像"项目经理"；Supervisor 像"质检员"
- LangGraph 官方文档明确区分这两种模式；Supervisor 适合高可靠性生产场景

---

**Q7. Reflection Agent 是什么？如何避免自我审查偏差？**

答题要点：
- Reflection 循环：Agent 输出 → Critic 评估并给出结构化反馈 → Reviser 修改 → 循环直到质量达标
- 基于 Reflexion 论文（Shinn et al., 2023）
- **避免自我审查偏差**：Critic 和 Reviser 是**两个独立 Agent**，有各自的系统提示，Critic 采用怀疑论人设
- 关键超参数：`max_reflection_rounds`（防止无限循环）、`quality_threshold`（何时停止）
- LangGraph 中天然表示为含环的图：`[writer] → [critic] → conditional_edge → [writer]（循环）或 END`

---

## 三、框架对比

**Q8. LangGraph、AutoGen、CrewAI 的核心区别？**

| 框架 | 核心抽象 | 优势 | 劣势 |
|------|---------|------|------|
| **LangGraph** | StateGraph（有状态图） | 高灵活性、内置持久化与中断恢复、生产就绪 | 学习曲线较陡 |
| **AutoGen** | 对话式 ConversableAgent | 入门容易、代码生成强、GroupChat 灵活 | 状态散落在对话历史，流程精确控制难 |
| **CrewAI** | 角色（Agent）+ 任务（Task） | 高层 API 简洁、角色扮演直觉化 | 持久化/中断需自行实现，灵活性低 |

---

**Q9. AutoGen 的 GroupChat 如何决定下一个发言的 Agent？**

答题要点：
- 默认策略 `auto`：`GroupChatManager`（本身是一个 LLM Agent）根据对话历史决定
- 其他策略：`round_robin`（轮流）、`random`（随机）、自定义 `speaker_selection_func`
- 生产建议：`auto` 模式中 GroupChatManager 自身消耗 Token 且选择逻辑不可控，**生产环境推荐自定义选择函数**

---

**Q10. CrewAI 的核心抽象是什么？**

答题要点：
- **Agent**：有 `role`、`goal`、`backstory`、可选 `tools` 和底层 LLM；role/backstory 注入系统提示
- **Task**：有 `description`、`expected_output`、指定的 `agent`、可选 `context`（前序任务输出）
- **Crew**：顶层容器，包含 agents 列表、tasks 列表和 `process` 设置
- **Process 类型**：
  - `Process.sequential`：任务顺序执行，每个任务输出作为下一个输入
  - `Process.hierarchical`：Manager LLM 动态分配任务给 Agents 并验证结果

---

## 四、Agent 记忆设计

**Q11. 如何设计 Agent 的记忆系统？**

答题要点（四层架构）：

| 层次 | 内容 | 特点 |
|------|------|------|
| 感知记忆 | 当前提示词中的原始输入 | 极短暂 |
| 工作记忆 | 上下文窗口内的对话历史 + 当前任务状态 | 受窗口限制 |
| 情节记忆 | 历史交互摘要，存储在向量 DB | 长期持久化 |
| 语义记忆 | 领域知识库，RAG 检索 | 静态知识 |

关键工程问题：
- **上下文溢出**：滑动窗口 + 摘要压缩（Summary Buffer Memory）
- **记忆污染**：每个 Agent 有独立工作记忆，仅通过共享 State 交互

---

## 五、故障处理与可靠性

**Q12. 如何设计 Multi-Agent 系统的错误处理和容错？**

答题要点：
- **Agent 级别**：每个工具调用加 try-catch，失败时返回结构化错误信息（不要抛异常）
- **Orchestrator 级别**：子任务失败 → 自动重试 3 次 → 降级（使用备用 Agent）→ 上报人工；关键路径失败 → 标记任务失败并保存 Checkpoint
- **幂等性设计**：相同子任务多次执行产生相同结果，防止重试副作用
- **死信队列（DLQ）**：无法处理的消息进入 DLQ，供事后分析

---

**Q13. 如何防止 Agent 陷入无限循环？**

答题要点：
1. `max_iterations`：硬性限制最大步骤数（AutoGen 默认 10）
2. 任务完成信号：要求 Agent 输出 `TERMINATE` 信号
3. 收敛检测：检测 Agent 是否在原地打转（连续 N 次相同动作）
4. Token 预算：按 Token 消耗设置预算上限
5. 人工中断：设置 Checkpoint，允许人工介入

---

## 六、评估与可观测性

**Q14. 如何评估 Multi-Agent 系统的质量？**

答题要点（4 个维度）：
1. **任务完成率**：最终任务成功率
2. **延迟**：端到端时间；区分 P50/P95/P99，并行 vs 串行任务
3. **Token 成本**：每个任务的总 Token 数；Multi-Agent 比单 Agent 贵 5-20 倍（多次 LLM 调用）
4. **准确性/质量**：用 Evaluator Agent 或人工标注评分输出质量

评估方法：
- **Golden Dataset**：准备标准问答对，自动化回归测试
- **LLM-as-Judge**：用更强的模型评估较弱模型的输出
- **轨迹评估（Trajectory Evaluation）**：不只评估最终结果，还评估 Agent 的决策路径是否合理

---

**Q15. Multi-Agent 系统如何实现可观测性？**

答题要点：
- 工具链：LangSmith（LangChain/LangGraph 生态）、Langfuse、自定义 trace ID
- 每个 LLM 调用和工具调用都应记录结构化事件：Agent ID、输入、输出、延迟、Token 数、错误标志
- 分布式追踪：一个用户请求产生的所有 Agent 调用共享同一个 trace ID
- 关键指标监控：检索延迟、LLM 延迟、答案质量（自动 RAGAS 指标）、embedding 漂移

---

## 七、安全与权限

**Q16. Multi-Agent 系统中如何防御 Prompt Injection？**

答题要点：
1. **最小权限工具集**：每个 Agent 只获得完成其任务所需的最小工具集
2. **输入验证**：对 Agent 接收的工具参数做 schema 验证
3. **沙箱隔离**：执行代码的 Agent 在隔离容器（Docker）内运行
4. **人工确认**：危险操作（写文件、调用 API）需要人工审批
5. **输出过滤**：Orchestrator 检查子 Agent 返回内容是否含敏感内容/异常结构

Anthropic 建议：对权限边界，假设"任何 Agent 都可能被攻陷"，实施纵深防御。

---

**Q17. 如何解决 Multi-Agent 系统中的"上下文爆炸"问题？**

答题要点：
1. 摘要压缩：每 N 轮压缩一次，只保留摘要
2. 滑动窗口：只保留最近 K 轮
3. Agent 隔离：每个子 Agent 有独立上下文；Orchestrator 只传递结构化任务和结果，不传全量历史
4. 外部记忆卸载：中间结果存向量 DB，Agent 按需检索而不是全量加载
5. 更细粒度任务分解：把大任务拆成每个子任务上下文更少的小单元
