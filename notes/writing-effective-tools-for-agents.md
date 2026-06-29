# 给 AI Agent 写高质量工具

> 沉淀时间：2026-06-29
> 来源：https://www.anthropic.com/engineering/writing-tools-for-agents
> 作者：Anthropic 工程团队（Ken Aizawa 等）

---

## 一句话总结

工具不是给代码看的，是给 AI 看的。好工具要少而精、名字清晰、返回有意义的内容、控制好 Token 用量，然后用评测 + AI 持续优化。

---

## 核心思维转变

**普通函数**：写给其他代码调用，行为 100% 可预测。

**Agent 工具**：写给 AI 调用，AI 自己决定要不要调、怎么调、调几次。

> 不能把 API 包一层就叫工具。要专门为 AI 的「认知方式」而设计。

---

## 如何写工具？三步走

### 第一步：先做原型，自己试用

- 快速搭一个工具，接进 Claude 或 MCP 测试
- 亲手试用，找问题、找感觉
- 收集真实用户反馈，建立对场景的直觉

### 第二步：建评估体系，量化效果

> 没有评估，就不知道改哪里、改了有没有用。

**好的评测任务 vs 差的评测任务：**

| ✅ 好的 | ❌ 差的 |
|--------|--------|
| "帮我约 Jane 下周开会，附上上次会议记录，订好会议室" | "约 jane@acme.corp 下周开会" |
| "客户 9182 被重复扣款三次，找相关日志，看其他用户是否受影响" | "搜索 payment_log 里的 customer_id=9182" |

**好的评测任务标准：**
- 贴近真实场景，不用"沙盒"假数据
- 需要多步工具调用
- 有明确可验证的结果

**除了准确率，还要追踪：**
- 每个任务的运行时间
- 工具调用总次数
- Token 消耗量
- 工具报错次数

### 第三步：让 Claude 帮你优化工具

把评测对话日志丢给 Claude Code，让它分析哪里有问题、怎么改进。可以循环执行：**评测 → 分析 → 改进 → 再评测**，直到性能达标。

> Anthropic 内部用这个方法，让 Claude 在 SWE-bench 评测上达到了当时的世界最高分。

---

## 五条核心原则

### 原则 1：选对工具——少而精

工具越多不代表越好，工具重叠或职责模糊会让 AI 困惑。

**设计原则：** 工具要让 AI 像人一样解决问题——直接跳到相关内容，而不是从头翻一遍。

| ❌ 不好的设计 | ✅ 好的设计 |
|-------------|------------|
| `list_users` + `list_events` + `create_event` | `schedule_event`（一个工具搞定安排会议） |
| `read_logs` | `search_logs`（只返回相关日志行 + 上下文） |
| `get_customer_by_id` + `list_transactions` + `list_notes` | `get_customer_context`（一次返回所有相关信息） |

---

### 原则 2：给工具取好名字（命名空间）

工具变多后，AI 容易搞混。用前缀分组，让 AI 一眼知道该用哪个：

```
asana_search              jira_search
asana_projects_search     asana_users_search
asana_tasks_create        asana_tasks_update
```

**注意：** 前缀还是后缀命名对效果有影响，建议用评测来决定哪种更适合你的场景。

---

### 原则 3：返回有意义的内容

工具返回给 AI 的内容要**语义明确、可读性强**：

| ❌ 不好 | ✅ 好 |
|--------|-------|
| `uuid: a3f9-bc12-...` | `name: Jane Smith` |
| `256px_image_url` | `image_url` |
| `mime_type: image/jpeg` | `file_type: JPEG` |

**实用技巧：加 `response_format` 参数**

让 AI 自己选返回详细版还是简洁版：

```python
enum ResponseFormat:
    DETAILED = "detailed"   # 包含所有字段（含技术 ID，用于后续工具调用）
    CONCISE = "concise"     # 只含核心内容，约节省 2/3 Token
```

---

### 原则 4：控制返回内容的量（Token 效率）

工具返回太多内容会撑爆 AI 的上下文，要做：
- **分页**（pagination）
- **范围过滤**（filtering）
- **截断 + 提示**（告诉 AI 还有更多内容，引导它缩小搜索范围）

Claude Code 默认限制工具返回最多 **25,000 Token**。

**错误信息也要写好，告诉 AI 怎么修：**

| ❌ 糟糕的错误 | ✅ 有帮助的错误 |
|-------------|--------------|
| `Error: 400 Bad Request` | `日期格式错误，请使用 YYYY-MM-DD，例如：2025-01-15` |
| `Invalid parameter` | `user_id 不能为空，请先调用 search_users 获取用户 ID` |

---

### 原则 5：认真写工具描述（Prompt 工程）

工具描述会被加进 AI 的上下文，**直接影响 AI 懂不懂怎么用**。

**写工具描述的心态：像给新员工写操作手册**

- 把你「默认知道」的背景知识写出来
- 参数名要无歧义（`user_id` 而不是 `user`）
- 说清楚输入输出格式，给出例子
- 说明工具之间的关联关系

> Anthropic 仅靠精细调整工具描述，就让 Claude 在 SWE-bench 评测上达到当时最高分。

---

## 和之前文章的对比

| 维度 | 本文（工具设计） | Anthropic Agent 指南 | OpenAI Agent 指南 |
|------|----------------|---------------------|------------------|
| 关注点 | 工具本身怎么写 | Agent 架构怎么设计 | Agent 组件怎么搭 |
| 核心主张 | 工具要为 AI 认知方式设计 | 简单优先，组合模式 | 单 Agent 优先，逐步升级 |
| 工具数量 | 少而精，不要包 API | 工具要文档化 | 三类工具（Data/Action/Orchestration） |
| 共同强调 | 评测驱动改进 | 测试是关键 | 先建基准再优化 |

---

## 适用场景

- 为自己的 Agent 系统开发 MCP 工具时
- 发现 Agent 频繁调用错误工具时
- 工具返回内容太多导致 Agent 效果差时
- 想用 Claude Code 自动优化工具描述时

---

*相关概念：MCP、Tool Use、Tool Description、Token Efficiency、Evaluation、Prompt Engineering*

*相关文档：[[agent-workflow-核心概念]] [[agent-openai实践指南]] [[context-engineering-上下文工程]] [[workflow-vs-agent-代码对比]]*
