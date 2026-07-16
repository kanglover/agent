# AI Agent 学习仓库

> 从零开始学习 AI Agent 开发的过程记录——包含阅读笔记、代码实验和动手项目。

---

## 这个仓库是什么？

一个边学边做的学习空间。内容围绕一个核心问题展开：

**如何让 AI 自主使用工具、完成多步骤任务？**

从读 Anthropic / OpenAI 的工程博客，到手写 Agent Loop、设计工具层、构建上下文工程——把每个阶段的理解都沉淀下来。

---

## 目录结构

```
├── code/                    ← 所有代码
│   ├── min_agent/           ← 最小 Agent Loop（核心项目）
│   ├── context_builder/     ← 上下文工程实验
│   ├── tools_v1.py          ← 工具设计：基础版（对比）
│   ├── tools_v2.py          ← 工具设计：增强版（对比）
│   ├── calculator_agent.py  ← 入门示例
│   ├── todo_agent.py        ← 实用示例
│   └── tests/               ← 单元测试
│
└── notes/                   ← 阅读笔记与知识沉淀
```

---

## 核心项目

### 🔁 最小 Agent Loop（`code/min_agent/`）

手写 **observe → think → act → observe** 循环，不依赖任何 Agent 框架。

```
用户任务
  ↓
[THINK] Claude 决策：调用工具 or 直接回答？
  ↓ tool_use
[ACT]   执行工具（search_notes / write_summary）
  ↓
[OBSERVE] 结果喂回 Claude，继续循环
  ↓ end_turn
任务完成，写入 trace.jsonl
```

**关键设计：**
- 最大 5 步，防止无限循环
- 每步写 JSONL trace（含 token 费用估算）
- 出错不崩溃，返回结构化失败原因

```bash
cd code
export ANTHROPIC_API_KEY="sk-ant-..."
python -m min_agent.main
```

---

### 🧱 工具层设计对比（`tools_v1.py` vs `tools_v2.py`）

同一套工具，两种设计风格，直观对比「写给 API 的工具」和「写给 AI 的工具」的区别。

| | 基础版（v1） | 增强版（v2） |
|--|------------|------------|
| 返回格式 | 纯字符串 | 结构化 JSON |
| 出错时 | raise / 模糊文字 | 错误码 + `next_action` |
| 分页 | ❌ | ✅ `max_results` |
| 截断 | ❌ | ✅ `MAX_CHARS_PER_RESULT` |
| 重试 | ❌ | ✅ 带退避的重试 |
| 高风险标记 | ❌ | ✅ `requires_confirmation` |

---

### 📐 上下文工程（`code/context_builder/`）

把发给 AI 的上下文拆成 5 层，结构化管理：

| 层 | 名称 | 内容 |
|----|------|------|
| 1 | System | 角色定义、规则、工具列表 |
| 2 | Task | 当前任务描述与目标 |
| 3 | Memory | 跨任务持久化知识 |
| 4 | Evidence | 检索到的外部内容（自动压缩） |
| 5 | Trace | 最近的工具调用记录（自动压缩） |

长内容自动截断并生成可追溯 `ref:xxxxxxxx`，不把完整日志/网页/PDF 直接塞进模型。

```python
bundle = (
    ContextBuilder()
    .set_system(role="助手", instructions=["简洁中文"])
    .set_task(description="总结笔记", goal="输出摘要")
    .add_evidence(source="note.md", content="...很长的内容...")  # 自动压缩
    .build()
)
print(render(bundle))
```

---

## 运行测试

```bash
cd code
pip install anthropic
pytest tests/ -v
```

---

## 学习笔记（`notes/`）

阅读 Anthropic / OpenAI 工程博客后的整理：

- `agent-workflow-核心概念.md` — Workflow vs Agent 的本质区别
- `agent-openai实践指南.md` — OpenAI Agent 设计指南核心内容
- `context-engineering-上下文工程.md` — 上下文工程业界最佳实践
- `workflow-vs-agent-代码对比.md` — 用代码说明两者核心差异
- `writing-effective-tools-for-agents.md` — Anthropic 工具设计最佳实践

---

## 参考资料

- [Building Effective Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Writing Effective Tools for Agents — Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [A Practical Guide to Building AI Agents — OpenAI](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
