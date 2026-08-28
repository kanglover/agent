# AI 示意代码

这里是我学习 AI Agent 开发过程中写的一批示意代码，用来探索「如何让 AI 使用工具、自主完成任务」这件事。

---

## 目录结构

```
code/
├── calculator_agent.py   ← 入门示例：给 AI 配一个计算器
├── todo_agent.py         ← 实用示例：让 AI 读文件、分析待办清单
├── todo.md               ← todo_agent 使用的待办清单样本
│
├── min_agent/            ← 最小 Agent Loop 实现（核心）
│   ├── main.py           ← 入口：批量运行预设任务
│   ├── agent.py          ← Agent 主循环（observe → think → act）
│   ├── tools.py          ← 工具实现（增强版，当前使用）
│   ├── tools_v1.py       ← 工具实现（基础版，对比用）
│   ├── tools_v2.py       ← 工具实现（增强版，早期版本）
│   ├── tracer.py         ← 执行追踪器，记录每步日志 + 费用
│   └── tasks.py          ← 预设任务列表
│
├── context_builder/      ← 上下文构造器（探索如何组织 prompt）
│   ├── builder.py        ← 链式 API，组装五层上下文
│   ├── layers.py         ← 五层结构定义（数据类）
│   └── compressor.py     ← 长内容压缩器（防止撑爆上下文）
│
└── tests/                ← 单元测试
    ├── test_agent.py
    ├── test_tools.py
    ├── test_tracer.py
    └── test_context_builder.py
```

---

## 各模块说明

### `calculator_agent.py` — 入门示例

**演示内容：** Tool Use 基础用法——给 AI 配两个计算工具（加法、乘法），让它自主决定何时调用。

**核心流程：**
```
用户发问 → AI 思考 → AI 调用工具 → 代码执行 → 结果返回给 AI → AI 给出最终答案
```

**运行方式：**
```bash
export ANTHROPIC_API_KEY="你的密钥"
python calculator_agent.py
# 示例输入：帮我算：25 加 37，然后把结果乘以 2
```

---

### `todo_agent.py` — 实用示例

**演示内容：** 让 AI 自主读取本地文件（`todo.md`），然后生成四项分析报告。

**Agent 完成的任务：**
1. 📋 列出所有未完成任务
2. ⚡ 给出优先级建议
3. 📂 按类别分组整理
4. 📅 给出今日行动建议

**运行方式：**
```bash
python todo_agent.py
```

---

### `min_agent/` — 最小 Agent Loop

这是核心探索模块，实现了一个完整的最小 Agent 闭环。

#### `agent.py` — 主循环

实现经典的 **observe → think → act** 循环：

```
while 没完成且步数未超限:
    think：调用 Claude 决策
    if end_turn → 任务完成，退出
    if tool_use → 执行工具，把结果喂回 Claude，继续下一步
```

最大步数限制为 5 步，防止无限循环。

#### `tools.py` — 增强版工具

提供两个工具：
- **`search_notes`**：搜索本地笔记库（只读，安全）
- **`write_summary`**：保存摘要文本（高风险，标注需确认）

增强点包括：
- 结构化错误返回（含 `next_action` 指导 AI 下一步）
- 分页支持（`max_results` 参数，防止塞爆上下文）
- 内容截断（单条结果超长自动截断）
- 带指数退避的重试机制
- 详细的工具描述（AI 知道怎么用、出错怎么办）

#### `tools_v1.py` — 基础版工具（对比用）

同样实现上面两个工具，但**故意保留各种缺陷**，用于和增强版对比学习：
- 出错只返回模糊文字，AI 不知道怎么重试
- 没有分页，数据多了全部返回
- 没有重试，网络抖一下就失败
- 工具描述极简，AI 不清楚参数格式

#### `tracer.py` — 执行追踪器

每步执行后，将以下内容追加写入 `trace.jsonl`：
- 步骤编号、时间戳
- AI 的思考摘要
- 调用的工具和参数
- 工具返回结果
- 本步 token 费用估算（基于 claude-opus-4-8 定价）

#### `main.py` — 批量运行入口

读取 `tasks.py` 中的预设任务，依次运行，汇总成功/失败结果，并提示查看 trace 日志。

**运行方式：**
```bash
cd code
export ANTHROPIC_API_KEY="你的密钥"
python -m min_agent.main
# 查看详细日志
cat trace.jsonl | python3 -m json.tool
```

---

### `context_builder/` — 上下文构造器

探索如何结构化地组织发给 AI 的上下文信息，避免把所有内容混成一大段文字。

#### 五层结构（`layers.py`）

| 层级 | 名称 | 存放内容 |
|------|------|---------|
| 层 1 | `SystemLayer` | 角色定义、行为规则、可用工具（固定不变）|
| 层 2 | `TaskLayer` | 当前任务描述、目标、约束 |
| 层 3 | `MemoryLayer` | 跨任务的持久化知识（用户偏好、历史摘要）|
| 层 4 | `EvidenceLayer` | 检索到的外部证据（笔记、文档、搜索结果）|
| 层 5 | `TraceLayer` | 最近的工具调用记录 |

#### `compressor.py` — 内容压缩器

当某层内容过长时自动截断，同时：
- 生成可追溯的引用 ID（`ref:xxxxxxxx`，MD5 前 8 位）
- 保留原始内容，可通过引用 ID 检索回来
- 相同内容产生相同 ID（可重现）

#### `builder.py` — 链式构建 API

```python
bundle = (
    ContextBuilder()
    .set_system(role="助手", instructions=["用简洁中文回复"])
    .set_task(description="总结笔记", goal="输出 3 句话摘要")
    .add_memory(key="用户偏好", value="喜欢用列表格式")
    .add_evidence(source="note.md", content="...很长的笔记内容...")
    .add_trace_step(step=1, tool="search_notes", args={...}, result="...")
    .build()
)
```

---

## 运行测试

```bash
cd code
pytest tests/
```

---

## 依赖

```bash
pip install anthropic
```

需要设置环境变量：
```bash
export ANTHROPIC_API_KEY="你的 API 密钥"
```
