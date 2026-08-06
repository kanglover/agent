# 多 Agent 协作系统

## 一句话总结

多 Agent 系统就像一个小团队：有一个项目经理（Coordinator）负责分配任务，几个专业员工（Worker）各司其职，大家通过共享文档（Shared State）协作，反复迭代直到产出合格的结果。

---

## 核心概念

### 1. 什么是多 Agent 系统？

单 Agent 系统 = 一个人单打独斗，既要做调研、又要写文档、还要自己检查。

多 Agent 系统 = 一个团队协作，每个人只做自己最擅长的事：
- 有人专门调研（Researcher）
- 有人专门写作（Writer）
- 有人专门审核（Reviewer）
- 有人专门协调（Coordinator）

### 2. 四个关键概念

| 概念 | 通俗解释 | 类比 |
|------|---------|------|
| **共享状态** | 所有 Agent 读写同一份数据 | 团队共享的在线文档 |
| **消息传递** | Agent 之间通过消息沟通 | 工作群里的@通知 |
| **路由决策** | Coordinator 决定下一步谁干活 | 项目经理分配任务 |
| **迭代改进** | 审核不通过时循环修改 | 反复改稿直到通过 |

---

## 系统架构

```
                    用户任务
                       │
                       ▼
            ┌─────────────────────┐
            │   Coordinator       │  ← 项目经理：拆分任务、分配工作
            │   （调度员）         │
            └─────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │Researcher│  │  Writer  │  │ Reviewer │
   │（研究员） │  │（写手）  │  │（审核员）│
   └──────────┘  └──────────┘  └──────────┘
          │            │            │
          └────────────┴────────────┘
                       │
                       ▼
                  最终结果
```

---

## 协作流程（端到端）

```
Step 1: 用户提交任务
        ↓
Step 2: Coordinator 拆分任务
        "把这个大任务拆成 3 个小任务：调研、写作、审核"
        ↓
Step 3: Coordinator → Researcher
        "你去调研一下这个主题的背景"
        ↓
Step 4: Researcher 完成调研，写入共享状态
        "调研报告已写好，放在共享文档里了"
        ↓
Step 5: Coordinator → Writer
        "调研完成了，你根据报告写一份技术方案"
        ↓
Step 6: Writer 完成草稿，写入共享状态
        "第 1 稿已写好"
        ↓
Step 7: Coordinator → Reviewer
        "写好了，你来审核一下质量"
        ↓
Step 8: Reviewer 审核
        ├─ ✅ 通过 → Coordinator 决定结束
        └─ ❌ 不通过 → 提出修改意见 → 回到 Step 5（Writer 修改）
        ↓
Step 9: 输出最终结果
```

---

## 代码文件

- `/Users/kangkang/Documents/workroot/agent/code/multi_agent_demo.py` —— 完整的多 Agent 协作示例

运行方法：
```bash
cd code
python3 multi_agent_demo.py
```

**特点**：纯 Python 实现，无需 API Key，直接运行即可看到完整的协作流程。

---

## 关键代码解析

### 1. 共享状态（TaskState）

```python
@dataclass
class TaskState:
    task: str = ""           # 用户原始任务
    research: str = ""       # 调研结果
    draft: str = ""          # 写手草稿
    review: str = ""         # 审核意见
    revision_count: int = 0  # 修订次数（防止死循环）
    is_approved: bool = False # 是否通过审核
```

所有 Agent 都读写这一份状态，就像团队共享一个在线文档。

### 2. Coordinator 的决策逻辑

```python
def _coordinator_decision(self, state: TaskState) -> str:
    if not state.subtasks:
        return "SPLIT_TASK"           # 还没拆分，先拆分
    elif not state.research:
        return "CALL_RESEARCHER"      # 还没调研，叫研究员
    elif not state.draft:
        return "CALL_WRITER"          # 还没写作，叫写手
    elif not state.review:
        return "CALL_REVIEWER"        # 还没审核，叫审核员
    elif state.is_approved:
        return "FINISH"               # 通过了，结束
    elif state.revision_count >= 2:
        return "FINISH"               # 改太多次了，强制结束
    elif state.needs_revision:
        return "CALL_WRITER"          # 需要修改，叫写手
    else:
        state.needs_revision = True
        return "CALL_WRITER"          # 标记需要修改
```

Coordinator 就像一个状态机，根据当前进度决定下一步。

### 3. 迭代改进的关键设计

Writer 修改完成后，会**清空旧的 review**，这样 Coordinator 下次就会重新调用 Reviewer 审核，而不是直接结束。

```python
def run(self, state: TaskState) -> TaskState:
    if state.needs_revision:
        # 修改模式
        draft = self._think(state)
        state.review = ""          # ← 清空旧审核意见
        state.needs_revision = False
    else:
        # 初稿模式
        draft = self._think(state)
    state.draft = draft
    return state
```

---

## 实际运行效果

```
🚀 多 Agent 协作系统启动
📋 任务：帮我写一份关于「如何设计一个高并发缓存系统」的技术方案

📌 Step 1
📋 任务拆分：
   1. 调研背景信息
   2. 撰写技术方案
   3. 审核质量

📌 Step 2
🔄 Coordinator → Researcher
📄 调研结果（前 200 字）：【调研报告】...

📌 Step 3
🔄 Coordinator → Writer
📄 草稿（前 200 字）：（第 1 稿）...

📌 Step 4
🔄 Coordinator → Reviewer
📄 审核意见（前 200 字）：NEEDS_REVISION...

📌 Step 5
🔄 Coordinator → Writer
📄 修改后的草稿（前 200 字）：（第 2 轮修订）...

📌 Step 6
🔄 Coordinator → Reviewer
📄 审核意见（前 200 字）：APPROVED ✅

📌 Step 7
🏁 任务完成！
```

---

## 从 Mock 到真实 LLM

当前代码使用 `MockLLM` 模拟 AI 回复，方便在没有 API Key 的情况下运行。

如果要接入真实的 Claude / GPT，只需替换 `MockLLM.generate()` 方法：

```python
class RealLLM:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.client = anthropic.Anthropic()  # 或 openai.OpenAI()

    def generate(self, prompt: str, context: TaskState) -> str:
        # 根据 agent_name 选择不同的 system prompt
        system = self._get_system_prompt(self.agent_name)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.content[0].text
```

---

## 扩展思路

1. **增加更多 Worker**：比如 Translator（翻译）、Formatter（格式化）、Tester（测试）
2. **并行执行**：Researcher 和 Writer 可以并行（如果任务允许）
3. **人类介入**：在关键节点暂停，等人类确认后再继续
4. **记忆持久化**：把每次运行的状态保存到文件，下次继续

---

## 相关文件

- `multi_agent_demo.py` —— 本示例的完整代码
- `examples/python/15_langgraph_multiagent.py` —— 使用 LangGraph 实现的多 Agent 系统（需要 API Key）
- `examples/python/16_autogen_multiagent.py` —— 使用 AutoGen 实现的多 Agent 系统
- `examples/python/17_crewai_basics.py` —— 使用 CrewAI 实现的多 Agent 系统
