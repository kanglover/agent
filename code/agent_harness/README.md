# Agent Harness — Agent 测评框架

## 一句话理解

**Agent Harness 就是 Agent 的"考试系统"** —— 给不同的 Agent 出同样的题，看谁做得更好。

## 用驾考来类比

| 驾考概念 | Harness 对应 | 文件 |
|---------|-------------|------|
| 考生须知（统一规则） | Agent 协议 | `protocol.py` |
| 考卷（一组题目） | 任务定义 | `tasks.py` |
| 评分标准 | 评估器 | `evaluator.py` |
| 考场总控 | Harness 核心 | `harness.py` |
| 考生们 | 示例 Agent | `agents/` |
| 考场工具（计算器等） | 工具集 | `tools.py` |

## 快速运行

```bash
# 进入 code 目录
cd code

# 默认模式：只跑模拟 Agent（不需要 API key）
python -m agent_harness.main

# LLM 模式：加上真实 Claude Agent（需要 ANTHROPIC_API_KEY）
python -m agent_harness.main --with-llm
```

## 架构数据流

```
Task (考题)
  │
  ▼
Harness.run()  ←── 考场总控
  │
  ├── Agent 1: SimpleAgent.run_with_lifecycle(task)
  │     ├── setup()
  │     ├── run(task, tools)  →  AgentResult
  │     └── teardown()
  │
  ├── Agent 2: MockLLMAgent.run_with_lifecycle(task)
  │     └── ...同上...
  │
  └── (可选) Agent 3: ReactAgent.run_with_lifecycle(task)
        └── ...调用 Claude API...
  │
  ▼
Evaluator.evaluate(task, result)  ←── 阅卷
  │
  ▼
Report  ←── 成绩单（排名 + 详细评分）
```

## 如何添加自己的 Agent

只需要 3 步：

```python
# 1. 继承协议
from agent_harness.protocol import AgentProtocol, AgentResult

class MyAgent(AgentProtocol):
    name = "我的Agent"

    # 2. 实现 run 方法
    def run(self, task, tools=None, context=None):
        # ... 你的 Agent 逻辑 ...
        return AgentResult(success=True, output="完成了", steps=1)

# 3. 注册到 Harness
harness.add_agent(MyAgent())
```

## 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 完成度 | 40% | 任务是否成功完成 |
| 关键词 | 30% | 答案是否覆盖关键内容 |
| 工具使用 | 20% | 是否正确使用必要工具 |
| 效率 | 10% | 步骤数是否合理（≤5步满分） |

## 文件说明

```
agent_harness/
├── __init__.py           # 包说明
├── protocol.py           # Agent 协议（接口定义）
├── tasks.py              # 任务定义（测试用例）
├── evaluator.py          # 评估器（评分逻辑）
├── harness.py            # Harness 核心（调度引擎）
├── tools.py              # 工具集（供 Agent 使用）
├── main.py               # 运行入口
├── README.md             # 本文档
└── agents/
    ├── __init__.py
    ├── simple_agent.py   # 规则匹配 Agent（基线）
    └── react_agent.py    # ReAct Agent + Mock Agent
```
