# Agent Harness Demo

一个用于教学演示的 **Agent Harness（智能体运行时骨架）**，零第三方依赖（纯 Python 标准库），完整展示构建一个 Agent 所需的六大基础设施子系统。

## 六大子系统

| 模块 | 文件 | 职责 |
|------|------|------|
| 上下文管理 | `harness/context.py` | 对话历史、token 估算、窗口超限自动摘要压缩、序列化往返 |
| 工具系统 | `harness/tools.py` | 工具基类 + 注册表 + 三个演示工具（计算器/Mock 搜索/读文件） |
| 执行编排 | `harness/orchestrator.py` | think → act → observe 主循环，串联所有子系统 |
| 记忆与状态 | `harness/memory.py` | 短期记忆、长期记忆（JSON 持久化）、任务状态机、快照/恢复 |
| 评估观测 | `harness/observer.py` | 全链路 trace、指标统计（token/工具调用/错误）、运行摘要 |
| 约束与恢复 | `harness/constraints.py` | 步数/token/超时/错误次数熔断、重试(指数退避)/降级/安全执行 |
| LLM 接口 | `harness/llm.py` | 脚本式 MockLLM，无需真实 API 即可演示完整 Agent 流程 |

## 架构

```
                 ┌─────────────────────┐
                 │    Orchestrator     │
                 │ think→act→observe   │
                 └─────────┬───────────┘
       ┌─────────┬─────────┼─────────┬─────────┐
       ▼         ▼         ▼         ▼         ▼
  Context     Tools     Memory   Observer  Constraints
  Manager    Registry           (trace)   (熔断/恢复)
       ▲                                        
       └──────────── MockLLM ────────────────────┘
                  (脚本式模拟)
```

## 快速开始

```bash
cd code/harness_demo

# 运行完整演示（5 个场景）
python3 main.py

# 运行测试（33 个用例）
python3 -m unittest tests.test_harness -v
```

无需安装任何依赖，Python 3.10+ 即可。

## 演示场景（main.py）

1. **标准多步任务** — 计算圆面积 → 搜索公式 → 总结，展示完整 think/act/observe 循环
2. **上下文自动压缩** — 小窗口下历史消息被摘要压缩，token 占用受控
3. **约束熔断** — 步数超限时安全中止，状态标记为 `aborted`
4. **记忆持久化与快照恢复** — 长期记忆写入磁盘跨会话读取；快照还原断点状态
5. **工具错误与恢复** — 非法表达式触发工具报错，错误写回上下文后任务仍能完成

## 核心用法

```python
from harness import (
    MockLLM, ContextManager, ToolRegistry,
    CalculatorTool, SearchTool, FileReadTool,
    Memory, Observer, ConstraintManager, Orchestrator,
    default_script,
)

# 组装六大子系统
llm = MockLLM(default_script())          # 1. LLM（可换成真实 API）
context = ContextManager(system_prompt="你是 AI 助手")
tools = ToolRegistry()
tools.register(CalculatorTool())
memory = Memory(storage_path="mem.json") # 2. 长期记忆持久化
observer = Observer()
constraints = ConstraintManager(max_steps=20)

# 运行
orch = Orchestrator(llm, context, tools, memory, observer, constraints)
result = orch.run("计算半径 5 的圆面积")

# 查看观测报告
print(observer.summary())
```

## 关键设计

- **单步循环**: `Orchestrator.step()` 每步执行 think（LLM 推理）→ act（工具执行）→ observe（结果回写、约束检查、上下文维护），返回是否继续
- **约束先于执行**: 每步开始前 `check_all(step, tokens, elapsed)`，error 级违规立即熔断
- **错误也是上下文**: 工具失败以 tool 消息写回，LLM 可以看到错误并自行修正
- **观测零侵入**: `Observer.log()` 通过 hooks 扩展，hook 异常自动吞掉不影响主流程
- **恢复策略可组合**: `Recovery.retry`（指数退避）、`with_fallback`（降级）、`safe_execute`（兜底默认值）

## 目录结构

```
harness_demo/
├── main.py                  # 演示入口
├── README.md
├── harness/
│   ├── __init__.py          # 公共 API 导出
│   ├── llm.py               # LLM 接口 + MockLLM
│   ├── context.py           # 上下文管理
│   ├── tools.py             # 工具系统
│   ├── memory.py            # 记忆与状态
│   ├── observer.py          # 评估观测
│   ├── constraints.py       # 约束与恢复
│   └── orchestrator.py      # 执行编排
└── tests/
    └── test_harness.py      # 单元 + 集成测试
```
