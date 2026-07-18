# ReAct Agent：核心组成与模板

> 沉淀时间：2026-07-06

## 一句话总结

ReAct = Reasoning + Acting，让 Agent 边想边做，通过"思考→行动→观察"循环完成复杂任务。

## 核心是什么？

普通 LLM：一次性输出答案，无法调用工具、无法分步推理。

ReAct：强制模型按固定格式输出，解析后驱动工具执行：

```
Thought:     我需要先查一下天气...（分析当前情况，决定下一步）
Action:      get_weather({"city": "北京"})  （调用工具）
Observation: 北京今天32℃，晴              （工具返回结果，系统填入）
Thought:     好，现在我有数据了，可以回答了
Final Answer: 北京今天很热，建议...
```

循环直到输出 `Final Answer`。

---

## 核心三要素

### 1. System Prompt 模板

```python
REACT_SYSTEM_PROMPT = """
你是一个能使用工具的助手。

可用工具：
{tool_descriptions}

严格按以下格式回答：

Thought: 分析当前情况，决定下一步
Action: tool_name({"param": "value"})
Observation: [系统填入工具结果，你不要写这行]
...（可多轮）
Thought: 我现在有足够信息了
Final Answer: [最终答案]

重要规则：
- 每次只能调用一个工具
- 必须等 Observation 回来再继续
- 不要编造 Observation 的内容
"""
```

### 2. 主循环（最小实现）

```python
def react_agent(question, tools, max_steps=10):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user",   "content": question}
    ]

    for step in range(max_steps):
        response = llm(messages)
        text = response.content

        # 检查是否完成
        if "Final Answer:" in text:
            return extract_final_answer(text)

        # 解析并执行工具
        action, params = parse_action(text)
        observation = tools[action](**params)

        # 把结果追加回去，进入下一轮
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",      "content": f"Observation: {observation}"})

    return "达到最大步数限制"
```

### 3. 工具定义格式

```python
tools = {
    "get_weather": get_weather_func,
    "search_web": search_web_func,
}

tool_descriptions = """
get_weather(city: str) -> str
  描述：查询指定城市的天气
  示例：get_weather({"city": "北京"})

search_web(query: str) -> str
  描述：搜索互联网信息
  示例：search_web({"query": "今天新闻"})
"""
```

---

## 四个核心组成部分

```
┌─────────────────────────────────────────────────────┐
│              ReAct Agent = 四个部分                  │
│                                                     │
│  1. LLM（大脑）  ← 负责推理、决策、生成 Thought      │
│  2. Tools（手）  ← 可调用的工具集（搜索/计算/API）   │
│  3. Memory（记忆）← 对话历史 messages               │
│  4. Loop（循环） ← 驱动 Thought→Action→Observation  │
└─────────────────────────────────────────────────────┘
```

---

## 和普通 LLM 调用的区别

| 对比项 | 普通 LLM | ReAct Agent |
|--------|---------|-------------|
| 能力 | 只能输出文字 | 能调用工具执行动作 |
| 推理方式 | 一次性输出 | 多步推理，逐步行动 |
| 知识来源 | 仅训练数据 | 训练数据 + 实时工具结果 |
| 适合任务 | 问答、写作 | 搜索、计算、操作系统、多步骤任务 |

---

## 三者（记忆、RAG、ReAct）的关系

```
用户输入
    │
    ├── 意图识别 → 决定用哪些工具/流程
    │
    ├── 短期记忆 → 当前对话 messages
    │
    ├── 长期记忆 → 历史经验（向量检索注入）
    │
    └── RAG 检索 → 外部知识（防幻觉/知识更新）
              │
              └── 全部注入 Prompt → ReAct 循环 → 工具调用 → 最终回答
```

记忆解决"知道什么"，RAG 解决"从哪找知识"，ReAct 解决"怎么一步步行动"。

---

## 适用场景

- 需要调用多个工具完成任务（搜索+计算+写作）
- 任务需要多步推理（中间结果影响下一步决策）
- 不确定需要几步能完成的开放性任务

## 不适合的场景

- 简单问答（一次 LLM 调用就够）
- 对延迟要求极高（多步循环慢）
- 工具调用有副作用且不可撤销（需要更谨慎的设计）

---
*相关概念：Tool Use、Function Calling、Agent Loop、LangGraph、意图识别、记忆机制*
