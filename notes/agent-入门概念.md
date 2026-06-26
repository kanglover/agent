# AI Agent 开发入门

> 整理自 2026-06-23 的学习对话

---

## 什么是 Agent（智能体）？

**普通 AI 对话** = 你问一句，它答一句（无状态、单次交互）

**AI Agent** = 你交任务，它自己思考、自己行动、自己完成（有目标、有循环、会用工具）

---

## Agent 的三个核心组件

| 组件 | 说明 |
|------|------|
| **大脑（LLM）** | Claude / GPT 等大模型，负责思考和决策 |
| **工具（Tools）** | Agent 能调用的能力（搜索、代码执行、文件读写等） |
| **循环（Loop）** | 接收任务 → 思考 → 调用工具 → 看结果 → 继续思考 → 完成 |

---

## 最简单的 Python Agent 代码结构

```python
import anthropic

client = anthropic.Anthropic()

# 1. 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "查询某个城市的天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

# 2. 工具执行函数（连接真实 API）
def execute_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        return f"{tool_input['city']}今天晴，22°C"

# 3. Agent 循环
messages = [{"role": "user", "content": "上海今天天气怎么样？"}]

while True:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    if response.stop_reason == "end_turn":
        print(response.content[0].text)
        break
    
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        messages.append({"role": "user", "content": tool_results})
```

---

## 学习路线图

1. **第一步**：学会用 Claude API 发消息（最基础的 AI 调用）
2. **第二步**：学习 Tool Use（给 AI 配工具）
3. **第三步**：搭建一个简单的 Agent 循环
4. **第四步**：了解 Managed Agents（托管式 Agent）

---

## 参考资源

- [Anthropic Claude API 文档](https://platform.claude.com/docs)
- [Tool Use 教程](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Agent 设计模式](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
