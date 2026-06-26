# Tool Use 实战：计算器工具

> 整理自 2026-06-23 的学习对话

---

## 核心理解

Tool Use 的工作分工：

| 谁做 | 做什么 |
|------|--------|
| AI（Claude） | 决定"用哪个工具"、"用什么参数" |
| 你的代码 | 真正执行工具（计算、查询、发请求等） |
| 你的代码 | 把结果告诉 AI，让它继续 |

---

## 工作流程

```
用户问题 → AI思考(要用add工具) → 我们执行add → 把结果给AI → AI继续思考 → 最终回答
```

---

## 完整代码

```python
import anthropic

client = anthropic.Anthropic()  # 需要 ANTHROPIC_API_KEY 环境变量

# 1. 定义工具——就像给 AI 一份工具说明书
tools = [
    {
        "name": "add",
        "description": "计算两个数相加",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "multiply",
        "description": "计算两个数相乘",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"}
            },
            "required": ["a", "b"]
        }
    }
]

# 2. 真正执行计算的函数（AI 下命令，代码执行）
def execute_tool(tool_name, tool_input):
    if tool_name == "add":
        return str(tool_input["a"] + tool_input["b"])
    elif tool_name == "multiply":
        return str(tool_input["a"] * tool_input["b"])
    return "未知工具"

# 3. Agent 主循环
messages = [
    {"role": "user", "content": "帮我算：25 加 37，然后把结果乘以 2"}
]

print("🤖 开始运行...\n")

while True:
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # 情况一：AI 完成了任务
    if response.stop_reason == "end_turn":
        print(f"✅ 最终回答：{response.content[0].text}")
        break
    
    # 情况二：AI 要用工具
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"🔧 使用工具：{block.name}，参数：{block.input}")
                result = execute_tool(block.name, block.input)
                print(f"   结果：{result}\n")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        messages.append({"role": "user", "content": tool_results})
```

---

## 运行方式

```bash
export ANTHROPIC_API_KEY="你的密钥"
python calculator_agent.py
```

## 预期输出

```
🤖 开始运行...

🔧 使用工具：add，参数：{'a': 25, 'b': 37}
   结果：62

🔧 使用工具：multiply，参数：{'a': 62, 'b': 2}
   结果：124

✅ 最终回答：25 加 37 等于 62，62 乘以 2 等于 124。
```

---

## 关键代码解释

- `tools` 列表：告诉 AI 有哪些工具，以及每个工具需要什么参数
- `execute_tool()` 函数：真正执行计算的地方，这里可以换成任何逻辑
- `response.stop_reason == "tool_use"`：检测 AI 是否要用工具
- `tool_use_id`：每次工具调用都有唯一 ID，结果必须对应回去

---

## 扩展练习

可以尝试添加更多工具：
- `subtract`：两数相减
- `divide`：两数相除（注意除以0的情况）
- `square_root`：开平方根
