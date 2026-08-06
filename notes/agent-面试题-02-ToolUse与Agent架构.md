# Agent 面试题精选 200 道 —— Tool Use 与 Agent 架构

> 目标群体：前端工程师转 Agent 开发，备战 AI 工程师岗位
> 覆盖主题：Function Calling、工具定义、ReAct、Agent Loop、并行调用、错误处理、安全性、记忆系统、上下文管理、Token Budget、可观测性、评测方法
> 日期：2026-07-03

---

## 目录

1. [Function Calling / Tool Use 协议](#一-function-calling--tool-use-协议)（Q1–Q22）
2. [工具定义 JSON Schema 最佳实践](#二-工具定义-json-schema-最佳实践)（Q23–Q38）
3. [工具调用完整生命周期](#三-工具调用完整生命周期)（Q39–Q53）
4. [ReAct 框架原理](#四-react-框架原理)（Q54–Q67）
5. [Agent Loop 实现细节](#五-agent-loop-实现细节)（Q68–Q85）
6. [并行工具调用](#六-并行工具调用)（Q86–Q97）
7. [工具错误处理与重试](#七-工具错误处理与重试)（Q98–Q111）
8. [工具调用安全性](#八-工具调用安全性)（Q112–Q122）
9. [Agent 规划能力](#九-agent-规划能力)（Q123–Q132）
10. [CoT 与工具调用结合](#十-cot-与工具调用结合)（Q133–Q142）
11. [Agent 记忆系统](#十一-agent-记忆系统)（Q143–Q155）
12. [上下文管理策略](#十二-上下文管理策略)（Q156–Q165）
13. [Token Budget 控制](#十三-token-budget-控制)（Q166–Q173）
14. [Agent 可观测性](#十四-agent-可观测性)（Q174–Q182）
15. [Agent 评测方法](#十五-agent-评测方法)（Q183–Q190）
16. [工具设计模式](#十六-工具设计模式)（Q191–Q196）
17. [Human-in-the-Loop 设计](#十七-human-in-the-loop-设计)（Q197–Q200）

---

## 一、Function Calling / Tool Use 协议

### Q1. 【基础】什么是 Function Calling？它解决了什么问题？

**难度**：★
**类型**：概念理解

**详细解答**：

Function Calling（也称 Tool Use）是让 LLM 能够"调用外部函数"的机制。LLM 本身是纯文本输入输出的模型，无法直接查数据库、发 HTTP 请求、读文件。Function Calling 通过在 API 层面定义一套协议，让模型能够：

1. 识别"当前任务需要调用某个工具"
2. 输出一个结构化的"调用指令"（包含函数名 + 参数）
3. 应用层执行实际调用，把结果返回给模型
4. 模型根据结果继续生成回答

**核心价值**：
- 实时数据访问（天气、股价、数据库查询）
- 执行副作用操作（发邮件、写文件、调用 API）
- 将复杂计算委托给专用工具（代码执行器、计算器）

**考察点**：Function Calling 的本质是"结构化输出 + 应用层执行"的协作机制，LLM 只负责"决策"，不负责"执行"。

---

### Q2. 【基础】Anthropic 的 Tool Use 协议和 OpenAI 的 Function Calling 有什么核心区别？

**难度**：★★
**类型**：协议对比

**详细解答**：

两套协议概念相同，但字段命名和消息结构有差异：

**OpenAI 格式（请求侧）**：
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气？"}],
    tools=tools
)
```

**OpenAI 响应**（参数是 JSON 字符串）：
```python
# response.choices[0].message.tool_calls[0].function.arguments 是字符串！
import json
args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

**Anthropic 格式（请求侧）**：
```python
tools = [
    {
        "name": "get_weather",
        "description": "获取城市当前天气",
        "input_schema": {          # 注意：是 input_schema，不是 parameters
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京天气？"}]
)
```

**Anthropic 响应**（参数已是 dict）：
```python
for block in response.content:
    if block.type == "tool_use":
        print(block.id)      # "toolu_01xxx"
        print(block.name)    # "get_weather"
        print(block.input)   # {"city": "北京"}  — 已解析为 dict
```

**核心区别汇总**：

| 特性 | OpenAI | Anthropic |
|------|--------|-----------|
| 工具参数字段名 | `parameters` | `input_schema` |
| 工具调用 ID 前缀 | `call_` | `toolu_` |
| 参数格式 | JSON 字符串（需 json.loads） | 已解析的 dict |
| 工具结果消息角色 | `tool` | `user`（嵌套 tool_result block） |

**Anthropic 返回工具结果的方式**：
```python
messages.append({"role": "assistant", "content": response.content})
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": "32°C，晴"
        }
    ]
})
```

**考察点**：熟悉两套 API 的字段差异，尤其 Anthropic 的 `input_schema` 和 `tool_result` 用 user 角色。

---

### Q3. 【中级】`tool_choice` 参数有什么作用？有哪几种模式？

**难度**：★★
**类型**：参数细节

**详细解答**：

`tool_choice` 控制模型是否必须调用工具：

**OpenAI 的 tool_choice**：
```python
tool_choice = "auto"       # 模型自主决定（默认）
tool_choice = "none"       # 禁用工具，只生成文本
tool_choice = "required"   # 必须调用至少一个工具
# 强制调用特定工具
tool_choice = {"type": "function", "function": {"name": "get_weather"}}
```

**Anthropic 的 tool_choice**：
```python
tool_choice = {"type": "auto"}              # 模型自主决定（默认）
tool_choice = {"type": "any"}               # 必须调用至少一个工具
tool_choice = {"type": "tool", "name": "get_weather"}  # 强制调用指定工具
```

**使用场景**：
- 结构化数据提取时用 `tool` 模式，确保输出符合 schema
- 纯对话时用 `none` 避免意外工具调用
- 调试 Agent 规划时用 `any` 强制触发工具

**考察点**：`tool_choice="tool"` 是实现可靠结构化输出的关键控制参数。

---

### Q4. 【中级】什么是 `stop_reason`？`tool_use` 和 `end_turn` 有什么区别？

**难度**：★★
**类型**：状态判断

**详细解答**：

**Anthropic 的 stop_reason**：

| stop_reason | 含义 | 后续动作 |
|-------------|------|---------|
| `end_turn` | 模型完成回答 | 返回给用户 |
| `tool_use` | 模型想调用工具 | 执行工具，继续循环 |
| `max_tokens` | 达到 token 上限 | 截断处理 |
| `stop_sequence` | 命中自定义停止词 | 业务逻辑处理 |

**Agent Loop 核心判断**：
```python
while True:
    response = client.messages.create(...)

    if response.stop_reason == "end_turn":
        return extract_text(response)  # 完成

    elif response.stop_reason == "tool_use":
        # 执行工具，将结果加入历史，继续循环
        results = execute_all_tools(response)
        messages = update_history(messages, response, results)

    elif response.stop_reason == "max_tokens":
        raise TokenLimitError("响应被截断，需增加 max_tokens")
```

**考察点**：stop_reason 是 Agent Loop 循环终止条件的判断依据。

---

### Q5. 【中级】工具调用的 `id`（tool_use_id）有什么作用？

**难度**：★★
**类型**：协议细节

**详细解答**：

工具调用 ID 用于**关联工具调用请求和执行结果**，在并行调用时至关重要：

```python
# 模型同时发出两个工具调用
response.content = [
    ToolUseBlock(id="toolu_01", name="get_weather", input={"city": "北京"}),
    ToolUseBlock(id="toolu_02", name="get_weather", input={"city": "上海"}),
]

# 执行后通过 id 关联结果
tool_results = [
    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "北京：32°C"},
    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "上海：28°C"},
]
```

如果 ID 不匹配，模型无法正确理解哪个结果对应哪个请求。

**考察点**：ID 是并行调用中结果匹配的唯一标识符，必须原样保留并传回。

---

### Q6. 【进阶】如何处理流式（streaming）模式下的工具调用？

**难度**：★★★
**类型**：实现细节

**详细解答**：

流式模式下，工具调用的参数通过多个 delta 事件拼接：

```python
import anthropic, json

client = anthropic.Anthropic()
tool_inputs = {}  # {tool_id: {"name": str, "input_str": str}}

with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=[weather_tool],
    messages=[{"role": "user", "content": "北京天气？"}]
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                tool_inputs[block.id] = {"name": block.name, "input_str": ""}

        elif event.type == "content_block_delta":
            if event.delta.type == "input_json_delta":
                # 拼接增量 JSON
                for tid in tool_inputs:
                    tool_inputs[tid]["input_str"] += event.delta.partial_json

        elif event.type == "message_stop":
            # 流结束，解析完整参数
            for tid, data in tool_inputs.items():
                data["input"] = json.loads(data["input_str"])
```

**TypeScript 版本**：
```typescript
const stream = await anthropic.messages.stream({
  model: "claude-opus-4-5",
  max_tokens: 1024,
  tools: [weatherTool],
  messages: [{ role: "user", content: "北京天气？" }],
});

// 使用便利方法
const finalMsg = await stream.finalMessage();
const toolCalls = finalMsg.content.filter(b => b.type === "tool_use");
```

**考察点**：流式工具调用需手动拼接增量 JSON；SDK 的 `finalMessage()` 方法可简化处理。

---

### Q7. 【进阶】为什么工具调用时模型可能同时输出文本和工具调用？如何处理？

**难度**：★★★
**类型**：边界情况

**详细解答**：

Anthropic 模型在同一响应中可同时包含文本和工具调用：

```python
# response.content 可能是这样的结构
[
    TextBlock(type="text", text="我来查一下天气..."),
    ToolUseBlock(type="tool_use", id="toolu_01", name="get_weather", input={...}),
]
```

**正确处理方式**：
```python
def process_response(response):
    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(block)

    # 可以先展示模型的思考过程
    if text_parts and tool_calls:
        print("模型说：", "".join(text_parts))

    return tool_calls

# 错误做法：只处理 response.content[0]
# 正确做法：遍历所有 blocks
```

**考察点**：不能假设 content 只有一个元素，必须遍历处理，否则会丢失工具调用。

---

### Q8. 【进阶】什么是 computer_use 工具？它与普通工具有何不同？

**难度**：★★★
**类型**：高级特性

**详细解答**：

`computer_use` 是 Anthropic 的内置工具类型，允许模型操作计算机界面（截屏、点击、输入）：

```python
tools = [
    {
        "type": "computer_20241022",   # 使用 type 而非 name
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
    },
    {
        "type": "bash_20241022",
        "name": "bash"
    }
]
```

**与普通工具的区别**：

| 特性 | 普通自定义工具 | computer_use |
|------|--------------|--------------|
| 定义方式 | `name` + `input_schema` | `type` 字段（内置 schema） |
| 输入来源 | 文本参数 | 截图（多模态） |
| 安全级别 | 中 | 高（需严格沙箱） |
| 适用场景 | API/数据库调用 | 界面自动化 |

**考察点**：computer_use 是多模态工具，需要视觉理解能力，且风险显著高于普通工具。

---

### Q9. 【中级】如何通过 `cache_control` 缓存工具定义以降低成本？

**难度**：★★★
**类型**：性能优化

**详细解答**：

当工具列表庞大且固定时，使用提示词缓存可大幅降低成本：

```python
tools = [
    {
        "name": "search_docs",
        "description": "搜索文档库（详细描述...几百字...）",
        "input_schema": {...}
    },
    # 更多工具...
    {
        "name": "get_weather",
        "description": "获取实时天气",
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"}  # 在最后一个工具上标记缓存断点
    }
]
```

**工作原理**：
- Anthropic 缓存到 `cache_control` 标记位置的所有 tokens
- 后续相同工具列表的请求读取缓存，费用约为原来的 10%
- 最适合工具定义几千 tokens 且固定的场景（如 coding agent）

**考察点**：Anthropic 提示词缓存对工具密集型 Agent 的成本优化效果显著（可节省 90% 的工具定义 token 费用）。

---

### Q10. 【基础】什么情况下模型会选择不调用工具，直接生成文本？

**难度**：★★
**类型**：行为理解

**详细解答**：

模型在以下情况下选择不调用工具：

1. **问题本身不需要工具**：`"1 + 1 等于多少？"` 直接回答
2. **工具描述与需求不匹配**：提供的工具与请求无关
3. **缺少必要参数**：无法从上下文推断工具所需参数
4. **安全策略限制**：检测到潜在有害操作
5. **`tool_choice: "none"`**：被明确禁止调用工具

```python
# 即使提供了天气工具，模型也会选择直接回答这个问题
tools = [{"name": "get_weather", ...}]
messages = [{"role": "user", "content": "写一首关于春天的诗"}]
# stop_reason = "end_turn"，不会调用工具
```

**考察点**：模型有"工具选择自主权"，工具是可用选项而非强制执行的命令；工具描述质量直接影响选择准确率。

---

### Q11. 【进阶】什么是 `disable_parallel_tool_use`？何时使用？

**难度**：★★★
**类型**：控制参数

**详细解答**：

`disable_parallel_tool_use` 是 Anthropic 的 tool_choice 中的可选字段，强制串行执行工具：

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    tool_choice={
        "type": "auto",
        "disable_parallel_tool_use": True  # 强制每次只调用一个工具
    },
    messages=messages
)
```

**适用场景**：
- 工具间有顺序依赖（B 依赖 A 的结果）
- 调试时需要逐步观察每个工具
- 工具有资源竞争（如并发写同一文件）
- 需要严格审计每步工具调用

**考察点**：并行提高效率，串行保证顺序依赖；根据业务需求选择合适模式。

---

### Q12. 【中级】如何正确地将工具结果注入对话历史？（Anthropic 格式）

**难度**：★★
**类型**：实现细节

**详细解答**：

Anthropic 的工具结果需要以特定格式注入，注意 role 是 `"user"`：

```python
def inject_tool_results(messages: list, response, results: dict) -> list:
    # Step 1: 追加 assistant 消息（包含 tool_use blocks）
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    # Step 2: 构建 tool_result 消息（role 必须是 user！）
    tool_result_blocks = []
    for block in response.content:
        if block.type == "tool_use":
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": results.get(block.id, "Error: tool not executed")
            })

    messages.append({
        "role": "user",        # 注意：工具结果用 user 角色！
        "content": tool_result_blocks
    })

    return messages

# 工具执行失败时标记错误
error_result = {
    "type": "tool_result",
    "tool_use_id": "toolu_01",
    "content": "Error: Connection timeout",
    "is_error": True  # 可选标记
}
```

**常见错误**：
- 用 `role: "tool"`（这是 OpenAI 格式！）
- 忘记先追加 assistant 消息
- 只追加 tool_result 而不追加 assistant 消息

**考察点**：Anthropic 的工具结果用 `user` 角色，这是初学者最常犯的错误。

---

### Q13. 【进阶】Extended Thinking 与工具调用如何结合使用？

**难度**：★★★★
**类型**：高级特性

**详细解答**：

Extended Thinking 允许模型在决定调用哪个工具之前进行深度推理：

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # 思考阶段的 token 预算
    },
    tools=tools,
    messages=messages
)

# 响应 content 的结构
# [
#   ThinkingBlock(type="thinking", thinking="<推理过程，不显示给用户>"),
#   TextBlock(type="text", text="我需要查询一下天气..."),
#   ToolUseBlock(type="tool_use", name="get_weather", ...),
# ]
```

**处理 thinking blocks 的关键规则**：
```python
for block in response.content:
    if block.type == "thinking":
        logger.debug(f"Model thinking: {block.thinking[:100]}...")
        # 不显示给用户，但必须保留在历史消息中！
    elif block.type == "text":
        print(block.text)
    elif block.type == "tool_use":
        result = execute_tool(block.name, block.input)
```

**重要约束**：thinking blocks 必须被完整传回给模型（包含在 assistant 消息中），不能删除，否则 API 会报错。

**考察点**：thinking blocks 的生命周期管理；理解其对工具选择准确率的提升作用。

---

### Q14. 【基础】工具定义中 `description` 字段有多重要？写好描述需要注意什么？

**难度**：★
**类型**：最佳实践

**详细解答**：

`description` 是工具定义中**最重要的字段**，模型完全依赖它来决定何时及如何使用工具。

**糟糕的描述**（导致工具被误用）：
```python
{"name": "execute_query", "description": "执行查询"}  # 太模糊！
```

**优秀的描述**：
```python
{
    "name": "execute_sql_query",
    "description": (
        "在只读数据库上执行 SQL SELECT 查询，返回结果集。"
        "仅支持 SELECT 语句，不支持 INSERT/UPDATE/DELETE。"
        "适用场景：统计分析、数据查询、报表生成。"
        "返回格式：JSON 数组，每行为一条记录。"
        "限制：查询超时 30 秒，结果最多 1000 行。"
    )
}
```

**好描述的要素**：
1. 说明工具的功能和适用场景
2. 说明工具的边界条件和限制
3. 说明输入输出格式
4. 区分与相似工具的不同（如读操作 vs 写操作）

**考察点**：description 直接影响模型的工具选择准确率，是"工具可发现性"（discoverability）的核心。

---

### Q15. 【中级】如何利用工具调用实现可靠的结构化输出（Structured Output）？

**难度**：★★
**类型**：应用模式

**详细解答**：

```python
from anthropic import Anthropic
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    skills: list[str]

def extract_user_profile(text: str) -> UserProfile:
    client = Anthropic()

    # 把 Pydantic model 的 schema 作为工具的 input_schema
    tools = [{
        "name": "extract_user_profile",
        "description": "从文本中提取用户信息，返回结构化数据",
        "input_schema": UserProfile.model_json_schema()
    }]

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_user_profile"},  # 强制调用！
        messages=[{"role": "user", "content": f"从以下文本提取用户信息：\n{text}"}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return UserProfile(**block.input)

    raise ValueError("未能提取用户信息")
```

**TypeScript + Zod 版本**：
```typescript
import { z } from "zod";
import zodToJsonSchema from "zod-to-json-schema";

const UserSchema = z.object({
  name: z.string(),
  age: z.number(),
  email: z.string().email(),
  skills: z.array(z.string()),
});

const response = await client.messages.create({
  model: "claude-opus-4-5",
  max_tokens: 1024,
  tools: [{
    name: "extract_user_profile",
    description: "提取用户信息",
    input_schema: zodToJsonSchema(UserSchema) as any,
  }],
  tool_choice: { type: "tool", name: "extract_user_profile" },
  messages: [{ role: "user", content: `提取用户信息：${text}` }],
});

const toolBlock = response.content.find(b => b.type === "tool_use");
return UserSchema.parse(toolBlock!.input);
```

**考察点**：用工具调用强制结构化输出比 JSON mode 更可靠，因为 Schema 验证在 API 层发生；`tool_choice: "tool"` 确保必然输出。

---

### Q16. 【进阶】什么是工具调用的幂等性？为何在 Agent 中特别重要？

**难度**：★★★
**类型**：设计原则

**详细解答**：

幂等性（Idempotency）：**相同工具调用执行多次，结果相同，不产生额外副作用**。

**幂等工具（安全可重试）**：
```python
def get_user(user_id: str) -> dict:
    # 只读操作，幂等
    return db.users.find_one({"id": user_id})

def set_config(key: str, value: str) -> dict:
    # PUT 语义，设置操作天然幂等
    db.config.upsert({"key": key}, {"$set": {"value": value}})
    return {"success": True}
```

**非幂等工具（危险）**：
```python
def send_email(to: str, subject: str, body: str) -> dict:
    # 每次调用都会发一封新邮件！
    email_service.send(to=to, subject=subject, body=body)

def append_to_log(message: str) -> dict:
    # 重试会导致重复日志
    log_file.append(message)
```

**非幂等工具的安全化方案**：
```python
def send_email_safe(to: str, subject: str, body: str, idempotency_key: str) -> dict:
    # 用幂等 key 防止重复发送
    cache_key = f"email_sent:{idempotency_key}"
    if redis.exists(cache_key):
        return {"sent": True, "skipped": True}
    email_service.send(to=to, subject=subject, body=body)
    redis.setex(cache_key, 3600, "1")
    return {"sent": True}
```

**考察点**：Agent 在错误重试时会重复调用工具，非幂等工具会导致"重复发邮件/重复扣款"等严重问题，必须通过幂等 key 保护。

---

### Q17. 【中级】如何处理工具调用的超时问题？

**难度**：★★★
**类型**：健壮性设计

**详细解答**：

```python
import asyncio
import anthropic

async def execute_tool_with_timeout(
    tool_name: str,
    tool_input: dict,
    timeout: float = 30.0
) -> str:
    try:
        return await asyncio.wait_for(
            run_tool_async(tool_name, tool_input),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # 超时信息作为 tool_result 返回给模型，而不是抛出异常！
        return f"Error: Tool '{tool_name}' timed out after {timeout}s"

async def agent_loop(messages: list, tools: list) -> str:
    client = anthropic.AsyncAnthropic()

    while True:
        response = await client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return extract_text(response)

        # 并行执行所有工具（各自带超时）
        tasks = [
            (block.id, execute_tool_with_timeout(block.name, block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        results = {tid: await task for tid, task in tasks}

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": res}
                for tid, res in results.items()
            ]
        })
```

**TypeScript 版本**：
```typescript
async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  errorMsg: string
): Promise<T> {
  const timer = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(errorMsg)), ms)
  );
  return Promise.race([promise, timer]);
}
```

**考察点**：超时信息应作为 `tool_result` 返回给模型，让模型决定如何处理（重试/降级/报错），而不是在 Python 层直接抛出异常。

---

### Q18. 【进阶】什么是工具调用幻觉？如何检测和缓解？

**难度**：★★★
**类型**：可靠性

**详细解答**：

**工具调用幻觉的三种类型**：

**类型 1：工具名幻觉**（调用不存在的工具）：
```python
# 模型生成了未定义的工具名
ToolUseBlock(name="search_internet", ...)  # 工具列表中没有此工具
```

**类型 2：参数幻觉**（传入未定义的字段）：
```python
ToolUseBlock(name="get_weather", input={
    "city": "北京",
    "humidity": True,   # schema 中未定义
    "wind_speed": "fast" # schema 中未定义
})
```

**类型 3：参数类型错误**：
```python
# schema 要求 int，模型传入 string
ToolUseBlock(name="get_page", input={"page_number": "3"})  # 应为整数 3
```

**检测与缓解**：
```python
from jsonschema import validate, ValidationError

def validate_and_execute(tool_name: str, tool_input: dict, tools: list) -> str:
    # 1. 检查工具是否存在
    tool_def = next((t for t in tools if t["name"] == tool_name), None)
    if not tool_def:
        return f"Error: Unknown tool '{tool_name}'. Available: {[t['name'] for t in tools]}"

    # 2. 验证参数 schema
    try:
        validate(instance=tool_input, schema=tool_def["input_schema"])
    except ValidationError as e:
        return f"Error: Invalid parameters - {e.message}"

    # 3. 执行工具
    return execute_tool(tool_name, tool_input)
```

**最佳实践**：将验证失败信息作为 tool_result 返回，让模型自动修正参数，通常 1-2 次迭代后模型能修正错误。

---

### Q19. 【中级】如何验证模型是否触发了并行工具调用？

**难度**：★★
**类型**：行为验证

**详细解答**：

Claude 默认支持并行工具调用，在一次响应中同时发出多个 tool_use blocks：

```python
def check_parallel_calls(response) -> int:
    tool_calls = [b for b in response.content if b.type == "tool_use"]
    print(f"本次响应包含 {len(tool_calls)} 个工具调用")
    for tc in tool_calls:
        print(f"  [{tc.id}] {tc.name}({tc.input})")
    return len(tool_calls)

# 并行调用示例
messages = [{"role": "user", "content": "同时告诉我北京和上海的天气"}]
response = client.messages.create(model="claude-opus-4-5", ...)
# 模型可能一次性发出两个 get_weather 调用

# 串行倾向示例
messages = [{"role": "user", "content": "先查北京天气，然后再查上海的"}]
# "先...然后..." 措辞倾向于触发串行调用
```

**考察点**：提示词设计影响并行 vs 串行行为；应用层必须能处理多个同时到来的工具调用。

---

### Q20. 【进阶】多轮对话中，工具调用历史应如何管理？

**难度**：★★★
**类型**：状态管理

**详细解答**：

完整的工具调用对话历史结构：

```python
messages = [
    # 第 1 轮
    {"role": "user", "content": "帮我查北京天气"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "我来查一下"},
            {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "北京"}}
        ]
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "32°C 晴"}]
    },
    {"role": "assistant", "content": "北京今天 32°C，晴天"},

    # 第 2 轮（模型有完整历史上下文）
    {"role": "user", "content": "那上海呢？"},
    # 模型知道之前查询过天气，会直接调用 get_weather 查上海
]
```

**上下文压缩（长对话优化）**：
```python
def summarize_old_tool_calls(messages: list, keep_recent: int = 3) -> list:
    """保留最近 N 轮工具调用，压缩早期历史"""
    tool_exchange_indices = []
    i = 0
    while i < len(messages) - 1:
        if (messages[i]["role"] == "assistant" and
            isinstance(messages[i]["content"], list) and
            any(isinstance(b, dict) and b.get("type") == "tool_use"
                for b in messages[i]["content"])):
            tool_exchange_indices.append(i)
            i += 2  # assistant + user(tool_result)
        else:
            i += 1

    if len(tool_exchange_indices) > keep_recent:
        # 用摘要替换旧工具调用对
        cutoff_idx = tool_exchange_indices[-keep_recent]
        summary = f"[已压缩：此前执行了 {len(tool_exchange_indices) - keep_recent} 次工具调用]"
        compressed = [{"role": "system", "content": summary}]
        compressed.extend(messages[cutoff_idx:])
        return compressed

    return messages
```

**考察点**：工具调用历史快速消耗上下文窗口，长对话必须有压缩策略。

---

### Q21. 【进阶】如何实现工具调用的"干运行"（dry run）模式？

**难度**：★★★
**类型**：测试策略

**详细解答**：

干运行让 Agent 完整规划工具调用但不实际执行，用于预览和确认：

```python
class DryRunExecutor:
    def __init__(self):
        self.call_log: list[dict] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.call_log.append({
            "tool": tool_name,
            "input": tool_input,
        })
        return f"[DRY RUN] Would call {tool_name}({tool_input})"

def agent_dry_run(user_message: str, tools: list) -> dict:
    executor = DryRunExecutor()
    messages = [{"role": "user", "content": user_message}]
    client = Anthropic()

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        if response.stop_reason == "end_turn":
            return {
                "would_execute": executor.call_log,
                "final_response_preview": extract_text(response)
            }

        # 用干运行执行器替换真实执行
        results = {}
        for block in response.content:
            if block.type == "tool_use":
                results[block.id] = executor.execute(block.name, block.input)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": res}
                for tid, res in results.items()
            ]
        })

# 使用
plan = agent_dry_run("删除所有超过30天的临时文件", tools)
print("计划执行的操作：")
for call in plan["would_execute"]:
    print(f"  {call['tool']}({call['input']})")
# 用户确认后再执行
```

**考察点**：干运行是 Human-in-the-Loop 设计的重要组成部分，特别适合高风险不可逆操作。

---

### Q22. 【进阶】如何为 Agent 工具调用编写单元测试？

**难度**：★★★
**类型**：测试工程

**详细解答**：

```python
import pytest
from unittest.mock import MagicMock

class MockToolExecutor:
    """工具执行的测试替身，通过依赖注入使 Agent 可测试"""
    def __init__(self, responses: dict):
        self.responses = responses
        self.call_history: list[dict] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.call_history.append({"tool": tool_name, "input": tool_input})
        if tool_name not in self.responses:
            return f"Error: Tool {tool_name} not mocked"
        result = self.responses[tool_name]
        # 支持 callable（动态响应）
        return result(tool_input) if callable(result) else result

@pytest.fixture
def mock_executor():
    return MockToolExecutor({
        "get_weather": '{"temp": 25, "condition": "sunny"}',
        "send_email": '{"success": true, "message_id": "123"}',
    })

def test_weather_query_calls_correct_tool(mock_executor):
    agent = WeatherAgent(tool_executor=mock_executor)
    result = agent.run("北京今天天气怎么样？")

    assert len(mock_executor.call_history) == 1
    assert mock_executor.call_history[0]["tool"] == "get_weather"
    assert mock_executor.call_history[0]["input"]["city"] == "北京"
    assert "25" in result

def test_agent_handles_tool_error_gracefully(mock_executor):
    mock_executor.responses["get_weather"] = "Error: Service unavailable"
    agent = WeatherAgent(tool_executor=mock_executor)
    result = agent.run("北京天气？")

    # 不崩溃，能给出降级回答
    assert result is not None
    assert "error" in result.lower() or "无法" in result

def test_agent_does_not_call_tool_for_simple_question(mock_executor):
    agent = WeatherAgent(tool_executor=mock_executor)
    result = agent.run("你好！")

    # 纯对话不应触发工具调用
    assert len(mock_executor.call_history) == 0
```

**TypeScript 版本**：
```typescript
describe("WeatherAgent", () => {
  const mockExecutor = jest.fn();

  beforeEach(() => mockExecutor.mockReset());

  it("calls get_weather with correct city", async () => {
    mockExecutor.mockResolvedValue('{"temp": 25}');
    const agent = new WeatherAgent({ toolExecutor: mockExecutor });

    await agent.run("北京天气？");

    expect(mockExecutor).toHaveBeenCalledWith(
      "get_weather",
      expect.objectContaining({ city: "北京" })
    );
  });
});
```

**考察点**：依赖注入（工具执行器作为构造参数）是使 Agent 可测试的关键设计；测试应覆盖正常调用、工具错误、不触发工具三种情形。

---
