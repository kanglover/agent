# Workflow vs Agent 代码对比

> 沉淀时间：2026-06-24
> 场景：用户发来消息，系统帮他解决客服问题

---

## 一句话总结

**Workflow**：你写了一个剧本，AI 按台词念
**Agent**：你给了 AI 一个工具箱，它自己决定怎么演

---

## 核心区别一览

| | Workflow | Agent |
|--|----------|-------|
| **流程控制者** | 代码写死了顺序 | AI 自己决定 |
| **AI 调用几次** | 固定次数 | 不确定，直到完成才停 |
| **能不能跳步骤** | 不能 | 可以 |
| **工具怎么用** | 代码决定调哪个 | AI 自己选 |
| **关键结构** | `step1 → step2 → step3` | `while True` + 判断是否完成 |
| **AI 的角色** | 流水线上的工人 | 独立工作的员工 |

---

## Workflow 版本

**特点**：每一步都由「代码」决定，AI 只负责当前被交到手里的任务。

```python
import anthropic

client = anthropic.Anthropic()

def step1_classify(user_message):
    """第1步：固定让 AI 判断消息类型"""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"判断这条消息是「投诉」还是「咨询」，只回答一个词：{user_message}"
        }]
    )
    return response.content[0].text  # 返回 "投诉" 或 "咨询"


def step2_find_answer(user_message, category):
    """第2步：固定让 AI 生成回复"""
    prompt = f"用户{'投诉' if category == '投诉' else '咨询'}：{user_message}，请生成一段回复"
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def step3_format_reply(answer):
    """第3步：固定让 AI 把回复格式化得更礼貌"""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"把这段回复改得更礼貌、更专业：{answer}"
        }]
    )
    return response.content[0].text


def workflow_handle(user_message):
    """
    整个流程由代码控制，永远走这三步：
    分类 → 生成回复 → 格式化
    不管发生什么，都不会改变顺序
    """
    print("▶ 第1步：分类消息...")
    category = step1_classify(user_message)
    print(f"  结果：{category}")

    print("▶ 第2步：生成回复...")
    answer = step2_find_answer(user_message, category)

    print("▶ 第3步：格式化回复...")
    final = step3_format_reply(answer)

    print("✅ 完成！")
    return final


# 运行
result = workflow_handle("我买的手机坏了，要求退款！")
```

### 执行轨迹（永远一样）

```
▶ 第1步：分类消息...
  结果：投诉
▶ 第2步：生成回复...
▶ 第3步：格式化回复...
✅ 完成！
```

> ⚠️ 代码控制一切，AI 不知道自己在做哪步，更不能改变流程。

---

## Agent 版本

**特点**：代码只提供「工具箱」，AI 自己决定用什么工具、走什么路，直到任务完成。

```python
import anthropic

client = anthropic.Anthropic()

# --- 工具箱：AI 可以自由选择用哪个 ---

def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 状态：已发货，预计明天到达"

def process_refund(order_id: str) -> str:
    """处理退款"""
    return f"订单 {order_id} 退款已提交，3个工作日内到账"

def escalate_to_human(reason: str) -> str:
    """转接人工客服"""
    return f"已转接人工客服，原因：{reason}"


# 把工具的「说明书」告诉 AI
tools = [
    {
        "name": "check_order_status",
        "description": "查询订单的当前状态",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "process_refund",
        "description": "为用户办理退款",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "当问题复杂时，转接给人工客服",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "转接原因"}
            },
            "required": ["reason"]
        }
    }
]

def run_tool(tool_name, tool_input):
    """执行 AI 选择的工具"""
    if tool_name == "check_order_status":
        return check_order_status(tool_input["order_id"])
    elif tool_name == "process_refund":
        return process_refund(tool_input["order_id"])
    elif tool_name == "escalate_to_human":
        return escalate_to_human(tool_input["reason"])


def agent_handle(user_message):
    """
    Agent 的核心：一个「循环」
    AI 每次自己决定：用工具？还是直接回复用户？
    直到 AI 觉得任务完成，才停下来
    """
    messages = [{"role": "user", "content": user_message}]

    # ⬇️ 这个 while 循环就是 Agent 的灵魂
    while True:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1000,
            system="你是一个客服 Agent，根据用户情况自己判断该查订单、退款、还是转人工。",
            tools=tools,
            messages=messages
        )

        # AI 决定：直接回复用户（任务完成，退出循环）
        if response.stop_reason == "end_turn":
            final_reply = response.content[0].text
            print(f"Agent 最终回复：{final_reply}")
            return final_reply

        # AI 决定：调用工具（继续循环）
        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    print(f"▶ AI 决定调用工具：{block.name}，参数：{block.input}")

                    # 执行工具，拿到结果
                    result = run_tool(block.name, block.input)
                    print(f"  工具返回：{result}\n")

                    # 把工具结果告诉 AI，让它继续决策
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        }]
                    })


# 运行
agent_handle("我的订单 A123 到哪了？如果还没发货我要退款")
```

### 执行轨迹（每次可能不同）

**情况一：订单已发货**
```
用户：我的订单 A123 到哪了？如果还没发货我要退款

▶ AI 决定调用工具：check_order_status，参数：{"order_id": "A123"}
  工具返回：订单 A123 状态：已发货，预计明天到达

Agent 最终回复：您好！您的订单 A123 已经发货了，
预计明天就能到达，不需要退款哦～
```

**情况二：用户情绪激动**
```
用户：我要投诉，你们服务太差了！我要找你们负责人！

▶ AI 决定调用工具：escalate_to_human，参数：{"reason": "用户情绪激动，要求找负责人"}
  工具返回：已转接人工客服

Agent 最终回复：非常抱歉给您带来了不好的体验，
我已经为您转接人工客服，请稍候...
```

> ✅ 同样的代码结构，面对不同情况走完全不同的路——这就是 Agent 的核心价值。

---

## Agent 代码的两个关键结构

### 1. `while True` 循环 —— Agent 的灵魂

```python
while True:
    # AI 思考下一步
    response = client.messages.create(...)

    if response.stop_reason == "end_turn":
        # AI 说：我完成了 → 退出循环
        break

    if response.stop_reason == "tool_use":
        # AI 说：我要用工具 → 执行工具，继续循环
        ...
```

Workflow 没有这个循环——步骤执行完就结束了。

### 2. `stop_reason` 判断 —— AI 用来「表达意图」的信号

| stop_reason | 含义 |
|-------------|------|
| `"end_turn"` | AI 认为任务完成，直接回复用户 |
| `"tool_use"` | AI 想调用工具，还没完成 |

---

## 什么时候用哪个？

### 用 Workflow 的判断标准
- ✅ 步骤固定，永远这几步
- ✅ 对稳定性要求高，不能出错
- ✅ 成本敏感，需要控制 AI 调用次数
- 举例：翻译流水线、定时报表生成、固定格式的数据处理

### 用 Agent 的判断标准
- ✅ 步骤不确定，要根据情况动态决定
- ✅ 需要反复尝试、从错误中修正
- ✅ 工具调用不固定，AI 自己判断用哪个
- 举例：客服系统、编程助手、行业调研、复杂任务处理

---

## Workflow 的三种现实形态

| 形态 | 工具 | 特点 |
|------|------|------|
| **代码函数链** | Python / 任意语言 | 最灵活，需要写代码 |
| **可视化拖拽** | Dify、Coze、n8n | 不需要写代码，适合非技术人员 |
| **带条件分支** | 以上均可 | 有 if-else 判断，但分支是提前写好的 |

---

*相关概念：Workflow、Agent、Tool Use、while loop、stop_reason、Claude API*

*相关文档：[[agent-workflow-核心概念]] [[agent-openai实践指南]]*
