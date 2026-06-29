"""
计算器 Agent 示例
演示如何用 Tool Use 给 AI 配一个计算器工具

运行方法：
  export ANTHROPIC_API_KEY="你的密钥"
  python calculator_agent.py
"""

import anthropic
import os

client = anthropic.Anthropic()

# ===== 1. 定义工具（给 AI 看的"说明书"）=====
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


# ===== 2. 真正执行计算的函数（AI 下命令，代码执行）=====
def execute_tool(tool_name, tool_input):
    if tool_name == "add":
        result = tool_input["a"] + tool_input["b"]
        return str(result)
    elif tool_name == "multiply":
        result = tool_input["a"] * tool_input["b"]
        return str(result)
    else:
        return f"未知工具: {tool_name}"


# ===== 3. 主流程 =====
def _initialize_conversation(user_question: str):
    """
    初始化对话历史，打印用户问题

    Args:
        user_question: 用户输入的问题

    Returns:
        初始化后的 messages 列表
    """
    messages = [{"role": "user", "content": user_question}]
    print(f"👤 用户：{user_question}\n")
    return messages


def _call_ai_model(messages: list):
    """
    调用 AI 模型进行推理

    Args:
        messages: 对话历史列表

    Returns:
        AI 模型的响应对象
    """
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    return response


def _handle_final_answer(response) -> str:
    """
    处理 AI 返回最终答案的情况

    Args:
        response: AI 模型响应对象

    Returns:
        AI 返回的最终答案文本
    """
    answer = response.content[0].text
    print(f"✅ 最终回答：{answer}")
    return answer


def _execute_tool_requests(response, messages: list) -> list:
    """
    执行 AI 请求的工具调用，将工具结果加入对话历史

    Args:
        response: AI 模型响应对象
        messages: 对话历史列表（会被修改，加入 AI 回复和工具结果）

    Returns:
        工具执行结果列表
    """
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"🔧 AI 使用工具：{block.name}")
            print(f"   参数：{block.input}")

            result = execute_tool(block.name, block.input)
            print(f"   结果：{result}\n")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

    messages.append({"role": "user", "content": tool_results})
    return tool_results


def run_calculator_agent(user_question: str):
    """运行计算器 Agent，接受用户问题，返回最终答案"""

    messages = _initialize_conversation(user_question)

    while True:
        response = _call_ai_model(messages)

        if response.stop_reason == "end_turn":
            return _handle_final_answer(response)

        if response.stop_reason == "tool_use":
            _execute_tool_requests(response, messages)


# ===== 运行示例 =====
if __name__ == "__main__":
    run_calculator_agent("帮我算：25 加 37，然后把结果乘以 2")
