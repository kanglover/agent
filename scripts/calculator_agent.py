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
def run_calculator_agent(user_question: str):
    """运行计算器 Agent，接受用户问题，返回最终答案"""

    messages = [{"role": "user", "content": user_question}]

    print(f"👤 用户：{user_question}\n")

    while True:
        # 让 AI 思考
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        # 情况一：AI 完成了任务，给出最终答案
        if response.stop_reason == "end_turn":
            answer = response.content[0].text
            print(f"✅ 最终回答：{answer}")
            return answer

        # 情况二：AI 想用工具
        if response.stop_reason == "tool_use":
            # 把 AI 的回应记录到历史
            messages.append({"role": "assistant", "content": response.content})

            # 执行 AI 请求的每一个工具
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

            # 把工具结果告诉 AI
            messages.append({"role": "user", "content": tool_results})


# ===== 运行示例 =====
if __name__ == "__main__":
    run_calculator_agent("帮我算：25 加 37，然后把结果乘以 2")
