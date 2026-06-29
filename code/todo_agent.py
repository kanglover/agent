# todo_agent.py
# TODO 总结 Agent——读取 todo.md，输出四项分析

import anthropic
import os

client = anthropic.Anthropic()

# 定义工具——告诉 AI 它能读取文件
tools = [
    {
        "name": "read_file",
        "description": "读取指定的本地文件内容，返回文件的全部文字",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "要读取的文件名，例如 todo.md"
                }
            },
            "required": ["filename"]
        }
    }
]


def execute_tool(tool_name, tool_input):
    """执行 AI 请求的工具，返回结果字符串"""
    if tool_name == "read_file":
        filename = tool_input["filename"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"错误：找不到文件 {filepath}，请确认文件存在"
    return f"未知工具：{tool_name}"


def run_todo_agent():
    """运行 TODO 总结 Agent"""

    # 给 AI 的初始指令：告诉它要做什么
    messages = [
        {
            "role": "user",
            "content": (
                "请帮我分析我的 TODO 待办清单。"
                "请先读取 todo.md 文件，然后按以下格式输出总结：\n\n"
                "1. 📋 未完成任务清单（列出所有 [ ] 的任务）\n"
                "2. ⚡ 优先级建议（哪些最重要/最紧急，说明理由）\n"
                "3. 📂 按类别分组（每个分类各有哪些未完成任务）\n"
                "4. 📅 今日行动建议（今天应该先做什么）"
            )
        }
    ]

    print("🤖 TODO Agent 启动...\n")

    while True:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        # 情况一：AI 完成分析，输出最终总结
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        # 情况二：AI 要用工具读取文件
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"📖 读取文件：{block.input.get('filename', '')}...\n")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})


# 程序入口：直接运行时执行
if __name__ == "__main__":
    run_todo_agent()
