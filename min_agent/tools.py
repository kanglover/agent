# min_agent/tools.py

# 模拟的本地笔记库
NOTES_DB = {
    "agent": "Agent 是能代替用户独立完成任务的系统，核心是 observe→think→act 循环。",
    "workflow": "Workflow 是步骤固定的自动化流程，由代码控制每一步，AI 只负责当前环节。",
    "context engineering": "上下文工程是研究如何给 AI 提供正确信息的学问，包含 7 个核心组件。",
    "agent loop": "Agent Loop 是 while True 循环，AI 每次决定：调用工具还是直接返回结果。",
    "ai tools": "AI 工具分为对话类、图像类、视频类、编程类等，Claude Code 是编程类代表。",
}


def search_notes(query: str) -> str:
    """搜索笔记库，返回最相关的内容"""
    query_lower = query.lower()
    for key, value in NOTES_DB.items():
        if key in query_lower or query_lower in key:
            return value
    return f"未找到关于「{query}」的笔记"


def write_summary(text: str) -> str:
    """写入摘要，返回确认信息"""
    return f"摘要已保存（{len(text)} 字）：{text[:50]}..."


# Claude API 需要的工具定义格式
TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "description": "搜索本地笔记库，根据关键词查找相关内容",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，例如 agent、workflow"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_summary",
        "description": "将文本保存为摘要",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要保存的摘要内容"}
            },
            "required": ["text"],
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    """根据工具名称分发执行对应工具"""
    if name == "search_notes":
        return search_notes(args["query"])
    elif name == "write_summary":
        return write_summary(args["text"])
    else:
        raise ValueError(f"未知工具：{name}")
