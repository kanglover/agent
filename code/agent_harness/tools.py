# agent_harness/tools.py
"""
考场工具集 —— Harness 提供给 Agent 使用的工具

类比：
  考试时，考场会统一提供计算器、字典等工具。
  Agent 可以自由选择用哪些，但工具本身由 Harness 控制。

设计思路：
  1. 工具定义（TOOL_DEFINITIONS）：告诉 Agent "你有这些工具可用"
  2. 工具实现（具体函数）：工具的真正逻辑
  3. 工具分发（run_tool）：根据名字找到对应的实现去执行
"""
from __future__ import annotations

import json
import math


# ── 模拟数据库（真实场景替换为数据库 / API）─────────────────────

KNOWLEDGE_BASE = {
    "agent": {
        "title": "Agent 核心概念",
        "content": "Agent 是能代替用户独立完成任务的系统，核心是 observe→think→act 循环。"
                   "与普通 ChatBot 的区别在于 Agent 能自主决策下一步做什么。",
        "tags": ["基础", "架构"],
    },
    "workflow": {
        "title": "Workflow 与 Agent 的区别",
        "content": "Workflow 是步骤固定的自动化流程，由代码控制每一步，AI 只负责当前环节。"
                   "Agent 则是 AI 自己决定走哪条路。",
        "tags": ["基础", "对比"],
    },
    "context engineering": {
        "title": "上下文工程",
        "content": "上下文工程是研究如何给 AI 提供正确信息的学问，包含系统提示、工具描述、"
                   "记忆管理、RAG 检索等 7 个核心组件。",
        "tags": ["进阶", "实践"],
    },
    "react": {
        "title": "ReAct 模式",
        "content": "ReAct (Reasoning + Acting) 让 Agent 交替进行推理和行动。"
                   "每一步先思考（Thought）再行动（Action），然后观察结果（Observation）。",
        "tags": ["架构", "模式"],
    },
    "tool use": {
        "title": "工具使用",
        "content": "Tool Use 是让 AI 调用外部工具（搜索、计算、API 等）的能力。"
                   "Claude 通过 JSON schema 定义工具接口，AI 决定何时调用什么工具。",
        "tags": ["基础", "实践"],
    },
}

CALCULATOR_HISTORY: list[dict] = []


# ── 工具实现 ────────────────────────────────────────────────────

def search_knowledge(query: str, max_results: int = 3) -> str:
    """搜索知识库，返回匹配条目"""
    if not query or not query.strip():
        return json.dumps({
            "error": True, "code": "INVALID_QUERY",
            "message": "查询关键词不能为空",
        }, ensure_ascii=False)

    query_lower = query.lower().strip()
    matches = []
    for key, entry in KNOWLEDGE_BASE.items():
        if query_lower in key or key in query_lower:
            matches.append({"topic": key, **entry})
        elif any(query_lower in tag for tag in entry["tags"]):
            matches.append({"topic": key, **entry})

    if not matches:
        available = ", ".join(KNOWLEDGE_BASE.keys())
        return json.dumps({
            "error": True, "code": "NOT_FOUND",
            "message": f"未找到关于「{query}」的内容",
            "available_topics": available,
        }, ensure_ascii=False)

    results = matches[:max_results]
    return json.dumps({
        "total_found": len(matches),
        "returned": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2)


def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    if not expression or not expression.strip():
        return json.dumps({
            "error": True, "code": "EMPTY_EXPRESSION",
            "message": "表达式不能为空",
        }, ensure_ascii=False)

    # 安全白名单：只允许数字、运算符和数学函数
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "log": math.log,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
        record = {"expression": expression, "result": result}
        CALCULATOR_HISTORY.append(record)
        return json.dumps({
            "success": True,
            "expression": expression,
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": True, "code": "CALC_ERROR",
            "message": f"计算失败: {e}",
        }, ensure_ascii=False)


def write_note(title: str, content: str) -> str:
    """保存笔记（模拟写入）"""
    if not title or not content:
        return json.dumps({
            "error": True, "code": "INVALID_INPUT",
            "message": "标题和内容都不能为空",
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "title": title,
        "chars_saved": len(content),
        "preview": content[:80] + ("..." if len(content) > 80 else ""),
    }, ensure_ascii=False)


# ── Claude API 格式的工具定义 ──────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge",
        "description": (
            "搜索知识库，查找与关键词匹配的条目。"
            "返回 JSON 包含匹配结果的标题、内容和标签。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如 agent、workflow、react",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条，默认 3",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "安全计算数学表达式，支持加减乘除、sqrt、sin、cos、log 等。"
            "返回 JSON 包含表达式和计算结果。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 2+3、sqrt(16)、sin(pi/2)",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "write_note",
        "description": (
            "保存笔记到本地存储。需要提供标题和内容。"
            "返回 JSON 包含保存结果和内容预览。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "笔记标题",
                },
                "content": {
                    "type": "string",
                    "description": "笔记正文内容",
                },
            },
            "required": ["title", "content"],
        },
    },
]


# ── 工具分发器 ──────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "write_note": write_note,
}


def run_tool(name: str, args: dict) -> str:
    """
    根据工具名分发执行

    Args:
        name: 工具名称
        args: 工具参数（字典）

    Returns:
        工具执行结果（JSON 字符串）
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        available = ", ".join(TOOL_REGISTRY.keys())
        return json.dumps({
            "error": True, "code": "UNKNOWN_TOOL",
            "message": f"未知工具: {name}",
            "available_tools": available,
        }, ensure_ascii=False)

    try:
        return fn(**args)
    except TypeError as e:
        return json.dumps({
            "error": True, "code": "INVALID_ARGS",
            "message": f"参数错误: {e}",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": True, "code": "TOOL_ERROR",
            "message": f"工具执行失败: {e}",
        }, ensure_ascii=False)
