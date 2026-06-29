# tools_v1.py — 基础版
#
# 特点：简单直接，把 API 包一层就叫工具
# 缺点：
#   - 返回纯字符串，AI 难以解析结构
#   - 出错只会 raise 或返回模糊文字，AI 不知道下一步该怎么做
#   - 没有分页：数据多了会塞爆 AI 的上下文
#   - 没有截断：单条内容再长也全返回
#   - 没有重试：网络抖一下就直接失败
#   - 工具描述太简单：AI 不知道怎么正确使用

# ── 笔记库 ────────────────────────────────────────────────────
NOTES_DB = {
    "agent": "Agent 是能代替用户独立完成任务的系统，核心是 observe→think→act 循环。",
    "workflow": "Workflow 是步骤固定的自动化流程，由代码控制每一步，AI 只负责当前环节。",
    "context engineering": "上下文工程是研究如何给 AI 提供正确信息的学问，包含 7 个核心组件。",
    "agent loop": "Agent Loop 是 while True 循环，AI 每次决定：调用工具还是直接返回结果。",
    "ai tools": "AI 工具分为对话类、图像类、视频类、编程类等，Claude Code 是编程类代表。",
}


# ── 工具函数 ──────────────────────────────────────────────────

def search_notes(query: str) -> str:
    """搜索笔记，返回第一个匹配结果"""
    for key, value in NOTES_DB.items():
        if key in query.lower() or query.lower() in key:
            return value
    # ❌ 问题：AI 拿到这段话，不知道该用哪些关键词重试
    return f"没有找到关于 {query} 的内容"


def write_summary(text: str) -> str:
    """保存摘要"""
    # ❌ 问题：没有输入校验，空字符串也会"成功保存"
    # ❌ 问题：没有截断，text 有多长就保存多长
    return f"已保存：{text}"


# ── Claude API 工具定义 ───────────────────────────────────────
# ❌ 问题：描述太简单，AI 不知道参数格式、返回结构、出错怎么办

TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "description": "搜索笔记",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_summary",
        "description": "保存摘要",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"],
        },
    },
]


# ── 分发执行 ──────────────────────────────────────────────────

def run_tool(name: str, args: dict) -> str:
    if name == "search_notes":
        return search_notes(args["query"])
    elif name == "write_summary":
        return write_summary(args["text"])
    else:
        # ❌ 问题：raise 会让 Agent 崩溃，而不是优雅地报告问题
        raise ValueError(f"未知工具：{name}")
