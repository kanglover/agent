# min_agent/tools.py
import json
import time

# ── 常量配置 ──────────────────────────────────────────────────
MAX_RESULTS_DEFAULT = 3
MAX_CHARS_PER_RESULT = 200   # 单条笔记内容最大字符数
MAX_SUMMARY_LENGTH = 500     # write_summary 默认最大字数
TOOL_TIMEOUT_SECONDS = 5     # 工具超时（真实网络调用时生效）
MAX_RETRIES = 2              # 默认重试次数

# ── 模拟笔记库 ────────────────────────────────────────────────
NOTES_DB = {
    "agent": "Agent 是能代替用户独立完成任务的系统，核心是 observe→think→act 循环。",
    "workflow": "Workflow 是步骤固定的自动化流程，由代码控制每一步，AI 只负责当前环节。",
    "context engineering": "上下文工程是研究如何给 AI 提供正确信息的学问，包含 7 个核心组件。",
    "agent loop": "Agent Loop 是 while True 循环，AI 每次决定：调用工具还是直接返回结果。",
    "ai tools": "AI 工具分为对话类、图像类、视频类、编程类等，Claude Code 是编程类代表。",
}


# ── 内部工具函数 ──────────────────────────────────────────────

def _error(code: str, message: str, next_action: str) -> str:
    """构建标准错误 JSON，始终包含 next_action 指导 AI 下一步"""
    return json.dumps(
        {"error": True, "code": code, "message": message, "next_action": next_action},
        ensure_ascii=False,
    )


def _run_with_retry(fn, args: dict, retries: int = MAX_RETRIES) -> str:
    """带指数退避的重试执行，耗尽重试次数返回结构化错误"""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(**args)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
    return _error(
        "RETRY_EXHAUSTED",
        f"工具执行失败（已重试 {retries} 次）：{str(last_error)}",
        "请检查参数是否正确，或稍后再试",
    )


# ── 工具实现 ──────────────────────────────────────────────────

def search_notes(
    query: str,
    max_results: int = MAX_RESULTS_DEFAULT,
    format: str = "concise",
) -> str:
    """
    搜索本地笔记库。

    Args:
        query:       搜索关键词
        max_results: 最多返回几条，默认 3
        format:      "concise"（默认）或 "detailed"（含元数据 JSON）
    Returns:
        concise: "[topic] 内容\n..." 多行文本
        detailed: JSON 字符串，含 total_found / returned / results
        error:    JSON 字符串，含 error/code/next_action
    """
    if not query or not query.strip():
        return _error(
            "INVALID_QUERY",
            "查询关键词不能为空",
            "请提供有效的搜索关键词，例如：search_notes(query='agent')",
        )

    query_lower = query.lower().strip()
    matches = [
        {"topic": key, "content": value}
        for key, value in NOTES_DB.items()
        if key in query_lower or query_lower in key
    ]

    if not matches:
        available = "、".join(NOTES_DB.keys())
        return _error(
            "NOT_FOUND",
            f"未找到关于「{query}」的笔记",
            f"可用笔记主题：{available}。请用以上关键词重试，"
            f"例如：search_notes(query='agent')",
        )

    # 分页：最多返回 max_results 条
    results = matches[:max_results]
    truncated = len(matches) > max_results

    if format == "detailed":
        payload: dict = {
            "total_found": len(matches),
            "returned": len(results),
            "truncated": truncated,
            "results": results,
        }
        if truncated:
            payload["hint"] = (
                f"还有 {len(matches) - max_results} 条结果未显示，"
                f"增大 max_results 参数可获取更多"
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # concise 格式（默认，省 token）
    lines = []
    for r in results:
        content = r["content"]
        if len(content) > MAX_CHARS_PER_RESULT:
            content = content[:MAX_CHARS_PER_RESULT] + "…（内容已截断）"
        lines.append(f"[{r['topic']}] {content}")

    output = "\n".join(lines)
    if truncated:
        output += (
            f"\n（还有 {len(matches) - max_results} 条结果未显示，"
            f"缩小搜索范围或设置 max_results 参数）"
        )
    return output


def write_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    """
    将文本保存为摘要（高风险：写入数据）。

    ⚠️ REQUIRES_CONFIRMATION：此工具会写入数据，请在执行前确认内容正确。

    Args:
        text:       要保存的摘要内容，不能为空
        max_length: 最大保存字数，默认 500，超出部分自动截断
    Returns:
        JSON: {success, chars_saved, preview, requires_confirmation, warning?}
        error JSON: {error, code, next_action}
    """
    if not text or not text.strip():
        return _error(
            "INVALID_INPUT",
            "摘要内容不能为空",
            "请提供有效的文本，例如：write_summary(text='Agent 是能代替用户完成任务的系统...')",
        )

    truncated = False
    if len(text) > max_length:
        text = text[:max_length]
        truncated = True

    result: dict = {
        "success": True,
        "chars_saved": len(text),
        "preview": text[:80] + ("..." if len(text) > 80 else ""),
        "requires_confirmation": True,   # ⚠️ 高风险标记
    }
    if truncated:
        result["warning"] = (
            f"输入内容超过 {max_length} 字，已自动截断。"
            f"如需保存完整内容，请分段调用或增大 max_length 参数。"
        )
    return json.dumps(result, ensure_ascii=False)


# ── Claude API 工具定义 ───────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "x_requires_confirmation": False,   # 只读，无需确认
        "description": """\
搜索本地笔记库，返回匹配的笔记内容。

【使用场景】
- 查找特定主题的学习笔记（写摘要前先搜索背景知识）
- 验证某个概念是否有对应记录

【参数说明】
- query      (必填) 搜索关键词，支持模糊匹配
- max_results (可选, 默认 3) 最多返回几条，范围 1-10
- format     (可选, 默认 "concise") "concise"=省 token | "detailed"=含元数据 JSON

【返回结构 — concise】
  [主题名] 内容文本
  （如有更多结果会提示还剩几条）

【返回结构 — detailed】
  {"total_found": N, "returned": N, "truncated": bool, "results": [...]}

【错误码】
  NOT_FOUND    — 未找到匹配，next_action 列出可用主题
  INVALID_QUERY — 查询词为空，next_action 给出示例

【示例】
  search_notes(query="agent")
  → [agent] Agent 是能代替用户...

  search_notes(query="workflow", format="detailed")
  → {"total_found": 1, "returned": 1, ...}
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如：agent、workflow、context engineering",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回几条结果，默认 3，范围 1-10",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
                "format": {
                    "type": "string",
                    "enum": ["concise", "detailed"],
                    "description": '"concise"（省 token，默认）或 "detailed"（含元数据 JSON）',
                    "default": "concise",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_summary",
        "x_requires_confirmation": True,    # ⚠️ 高风险：写入数据
        "description": """\
将文本保存为摘要，写入本地存储。

⚠️ requires_confirmation=true，这是高风险操作，建议执行前向用户确认。

【使用场景】
- 将搜索到的笔记整理后永久保存
- 记录任务结论或学习成果

【参数说明】
- text       (必填) 要保存的摘要内容，不能为空
- max_length (可选, 默认 500) 最大保存字数，超出部分自动截断

【返回结构 — 成功】
  {
    "success": true,
    "chars_saved": 数字,
    "preview": "前80字...",
    "requires_confirmation": true,
    "warning": "可选，内容被截断时出现"
  }

【错误码】
  INVALID_INPUT — 文本为空，next_action 给出示例

【示例】
  write_summary(text="Agent 是能代替用户完成任务的系统，核心是 observe→think→act。")
  → {"success": true, "chars_saved": 42, "preview": "Agent 是...", "requires_confirmation": true}
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要保存的摘要内容，不能为空",
                },
                "max_length": {
                    "type": "integer",
                    "description": "最大保存字数，默认 500，超出自动截断，范围 10-2000",
                    "default": 500,
                    "minimum": 10,
                    "maximum": 2000,
                },
            },
            "required": ["text"],
        },
    },
]


# ── 分发执行 ──────────────────────────────────────────────────

def run_tool(name: str, args: dict) -> str:
    """
    根据工具名称分发执行，带重试。
    未知工具返回结构化错误（Agent 友好，不 raise）。
    """
    if name == "search_notes":
        return _run_with_retry(search_notes, args)
    elif name == "write_summary":
        return _run_with_retry(write_summary, args)
    else:
        available = "、".join(t["name"] for t in TOOL_DEFINITIONS)
        return _error(
            "UNKNOWN_TOOL",
            f"未知工具：{name}",
            f"可用工具：{available}。请检查工具名称拼写。",
        )
