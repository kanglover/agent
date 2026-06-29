# tools_v2.py — 增强版
#
# 遵循 Anthropic 工具设计最佳实践：
#   ✅ 结构化 JSON 返回，AI 能可靠解析
#   ✅ 每个错误都有 next_action，告诉 AI 下一步该怎么做
#   ✅ 分页（max_results）：控制返回条数，不塞爆上下文
#   ✅ 截断（MAX_CHARS_PER_RESULT / max_length）：单条内容有上限
#   ✅ 重试（_run_with_retry）：网络抖动自动恢复
#   ✅ requires_confirmation：高风险工具有明确标记
#   ✅ 丰富的工具描述：AI 知道怎么用、出错了怎么改
import json
import time

# ── 常量 ──────────────────────────────────────────────────────
MAX_RESULTS_DEFAULT = 3      # 默认最多返回几条搜索结果
MAX_CHARS_PER_RESULT = 200   # 单条笔记内容最多显示多少字
MAX_SUMMARY_LENGTH = 500     # write_summary 默认最大字数
TOOL_TIMEOUT_SECONDS = 5     # 工具执行超时（接入真实 API 时生效）
MAX_RETRIES = 2              # 自动重试次数

# ── 笔记库（和 v1 相同，数据不变）─────────────────────────────
NOTES_DB = {
    "agent": "Agent 是能代替用户独立完成任务的系统，核心是 observe→think→act 循环。",
    "workflow": "Workflow 是步骤固定的自动化流程，由代码控制每一步，AI 只负责当前环节。",
    "context engineering": "上下文工程是研究如何给 AI 提供正确信息的学问，包含 7 个核心组件。",
    "agent loop": "Agent Loop 是 while True 循环，AI 每次决定：调用工具还是直接返回结果。",
    "ai tools": "AI 工具分为对话类、图像类、视频类、编程类等，Claude Code 是编程类代表。",
}


# ── 内部工具函数 ──────────────────────────────────────────────

def _error(code: str, message: str, next_action: str) -> str:
    """
    ✅ 统一错误格式。
    每个错误都告诉 AI：发生了什么（message）+ 下一步怎么做（next_action）。
    AI 读到 next_action 就知道该怎么修正参数或换个策略，而不是一脸懵。
    """
    return json.dumps(
        {"error": True, "code": code, "message": message, "next_action": next_action},
        ensure_ascii=False,
    )


def _run_with_retry(fn, args: dict, retries: int = MAX_RETRIES) -> str:
    """
    ✅ 带退避的重试执行。
    网络抖动或临时错误时自动重试，耗尽次数后返回结构化错误（不 crash）。
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn(**args)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))  # 简单线性退避
    return _error(
        "RETRY_EXHAUSTED",
        f"执行失败，已重试 {retries} 次：{last_error}",
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

    ✅ 改进 1：空 query 提前校验，给出具体示例
    ✅ 改进 2：未找到时告诉 AI 可用的关键词，不让 AI 瞎猜
    ✅ 改进 3：max_results 分页，防止一次返回太多内容
    ✅ 改进 4：内容超长自动截断，不塞爆上下文
    ✅ 改进 5：format 参数——concise 省 token，detailed 含元数据
    """
    # 校验：query 不能为空
    if not query or not query.strip():
        return _error(
            "INVALID_QUERY",
            "查询关键词不能为空",
            "请提供有效关键词，例如：search_notes(query='agent')",
        )

    query_lower = query.lower().strip()
    matches = [
        {"topic": key, "content": value}
        for key, value in NOTES_DB.items()
        if key in query_lower or query_lower in key
    ]

    # 未找到：给 AI 列出可用主题
    if not matches:
        available = "、".join(NOTES_DB.keys())
        return _error(
            "NOT_FOUND",
            f"未找到关于「{query}」的笔记",
            f"可用主题：{available}。请用这些关键词重试，例如：search_notes(query='agent')",
        )

    # 分页：超出 max_results 时截断，并告知 AI 还有多少条
    results = matches[:max_results]
    truncated = len(matches) > max_results

    # detailed 格式：返回完整 JSON 元数据（AI 需要知道 total_found 等信息时用）
    if format == "detailed":
        payload = {
            "total_found": len(matches),
            "returned": len(results),
            "truncated": truncated,
            "results": results,
        }
        if truncated:
            payload["hint"] = (
                f"还有 {len(matches) - max_results} 条未显示，"
                f"增大 max_results 可获取更多"
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # concise 格式（默认）：省 token，每条一行
    lines = []
    for r in results:
        content = r["content"]
        # 截断过长的单条内容
        if len(content) > MAX_CHARS_PER_RESULT:
            content = content[:MAX_CHARS_PER_RESULT] + "…（内容已截断）"
        lines.append(f"[{r['topic']}] {content}")

    output = "\n".join(lines)
    if truncated:
        output += (
            f"\n（还有 {len(matches) - max_results} 条未显示，"
            f"缩小关键词或设置 max_results 参数）"
        )
    return output


def write_summary(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    """
    将文本保存为摘要（高风险工具，标记 requires_confirmation）。

    ✅ 改进 1：空文本提前校验，给出具体示例
    ✅ 改进 2：超长内容自动截断，并在响应里说明（而不是悄悄截掉）
    ✅ 改进 3：结构化 JSON 返回，AI 知道保存了多少字、预览是什么
    ✅ 改进 4：requires_confirmation=True 标记高风险，调用方可拦截确认
    """
    # 校验：文本不能为空
    if not text or not text.strip():
        return _error(
            "INVALID_INPUT",
            "摘要内容不能为空",
            "请提供有效文本，例如：write_summary(text='Agent 是...')",
        )

    # 截断超长内容，并告知 AI
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length]

    result = {
        "success": True,
        "chars_saved": len(text),
        "preview": text[:80] + ("..." if len(text) > 80 else ""),
        "requires_confirmation": True,  # ⚠️ 高风险标记：写入操作，建议向用户确认
    }
    if truncated:
        result["warning"] = (
            f"内容超过 {max_length} 字已截断。"
            f"如需保存完整内容，请分段调用或增大 max_length 参数。"
        )
    return json.dumps(result, ensure_ascii=False)


# ── Claude API 工具定义 ───────────────────────────────────────
# ✅ 每个工具都有：使用场景、参数说明、返回结构、错误码、示例
# ✅ AI 读完定义就知道怎么用，不需要靠猜

TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "x_requires_confirmation": False,   # ✅ 只读，无需确认
        "description": """\
搜索本地笔记库，返回匹配的笔记内容。

【使用场景】
- 查找特定主题的学习笔记
- 在写摘要前先查相关背景知识

【参数说明】
- query       (必填) 搜索关键词，支持模糊匹配
- max_results (可选, 默认 3) 最多返回几条，范围 1-10
- format      (可选, 默认 "concise") "concise"=省 token | "detailed"=含元数据 JSON

【返回结构 — concise（默认）】
  [主题名] 内容文本
  （若还有更多结果，会提示条数）

【返回结构 — detailed】
  {"total_found": N, "returned": N, "truncated": bool, "results": [...]}

【错误码】
  NOT_FOUND     — 未找到匹配，next_action 列出所有可用主题
  INVALID_QUERY — 关键词为空，next_action 给出使用示例

【示例】
  search_notes(query="agent")
  → [agent] Agent 是能代替用户...

  search_notes(query="workflow", format="detailed")
  → {"total_found": 1, "returned": 1, "truncated": false, ...}
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
                    "description": "最多返回几条，默认 3，范围 1-10",
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
        "x_requires_confirmation": True,    # ✅ 高风险：写入数据，需要标记
        "description": """\
将文本保存为摘要，写入本地存储。

⚠️ requires_confirmation=true：此工具会写入数据，执行前建议向用户确认。

【使用场景】
- 将整理好的笔记内容永久保存
- 记录任务结论

【参数说明】
- text       (必填) 要保存的摘要文本，不能为空
- max_length (可选, 默认 500) 最大字数，超出部分自动截断

【返回结构 — 成功】
  {
    "success": true,
    "chars_saved": 数字,
    "preview": "前 80 字...",
    "requires_confirmation": true,
    "warning": "可选，内容被截断时出现"
  }

【错误码】
  INVALID_INPUT — 文本为空，next_action 给出使用示例

【示例】
  write_summary(text="Agent 是能代替用户完成任务的系统...")
  → {"success": true, "chars_saved": 30, "requires_confirmation": true}
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
    ✅ 未知工具返回结构化错误，不 raise。
    AI 读到错误后知道可用工具有哪些，可以自行修正。
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
