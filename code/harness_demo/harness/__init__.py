"""harness: Agent 基础设施演示包。

六大子系统:
- llm          LLM 接口层（脚本式 MockLLM）
- context      上下文管理（消息历史、token 估算、摘要压缩）
- tools        工具系统（注册表 + 计算器/搜索/读文件）
- memory       记忆与状态（短期/长期记忆、任务状态、快照恢复）
- observer     评估观测（trace、指标、运行摘要）
- constraints  约束与恢复（步数/token/超时限制、重试/降级）
- orchestrator 执行编排（think → act → observe 主循环）
"""

from .llm import MockLLM, LLMResponse, ToolCall, TokenUsage, default_script
from .context import ContextManager, Message
from .tools import Tool, ToolRegistry, ToolResult, CalculatorTool, SearchTool, FileReadTool
from .memory import Memory, AgentState
from .observer import Observer, TraceEvent
from .constraints import Constraint, ConstraintManager, Recovery
from .orchestrator import Orchestrator, AbortRunError

__version__ = "0.1.0"

__all__ = [
    # llm
    "MockLLM", "LLMResponse", "ToolCall", "TokenUsage", "default_script",
    # context
    "ContextManager", "Message",
    # tools
    "Tool", "ToolRegistry", "ToolResult", "CalculatorTool", "SearchTool", "FileReadTool",
    # memory
    "Memory", "AgentState",
    # observer
    "Observer", "TraceEvent",
    # constraints
    "Constraint", "ConstraintManager", "Recovery",
    # orchestrator
    "Orchestrator", "AbortRunError",
]
