# code/context_builder/layers.py
from dataclasses import dataclass, field


@dataclass
class SystemLayer:
    """层 1：角色定义、行为规则、可用工具（固定，不随任务变）"""
    role: str
    instructions: list[str]
    tool_names: list[str] = field(default_factory=list)


@dataclass
class TaskLayer:
    """层 2：当前任务描述、目标、约束（每次任务确定后不变）"""
    description: str
    goal: str
    constraints: list[str] = field(default_factory=list)


@dataclass
class MemoryItem:
    key: str
    value: str


@dataclass
class MemoryLayer:
    """层 3：跨任务持久化知识（用户偏好、历史摘要等）"""
    items: list[MemoryItem] = field(default_factory=list)


@dataclass
class EvidenceItem:
    source: str       # 来源标识（文件名、URL 等）
    summary: str      # 压缩后的摘要（或原文，若未超限）
    ref_id: str       # 可追溯引用 ID（"ref:xxxxxxxx"，未压缩时为 ""）
    char_count: int   # 原始字数（透明度）


@dataclass
class EvidenceLayer:
    """层 4：检索到的外部证据（笔记、文档、搜索结果）—— 超长自动压缩"""
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class TraceStep:
    step: int
    tool: str
    args: dict
    result_summary: str   # 工具结果摘要（超长时压缩）
    result_ref: str       # 可追溯引用 ID（未压缩时为 ""）
    result_char_count: int  # 原始结果字数


@dataclass
class TraceLayer:
    """层 5：最近的工具调用记录 —— 完整日志不直接塞进 prompt"""
    steps: list[TraceStep] = field(default_factory=list)


@dataclass
class ContextBundle:
    """五层 Context 的顶层容器，由 ContextBuilder.build() 生成"""
    system: SystemLayer
    task: TaskLayer
    memory: MemoryLayer
    evidence: EvidenceLayer
    trace: TraceLayer
