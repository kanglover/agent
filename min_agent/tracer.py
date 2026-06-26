# min_agent/tracer.py
import json
from datetime import datetime
from pathlib import Path

TRACE_FILE = Path("trace.jsonl")

# claude-opus-4-8 定价
INPUT_PRICE_PER_M = 15    # $15 / 百万 token
OUTPUT_PRICE_PER_M = 75   # $75 / 百万 token


def build_trace_entry(
    step: int,
    thought_summary: str,
    tool: str | None,
    args: dict | None,
    observation: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """构建一条标准 trace 记录"""
    cost = (input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
    return {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "thought_summary": thought_summary,
        "tool": tool,
        "args": args,
        "observation": observation,
        "cost_estimate": round(cost, 6),
    }


def write_trace(entry: dict) -> None:
    """将一条 trace 记录追加写入 JSONL 文件"""
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
