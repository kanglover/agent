# code/context_builder/renderer.py
import json
from context_builder.layers import ContextBundle

_SEP = "\n" + "─" * 40 + "\n"


def render(bundle: ContextBundle) -> str:
    """
    把 ContextBundle 渲染成结构化 prompt 字符串（纯函数，无副作用）。

    顺序固定：system → task → memory → evidence → trace
    空层自动跳过。同一 bundle 重复调用必然得到完全相同的输出。
    """
    sections: list[str] = []

    # ── 1. SYSTEM ───────────────────────────────────────────
    lines = ["[SYSTEM]", f"role: {bundle.system.role}"]
    for i, instr in enumerate(bundle.system.instructions, 1):
        lines.append(f"  {i}. {instr}")
    if bundle.system.tool_names:
        lines.append(f"tools: {', '.join(bundle.system.tool_names)}")
    sections.append("\n".join(lines))

    # ── 2. TASK ─────────────────────────────────────────────
    lines = [
        "[TASK]",
        f"description: {bundle.task.description}",
        f"goal: {bundle.task.goal}",
    ]
    if bundle.task.constraints:
        lines.append("constraints:")
        for c in bundle.task.constraints:
            lines.append(f"  - {c}")
    sections.append("\n".join(lines))

    # ── 3. MEMORY（可选）────────────────────────────────────
    if bundle.memory.items:
        lines = ["[MEMORY]"]
        for item in bundle.memory.items:
            lines.append(f"  {item.key}: {item.value}")
        sections.append("\n".join(lines))

    # ── 4. RETRIEVED EVIDENCE（可选）────────────────────────
    if bundle.evidence.items:
        lines = ["[RETRIEVED EVIDENCE]"]
        for ev in bundle.evidence.items:
            ref_note = f" | ref: {ev.ref_id}" if ev.ref_id else ""
            lines.append(f"  source: {ev.source}{ref_note} | {ev.char_count} chars")
            lines.append(f"  summary: {ev.summary}")
            lines.append("")
        sections.append("\n".join(lines).rstrip())

    # ── 5. RECENT TRACE（可选）──────────────────────────────
    if bundle.trace.steps:
        lines = ["[RECENT TRACE]"]
        for s in bundle.trace.steps:
            args_str = json.dumps(s.args, ensure_ascii=False, sort_keys=True)
            ref_note = f" | ref: {s.result_ref}" if s.result_ref else ""
            lines.append(
                f"  step {s.step} | tool: {s.tool} | args: {args_str}{ref_note}"
            )
            lines.append(f"    result: {s.result_summary}")
        sections.append("\n".join(lines))

    return _SEP.join(sections)
