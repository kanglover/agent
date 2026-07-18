"""
min_claude_code.py — Claude Code 最小实现 v2

═══════════════════════════════════════════════════════
核心特点：Edit 工具 + Hook 中间件
═══════════════════════════════════════════════════════

【Edit 工具——Claude Code 最标志性的能力】
普通 write_file 需要 LLM 输出整个文件，而 edit_file：
  - 只输出「哪里要改」：old_str（改前）→ new_str（改后）
  - 找不到 old_str → 报错，防止 LLM 幻觉导致错误覆盖
  - 有多处匹配 → 报错，强制精确定位，避免改错地方
  - 配合 read_file：先读懂内容，再精准改

【Hook 中间件——工具执行的拦截层】
  PreHook:  (tool_name, args) → args | None   返回 None = 阻断执行
  PostHook: (tool_name, args, result) → result  可修改返回值

【write_file vs edit_file】
  write_file：覆盖整个文件（适合新建文件）
  edit_file：精准替换片段（适合修改已有代码）← Claude Code 优先用这个

运行：python code/min_claude_code.py
"""

import os
import subprocess
import textwrap
import anthropic
from typing import Callable, Optional

# ── 类型 ──────────────────────────────────────────────────────
PreHook  = Callable[[str, dict], Optional[dict]]   # None = 阻断
PostHook = Callable[[str, dict, str], str]          # 可修改结果

client = anthropic.Anthropic()


# ══════════════════════════════════════════════════════════════
# 工具实现
# ══════════════════════════════════════════════════════════════

def read_file(path: str) -> str:
    """读取文件，带行号（方便 LLM 精准定位）"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        numbered = "".join(f"{i+1:3d} | {l}" for i, l in enumerate(lines))
        return f"[{path}] {len(lines)} 行\n{numbered}"
    except FileNotFoundError:
        return f"[错误] 文件不存在：{path}"
    except Exception as e:
        return f"[错误] 读取失败：{e}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """
    ★ Claude Code 标志性工具：精准替换文件片段

    工作方式：
      1. 读取文件全文
      2. 查找 old_str（必须完全匹配，包括缩进和换行）
      3. 替换为 new_str
      4. 返回变更摘要（不返回整个文件，省 token）

    为什么比 write_file 更好：
      - LLM 只需输出变化的部分，不用记住整个文件
      - 有验证机制，找不到就报错，不会静默写错
      - 适合修改大文件中的一小段
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"[错误] 文件不存在：{path}，请先用 write_file 创建"
    except Exception as e:
        return f"[错误] 读取失败：{e}"

    # 验证：old_str 必须存在
    if old_str not in content:
        preview = old_str[:80].replace("\n", "↵")
        return (
            f"[错误] 未找到要替换的内容，请重新 read_file 确认准确文本：\n"
            f"  期望找到：「{preview}」\n"
            f"  提示：注意空格、缩进、换行是否完全一致"
        )

    # 验证：不允许多处匹配（避免改错地方）
    count = content.count(old_str)
    if count > 1:
        return (
            f"[错误] 找到 {count} 处匹配，请在 old_str 中加入更多上下文行来精确定位"
        )

    # 执行替换
    new_content = content.replace(old_str, new_str, 1)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"[错误] 写入失败：{e}"

    # 返回变更摘要
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    delta     = new_lines - old_lines
    sign      = f"+{delta}" if delta >= 0 else str(delta)
    return (
        f"[成功] {path} 已修改（净变化 {sign} 行）\n"
        f"  改前：{old_str.strip()[:60]}\n"
        f"  改后：{new_str.strip()[:60]}"
    )


def write_file(path: str, content: str) -> str:
    """写入整个文件（仅用于新建文件，修改已有文件请用 edit_file）"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[成功] 已创建 {path}（{content.count(chr(10)) + 1} 行）"
    except Exception as e:
        return f"[错误] 写入失败：{e}"


def bash_exec(command: str) -> str:
    """执行 shell 命令，超时 15 秒"""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        out = (r.stdout + r.stderr).strip() or "(无输出)"
        return f"[exit {r.returncode}]\n{out}"
    except subprocess.TimeoutExpired:
        return "[错误] 命令超时（15 秒）"
    except Exception as e:
        return f"[错误] 执行失败：{e}"


# ── 工具注册表 ────────────────────────────────────────────────
TOOLS: dict[str, Callable] = {
    "read_file":  read_file,
    "edit_file":  edit_file,
    "write_file": write_file,
    "bash_exec":  bash_exec,
}

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "读取文件内容（带行号）。修改文件前必须先 read_file 确认准确文本。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "★ 首选修改工具：精准替换文件中的一段文本。\n"
            "必须先 read_file 拿到准确的 old_str（包括缩进/换行），再调用。\n"
            "old_str 找不到或有多处匹配时会报错。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "文件路径"},
                "old_str": {"type": "string", "description": "要替换的原始文本（须与文件完全一致）"},
                "new_str": {"type": "string", "description": "替换后的新文本"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "write_file",
        "description": "写入整个文件（仅用于新建文件，修改已有文件请用 edit_file）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash_exec",
        "description": "执行 shell 命令，返回 stdout + stderr。",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


# ══════════════════════════════════════════════════════════════
# Hook 系统
# ══════════════════════════════════════════════════════════════

def run_pre_hooks(hooks: list[PreHook], name: str, args: dict) -> Optional[dict]:
    """依次执行 PreHook 链。任意 Hook 返回 None → 立即阻断"""
    for hook in hooks:
        args = hook(name, args)
        if args is None:
            return None
    return args


def run_post_hooks(hooks: list[PostHook], name: str, args: dict, result: str) -> str:
    """依次执行 PostHook 链，每个 Hook 可修改 result"""
    for hook in hooks:
        result = hook(name, args, result)
    return result


# ── 内置 Hook 示例 ────────────────────────────────────────────

def audit_pre_hook(tool_name: str, args: dict) -> Optional[dict]:
    """审计 Hook：打印每次工具调用（重点展示 edit_file 的差量信息）"""
    if tool_name == "edit_file":
        old = args.get("old_str", "")[:50].replace("\n", "↵")
        new = args.get("new_str", "")[:50].replace("\n", "↵")
        print(f"  [HOOK] edit_file: 「{old}」→「{new}」")
    else:
        first_val = next(iter(args.values()), "")
        print(f"  [HOOK] {tool_name}: {str(first_val)[:70]}")
    return args


def block_dangerous_pre_hook(tool_name: str, args: dict) -> Optional[dict]:
    """安全 Hook：阻断危险命令"""
    if tool_name == "bash_exec":
        cmd = args.get("command", "")
        for danger in ["rm -rf /", "sudo rm", ":(){:|:&};:", "> /dev/sda"]:
            if danger in cmd:
                print(f"  [HOOK:BLOCKED] 危险命令：{cmd[:60]}")
                return None
    return args


# ══════════════════════════════════════════════════════════════
# Agent Loop
# ══════════════════════════════════════════════════════════════

def run_agent(
    task: str,
    pre_hooks:  list[PreHook]  = [],
    post_hooks: list[PostHook] = [],
    max_steps:  int = 15,
) -> str:
    """
    Claude Code Agent Loop（带 Hook 中间件）

    每轮：LLM 决策 → PreHook 链 → 执行工具 → PostHook 链 → 更新上下文 → 循环
    """
    messages = [{"role": "user", "content": task}]

    for step in range(max_steps):
        print(f"\n── Step {step + 1} {'─' * 44}")

        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            system=(
                "你是 Claude Code，一个专业的编程助手。\n"
                "修改文件时优先使用 edit_file（精准替换），不要用 write_file 覆盖整个文件。\n"
                "使用 edit_file 前必须先 read_file 确认原始文本。"
            ),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final = next((b.text for b in response.content if hasattr(b, "text")), "完成")
            print(f"\n✅ 完成：{final[:300]}")
            return final

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                name = block.name
                args = dict(block.input)
                print(f"🔧 {name}")

                args = run_pre_hooks(pre_hooks, name, args)
                if args is None:
                    result = f"[阻断] {name} 被安全策略拒绝"
                    print(f"  🚫 {result}")
                else:
                    fn     = TOOLS.get(name)
                    result = fn(**args) if fn else f"[错误] 未知工具：{name}"
                    result = run_post_hooks(post_hooks, name, args, result)
                    print(f"  → {result.splitlines()[0]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})

    return f"[超时] 超过 {max_steps} 步"


# ══════════════════════════════════════════════════════════════
# 演示：修复有 bug 的计算器
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 准备一个有 bug 的目标文件
    demo_file = "/tmp/demo_calc.py"
    with open(demo_file, "w") as f:
        f.write(textwrap.dedent("""\
            def add(a, b):
                return a - b          # bug: 应该是 a + b

            def multiply(a, b):
                result = 0
                for i in range(b):
                    result += a
                return result

            if __name__ == "__main__":
                print("add(3,4) =",      add(3, 4))       # 期望 7
                print("multiply(3,4) =", multiply(3, 4))  # 期望 12
        """))

    print("=" * 60)
    print("Claude Code 最小实现 v2 — Edit 工具演示")
    print("=" * 60)
    print(f"目标文件：{demo_file}")
    print("任务：找到 bug → 用 edit_file 精准修复 → 运行验证")
    print()

    run_agent(
        task=(
            f"请修复 {demo_file} 中的 bug：\n"
            "1. read_file 读取文件，找到 bug 所在行\n"
            "2. edit_file 精准修复（不要用 write_file 覆盖整个文件）\n"
            "3. bash_exec 运行文件，确认输出正确"
        ),
        pre_hooks=[audit_pre_hook, block_dangerous_pre_hook],
    )
