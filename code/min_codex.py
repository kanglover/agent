"""
min_codex.py — Codex CLI 最小实现 v2

═══════════════════════════════════════════════════════
核心特点：apply_patch（差量修改）+ Approval 审批模式
═══════════════════════════════════════════════════════

【apply_patch——Codex CLI 最标志性的能力】
其他 Agent 用 write_file（输出整个文件）。Codex 让 LLM 输出 diff（只描述变化）：

  优点：
  - 省 token：LLM 只需描述改了什么，不用复述整个文件
  - 可审查：用户在 suggest 模式下看到的是 diff，直观清晰
  - 精准：改动范围清楚，不会误改其他部分

  Patch 格式（V4A 简化版，Codex CLI 同款）：
  ┌─────────────────────────────────────────┐
  │ *** Update File: path/to/file.py        │
  │ @@                                      │
  │  context_line  （空格开头 = 不变的行）  │
  │ -old_line      （减号开头 = 删除）      │
  │ +new_line      （加号开头 = 新增）      │
  └─────────────────────────────────────────┘

【Approval 审批模式】
  suggest:   展示 diff，等用户 y/n 确认
  auto:      自动快照后执行，可 rollback
  full-auto: 直接执行，不问不拍照

运行：
  python code/min_codex.py --mode suggest    # 推荐初学者
  python code/min_codex.py --mode auto
  python code/min_codex.py --mode full-auto
"""

import os
import sys
import subprocess
import textwrap
import anthropic
from typing import Literal

client = anthropic.Anthropic()

# ── 审批模式 ──────────────────────────────────────────────────
ApprovalMode = Literal["suggest", "auto", "full-auto"]

# ── 快照栈（实现 rollback）────────────────────────────────────
SNAPSHOTS: list[dict] = []   # [{"path": str, "before": str | None}]


# ══════════════════════════════════════════════════════════════
# 核心工具：apply_patch
# ══════════════════════════════════════════════════════════════

def apply_patch(patch: str) -> str:
    """
    ★ Codex CLI 标志性工具：将 diff 应用到文件

    接受的 patch 格式（V4A 简化版）：
      *** Update File: path/to/file.py
      @@
       context_line    ← 空格开头，不修改，用于定位
      -removed_line    ← 减号开头，删除此行
      +added_line      ← 加号开头，新增此行

    支持多个文件、多个 hunk（@@ 分隔）。
    每个 hunk 用上下文行在文件中定位，然后执行删除/新增。
    """
    results = []
    # 按文件分割 patch
    file_sections = _split_by_file(patch)

    if not file_sections:
        return "[错误] patch 格式无效，需要包含 '*** Update File: <path>' 行"

    for path, hunks_text in file_sections:
        result = _apply_to_file(path, hunks_text)
        results.append(f"{path}: {result}")

    return "\n".join(results)


def _split_by_file(patch: str) -> list[tuple[str, str]]:
    """将 patch 按文件分割"""
    sections = []
    current_path = None
    current_lines = []

    for line in patch.splitlines():
        if line.startswith("*** Update File:"):
            if current_path is not None:
                sections.append((current_path, "\n".join(current_lines)))
            current_path  = line[len("*** Update File:"):].strip()
            current_lines = []
        elif current_path is not None:
            current_lines.append(line)

    if current_path is not None:
        sections.append((current_path, "\n".join(current_lines)))

    return sections


def _apply_to_file(path: str, hunks_text: str) -> str:
    """将 hunks 应用到单个文件"""
    # 读取原文件
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines(keepends=True)
    except FileNotFoundError:
        # 新建文件
        lines = []

    # 解析 hunks（@@ 分隔）
    hunks = _parse_hunks(hunks_text)
    if not hunks:
        return "[错误] 没有找到有效的 hunk（需要 @@ 行）"

    # 依次应用每个 hunk
    for hunk in hunks:
        ok, lines, msg = _apply_hunk(lines, hunk)
        if not ok:
            return f"[错误] hunk 应用失败：{msg}"

    # 写回文件
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        changed = sum(1 for h in hunks for l in h if l.startswith(("+", "-")))
        return f"成功（{len(hunks)} 个 hunk，{changed} 行变更）"
    except Exception as e:
        return f"[错误] 写入失败：{e}"


def _parse_hunks(text: str) -> list[list[str]]:
    """将 hunks_text 按 @@ 分割成 hunk 列表"""
    hunks = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "@@":
            if current:
                hunks.append(current)
            current = []
        else:
            if line:   # 跳过空行
                current.append(line)
    if current:
        hunks.append(current)
    return hunks


def _apply_hunk(lines: list[str], hunk: list[str]) -> tuple[bool, list[str], str]:
    """
    把一个 hunk 应用到 lines 上。
    用上下文行（空格开头）定位插入点，然后删除/新增。
    """
    # 分类 hunk 行
    ctx_lines = [l[1:] for l in hunk if l.startswith(" ")]  # 上下文（用于定位）
    del_lines = [l[1:] for l in hunk if l.startswith("-")]  # 要删除
    add_lines = [l[1:] for l in hunk if l.startswith("+")]  # 要新增

    # 用上下文 + 删除行组合定位
    anchor_text = "".join(
        (l[1:] if l.startswith((" ", "-")) else "") for l in hunk
        if l.startswith((" ", "-"))
    )

    file_text = "".join(lines)

    if anchor_text and anchor_text not in file_text:
        return False, lines, f"未找到 hunk 上下文：{anchor_text[:60]!r}"

    # 构建替换文本（删除行 → 新增行）
    del_text = "".join(del_lines)
    add_text = "".join(l if l.endswith("\n") else l + "\n" for l in add_lines)

    if del_text:
        del_text_with_nl = "".join(l if l.endswith("\n") else l + "\n" for l in del_lines)
        if del_text_with_nl in file_text:
            new_text = file_text.replace(del_text_with_nl, add_text, 1)
        elif del_text in file_text:
            new_text = file_text.replace(del_text, add_text, 1)
        else:
            return False, lines, f"未找到要删除的行：{del_text[:60]!r}"
    else:
        # 纯插入：在上下文后面插入
        if ctx_lines:
            anchor = "".join(l if l.endswith("\n") else l + "\n" for l in ctx_lines[-1:])
            new_text = file_text.replace(anchor, anchor + add_text, 1)
        else:
            new_text = file_text + add_text

    return True, [new_text], ""


def _render_patch_preview(patch: str) -> str:
    """将 patch 渲染为彩色预览（终端显示用）"""
    lines = []
    for line in patch.splitlines():
        if line.startswith("*** Update File:"):
            lines.append(f"\033[1;36m{line}\033[0m")      # 青色粗体
        elif line.startswith("@@"):
            lines.append(f"\033[33m{line}\033[0m")         # 黄色
        elif line.startswith("-"):
            lines.append(f"\033[31m{line}\033[0m")         # 红色
        elif line.startswith("+"):
            lines.append(f"\033[32m{line}\033[0m")         # 绿色
        else:
            lines.append(line)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 快照 / Rollback
# ══════════════════════════════════════════════════════════════

def take_snapshot(path: str) -> None:
    """记录文件当前内容，支持 rollback"""
    try:
        content = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        content = None
    SNAPSHOTS.append({"path": path, "before": content})
    print(f"  [快照 #{len(SNAPSHOTS)}] 已记录 {path}")


def rollback() -> str:
    """恢复到最近一次快照"""
    if not SNAPSHOTS:
        return "[Rollback] 没有可回滚的快照"
    snap = SNAPSHOTS.pop()
    path, before = snap["path"], snap["before"]
    if before is None:
        if os.path.exists(path):
            os.remove(path)
        return f"[Rollback] 已删除 {path}（原本不存在）"
    with open(path, "w", encoding="utf-8") as f:
        f.write(before)
    return f"[Rollback] 已恢复 {path} 到快照 #{len(SNAPSHOTS) + 1} 的状态"


# ══════════════════════════════════════════════════════════════
# 工具（仅 read_file + apply_patch，不再暴露 write_file）
# ══════════════════════════════════════════════════════════════

def read_file(path: str) -> str:
    """读取文件，带行号"""
    try:
        lines = open(path, encoding="utf-8").readlines()
        numbered = "".join(f"{i+1:3d} | {l}" for i, l in enumerate(lines))
        return f"[{path}] {len(lines)} 行\n{numbered}"
    except FileNotFoundError:
        return f"[错误] 文件不存在：{path}"
    except Exception as e:
        return f"[错误] {e}"


def bash_exec(command: str) -> str:
    """执行 shell 命令"""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        out = (r.stdout + r.stderr).strip() or "(无输出)"
        return f"[exit {r.returncode}]\n{out}"
    except subprocess.TimeoutExpired:
        return "[错误] 超时"
    except Exception as e:
        return f"[错误] {e}"


TOOLS = {
    "read_file":   read_file,
    "apply_patch": apply_patch,
    "bash_exec":   bash_exec,
}

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "读取文件内容（带行号），修改前必须先 read。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "apply_patch",
        "description": textwrap.dedent("""\
            ★ 核心工具：用 diff 格式修改文件（不需要输出整个文件内容）。

            patch 格式（每个文件一个块）：
              *** Update File: path/to/file.py
              @@
               context_line   （空格开头，用于定位，不修改）
              -old_line        （减号开头，删除此行）
              +new_line        （加号开头，新增此行）

            示例（修复一个 bug）：
              *** Update File: /tmp/calc.py
              @@
               def add(a, b):
              -    return a - b
              +    return a + b
        """),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "符合上述格式的 patch 文本",
                }
            },
            "required": ["patch"],
        },
    },
    {
        "name": "bash_exec",
        "description": "执行 shell 命令。【高风险，需要审批】",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

RISKY_TOOLS = {"apply_patch", "bash_exec"}


# ══════════════════════════════════════════════════════════════
# 审批逻辑
# ══════════════════════════════════════════════════════════════

def request_approval(tool_name: str, args: dict) -> bool:
    """
    展示将要执行的操作，等待用户确认。
    apply_patch 时展示渲染后的 diff，其他工具展示参数。
    """
    print(f"\n  ⚠️  [{tool_name}] 请求执行：")

    if tool_name == "apply_patch":
        patch = args.get("patch", "")
        print(_render_patch_preview(patch))
    else:
        for k, v in args.items():
            print(f"     {k}: {str(v)[:200]}")

    while True:
        choice = input("\n  确认执行？[y/n/rollback] > ").strip().lower()
        if choice == "y":
            return True
        elif choice == "n":
            return False
        elif choice == "rollback":
            print(f"  {rollback()}")
        else:
            print("  请输入 y / n / rollback")


def execute_with_approval(tool_name: str, args: dict, mode: ApprovalMode) -> str:
    """根据模式决定是否审批，并在必要时打快照"""
    is_risky = tool_name in RISKY_TOOLS

    if not is_risky:
        return TOOLS[tool_name](**args)

    if mode == "suggest":
        if not request_approval(tool_name, args):
            return f"[跳过] 用户拒绝执行 {tool_name}"
        # 对 apply_patch 自动快照（以便回滚）
        if tool_name == "apply_patch":
            _snapshot_patch_targets(args.get("patch", ""))

    elif mode == "auto":
        print(f"  [auto] 自动快照后执行 {tool_name}")
        if tool_name == "apply_patch":
            _snapshot_patch_targets(args.get("patch", ""))

    # full-auto: 直接执行
    return TOOLS[tool_name](**args)


def _snapshot_patch_targets(patch: str) -> None:
    """从 patch 中提取所有目标文件并打快照"""
    for line in patch.splitlines():
        if line.startswith("*** Update File:"):
            path = line[len("*** Update File:"):].strip()
            take_snapshot(path)


# ══════════════════════════════════════════════════════════════
# Agent Loop
# ══════════════════════════════════════════════════════════════

def run_agent(task: str, mode: ApprovalMode = "suggest", max_steps: int = 15) -> str:
    """
    Codex CLI Agent Loop（apply_patch + Approval）

    每轮：LLM 决策 → 判断风险 → [审批/快照] → 执行 → 更新上下文 → 循环
    """
    messages = [{"role": "user", "content": task}]

    print(f"\n🔑 审批模式：{mode}（输入 rollback 可随时回滚）\n")

    for step in range(max_steps):
        print(f"── Step {step + 1} {'─' * 44}")

        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            system=(
                "你是 Codex，一个编程助手。修改文件时必须使用 apply_patch 工具，\n"
                "输出标准 patch 格式（*** Update File + @@ hunk），不要直接输出整个文件。\n"
                "先用 read_file 读取文件，然后生成只包含变更部分的 patch。"
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

                print(f"🔧 {block.name}")
                result = execute_with_approval(block.name, dict(block.input), mode)
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
# 演示
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode: ApprovalMode = "suggest"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]  # type: ignore

    # 准备演示文件（有意写一个有 bug 的版本）
    demo_file = "/tmp/demo_server.py"
    with open(demo_file, "w") as f:
        f.write(textwrap.dedent("""\
            class Server:
                def __init__(self, host, port):
                    self.host = host
                    self.port = port
                    self.running = False

                def start(self):
                    self.running = True
                    print(f"Server started on {self.port}")   # bug: 缺少 host

                def stop(self):
                    self.running = False
                    print("Server stopped")

                def status(self):
                    state = "running" if self.running else "stopped"
                    return f"{state}"    # bug: 应该返回 host:port 信息
        """))

    print("=" * 60)
    print(f"Codex CLI 最小实现 v2 — apply_patch 演示（{mode} 模式）")
    print("=" * 60)
    print(f"目标文件：{demo_file}")
    print("任务：用 apply_patch（diff）修复两处 bug")
    print()

    run_agent(
        task=(
            f"请修复 {demo_file} 中的两处 bug：\n"
            "1. start() 方法的打印缺少 host 信息\n"
            "2. status() 返回值应该包含 host:port\n"
            "请用 apply_patch 以 diff 格式修改，不要重写整个文件。"
        ),
        mode=mode,
    )
