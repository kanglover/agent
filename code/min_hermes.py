"""
min_hermes.py — Hermes Agent 最小实现

═══════════════════════════════════════════════════════
核心特点：持久化记忆 + Skills 学习循环 + 多平台网关
═══════════════════════════════════════════════════════

【与 Claude Code / Codex CLI 的核心区别】

  Claude Code / Codex CLI：
    - 单次会话，关闭终端后所有上下文丢失
    - 每次从零开始，不会「记住」你的偏好和项目
    - 只支持终端交互

  Hermes Agent：
    - 跨会话持久记忆（SQLite + MEMORY.md）
    - 自动创建/改进 Skill，越用越聪明
    - 支持 CLI + 20+ 消息平台（Telegram、Discord、Slack 等）
    - 可部署在服务器上，不依赖本地电脑

【本最小实现演示的关键特性】

  1. 持久化会话：SQLite 存储对话历史，重启后自动恢复
  2. Skills 系统：Agent 自动把成功的工作流保存为可复用 Skill
  3. 多源 Prompt 组装：SOUL.md（个性）+ MEMORY.md（记忆）+ 上下文文件
  4. 工具注册表：中央注册，支持权限控制
  5. 上下文压缩：超出阈值时自动摘要压缩

运行：python code/min_hermes.py
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

HERMES_HOME = os.path.expanduser("~/.min_hermes")
STATE_DB = os.path.join(HERMES_HOME, "state.db")
SKILLS_DIR = os.path.join(HERMES_HOME, "skills")
MEMORY_FILE = os.path.join(HERMES_HOME, "MEMORY.md")
SOUL_FILE = os.path.join(HERMES_HOME, "SOUL.md")

# 模拟 LLM 调用（实际使用时会调用真实 API）
class MockLLM:
    """模拟 LLM，用于演示核心逻辑而不依赖真实 API。"""

    def __init__(self):
        self.turn_count = 0

    def call(self, system_prompt: str, messages: list, tools: list) -> dict:
        """
        模拟 LLM 调用。
        根据对话内容决定是 end_turn 还是 tool_use。
        """
        self.turn_count += 1
        raw_content = messages[-1]["content"] if messages else ""
        # content 可能是 str 或 list（tool_results），统一转为字符串
        if isinstance(raw_content, list):
            last_message = " ".join(str(item) for item in raw_content)
        else:
            last_message = str(raw_content)

        # 模拟：如果用户要求写文件，就调用 write_file 工具
        if "写" in last_message or "write" in last_message.lower() or "创建" in last_message:
            if "fibonacci" in last_message.lower() or "fibonacci" in system_prompt.lower():
                return {
                    "stop_reason": "tool_use",
                    "tool_calls": [{
                        "id": f"tool_{self.turn_count}",
                        "name": "write_file",
                        "input": {
                            "path": "/tmp/fibonacci.py",
                            "content": "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nif __name__ == '__main__':\n    print(f'fibonacci(10) = {fibonacci(10)}')\n"
                        }
                    }],
                    "text": "我来创建 fibonacci.py 文件。"
                }

        # 模拟：如果用户要求运行代码，就调用 bash_exec 工具
        if "运行" in last_message or "测试" in last_message or "run" in last_message.lower():
            return {
                "stop_reason": "tool_use",
                "tool_calls": [{
                    "id": f"tool_{self.turn_count}",
                    "name": "bash_exec",
                    "input": {"command": "python /tmp/fibonacci.py"}
                }],
                "text": "我来运行代码验证结果。"
            }

        # 模拟：如果用户要求读取文件，就调用 read_file 工具
        if "读" in last_message or "read" in last_message.lower() or "查看" in last_message:
            return {
                "stop_reason": "tool_use",
                "tool_calls": [{
                    "id": f"tool_{self.turn_count}",
                    "name": "read_file",
                    "input": {"path": "/tmp/fibonacci.py"}
                }],
                "text": "我来读取文件内容。"
            }

        # 模拟：如果完成了复杂任务，调用 skill_manage 保存 Skill
        if self.turn_count >= 3 and "skill" not in last_message.lower():
            # 检查是否应该创建 skill（模拟：每 3+ 轮后询问）
            return {
                "stop_reason": "tool_use",
                "tool_calls": [{
                    "id": f"tool_{self.turn_count}",
                    "name": "skill_manage",
                    "input": {
                        "operation": "create",
                        "name": "fibonacci-workflow",
                        "content": "---\nname: fibonacci-workflow\ndescription: 创建并运行 Fibonacci 代码的完整工作流\nversion: 1.0.0\n---\n\n## When to Use\n用户要求写 Fibonacci 相关代码时。\n\n## Procedure\n1. 用 write_file 创建 /tmp/fibonacci.py\n2. 用 read_file 检查代码内容\n3. 用 bash_exec 运行验证\n\n## Pitfalls\n- 确保使用 lru_cache 优化递归性能\n- 验证 fibonacci(10) 输出为 55"
                    }
                }],
                "text": "这个工作流看起来很有用，我把它保存为 Skill 以便将来复用。"
            }

        # 默认：直接回复
        return {
            "stop_reason": "end_turn",
            "text": f"任务已完成（共 {self.turn_count} 轮交互）。"
        }


llm = MockLLM()


# ══════════════════════════════════════════════════════════════
# 工具实现
# ══════════════════════════════════════════════════════════════

def read_file(path: str) -> str:
    """读取文件内容（带行号）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        numbered = "".join(f"{i+1:3d} | {l}" for i, l in enumerate(lines))
        return f"[{path}] {len(lines)} 行\n{numbered}"
    except FileNotFoundError:
        return f"[错误] 文件不存在：{path}"
    except Exception as e:
        return f"[错误] {e}"


def write_file(path: str, content: str) -> str:
    """将内容写入文件（会覆盖）。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[成功] 已写入 {path}（{content.count(chr(10))+1} 行）"
    except Exception as e:
        return f"[错误] 写入失败：{e}"


def bash_exec(command: str) -> str:
    """执行 shell 命令，返回输出。"""
    import subprocess
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        out = (r.stdout + r.stderr).strip() or "(无输出)"
        return f"[exit {r.returncode}]\n{out}"
    except subprocess.TimeoutExpired:
        return "[错误] 超时"
    except Exception as e:
        return f"[错误] {e}"


def skill_manage(operation: str, name: str = None, content: str = None,
                 old_string: str = None, new_string: str = None) -> str:
    """
    管理 Skill：创建、更新、删除可复用的工作流知识。
    这是 Hermes 的核心学习机制——Agent 把经验保存为 Skill。
    """
    skill_path = os.path.join(SKILLS_DIR, f"{name}.md")

    if operation == "create":
        if os.path.exists(skill_path):
            return f"[跳过] Skill '{name}' 已存在"
        os.makedirs(SKILLS_DIR, exist_ok=True)
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"[Skill 创建] {name} → {skill_path}"

    elif operation == "patch":
        if not os.path.exists(skill_path):
            return f"[错误] Skill '{name}' 不存在"
        with open(skill_path, "r", encoding="utf-8") as f:
            old_content = f.read()
        if old_string not in old_content:
            return f"[错误] 找不到要替换的文本"
        new_content = old_content.replace(old_string, new_string, 1)
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"[Skill 更新] {name}（patch）"

    elif operation == "edit":
        os.makedirs(SKILLS_DIR, exist_ok=True)
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"[Skill 重写] {name}"

    elif operation == "delete":
        if os.path.exists(skill_path):
            os.remove(skill_path)
            return f"[Skill 删除] {name}"
        return f"[错误] Skill '{name}' 不存在"

    elif operation == "list":
        if not os.path.exists(SKILLS_DIR):
            return "[]"
        skills = [f.replace(".md", "") for f in os.listdir(SKILLS_DIR) if f.endswith(".md")]
        return json.dumps(skills, ensure_ascii=False)

    return f"[错误] 未知操作：{operation}"


# ── 工具注册表 ──────────────────────────────────────────────
TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "bash_exec": bash_exec,
    "skill_manage": skill_manage,
}

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "读取文件内容（带行号）。适用于查看源代码、检查日志等。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件绝对路径"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "将内容写入文件（会覆盖）。适用于创建新文件或修改现有文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件绝对路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash_exec",
        "description": "执行 shell 命令，返回输出。适用于运行代码、安装依赖等。",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的 shell 命令"}},
            "required": ["command"],
        },
    },
    {
        "name": "skill_manage",
        "description": """管理可复用的 Skill（程序性记忆）。

适用场景：
- 成功完成复杂任务后，把该工作流保存为 Skill
- 发现更好的方法时，patch 更新现有 Skill
- 列出所有已保存的 Skill

操作类型：
- create: 创建新 Skill（name + content）
- patch: 针对性修改（name + old_string + new_string）
- edit: 完全重写（name + content）
- delete: 删除 Skill（name）
- list: 列出所有 Skill（无需 name）
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "patch", "edit", "delete", "list"],
                    "description": "操作类型"
                },
                "name": {"type": "string", "description": "Skill 名称"},
                "content": {"type": "string", "description": "完整的 SKILL.md 内容（create/edit 时必填）"},
                "old_string": {"type": "string", "description": "要替换的旧文本（patch 时必填）"},
                "new_string": {"type": "string", "description": "替换后的新文本（patch 时必填）"},
            },
            "required": ["operation"],
        },
    },
]


# ══════════════════════════════════════════════════════════════
# 会话持久化（SQLite）
# ══════════════════════════════════════════════════════════════

class SessionStorage:
    """
    Hermes 的会话存储：SQLite + FTS5 全文检索。

    与 Claude Code / Codex CLI 的区别：
    - 它们：会话结束即丢失，每次从零开始
    - Hermes：跨重启自动恢复，支持全文搜索历史对话
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 会话表：存储每轮对话
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES sessions(id)
            )
        """)

        # 创建全文搜索索引（FTS5）
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                content, session_key,
                content='sessions', content_rowid='id'
            )
        """)

        conn.commit()
        conn.close()

    def save_message(self, session_key: str, role: str, content: str = None,
                     tool_calls: list = None, parent_id: int = None) -> int:
        """保存一条消息到数据库。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None

        cursor.execute("""
            INSERT INTO sessions (session_key, role, content, tool_calls, parent_id)
            VALUES (?, ?, ?, ?, ?)
        """, (session_key, role, content, tool_calls_json, parent_id))

        msg_id = cursor.lastrowid

        # 同步到 FTS5 索引
        if content:
            cursor.execute("""
                INSERT INTO sessions_fts (rowid, content, session_key)
                VALUES (?, ?, ?)
            """, (msg_id, content, session_key))

        conn.commit()
        conn.close()
        return msg_id

    def load_session(self, session_key: str) -> list[dict]:
        """加载指定会话的历史消息。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT role, content, tool_calls FROM sessions
            WHERE session_key = ?
            ORDER BY id ASC
        """, (session_key,))

        messages = []
        for role, content, tool_calls in cursor.fetchall():
            msg = {"role": role}
            if content:
                msg["content"] = content
            if tool_calls:
                msg["tool_calls"] = json.loads(tool_calls)
            messages.append(msg)

        conn.close()
        return messages

    def search_history(self, query: str, session_key: str = None) -> list[dict]:
        """全文搜索历史对话。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_key:
            cursor.execute("""
                SELECT s.id, s.role, s.content, s.timestamp
                FROM sessions_fts f
                JOIN sessions s ON f.rowid = s.id
                WHERE sessions_fts MATCH ? AND s.session_key = ?
                ORDER BY rank
            """, (query, session_key))
        else:
            cursor.execute("""
                SELECT s.id, s.role, s.content, s.timestamp
                FROM sessions_fts f
                JOIN sessions s ON f.rowid = s.id
                WHERE sessions_fts MATCH ?
                ORDER BY rank
            """, (query,))

        results = []
        for rowid, role, content, timestamp in cursor.fetchall():
            results.append({
                "id": rowid,
                "role": role,
                "content": content[:200] + "..." if content and len(content) > 200 else content,
                "timestamp": timestamp,
            })

        conn.close()
        return results


# ══════════════════════════════════════════════════════════════
# Prompt 构建器
# ══════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    从多源组装系统 prompt：SOUL.md（个性）+ MEMORY.md（记忆）+ Skills + 上下文文件

    这是 Hermes 的核心设计：prompt 不是静态的，而是根据用户和项目动态组装。
    """

    def __init__(self, hermes_home: str):
        self.hermes_home = hermes_home
        self.skills_dir = os.path.join(hermes_home, "skills")
        self.memory_file = os.path.join(hermes_home, "MEMORY.md")
        self.soul_file = os.path.join(hermes_home, "SOUL.md")

    def build(self, load_skills: bool = True) -> str:
        """组装完整的系统 prompt。"""
        parts = []

        # 1. SOUL.md —— 个性定义（你是谁，如何说话）
        soul = self._load_file(self.soul_file, default="你是一个 helpful 的 AI 助手。")
        parts.append(f"## 个性（SOUL.md）\n{soul}")

        # 2. MEMORY.md —— 持久记忆（你记住了什么）
        memory = self._load_file(self.memory_file, default="（暂无持久记忆）")
        parts.append(f"## 记忆（MEMORY.md）\n{memory}")

        # 3. Skills —— 可复用的工作流知识
        if load_skills:
            skills = self._load_skills()
            if skills:
                parts.append(f"## 可用 Skills\n{skills}")

        # 4. 工具使用指引
        parts.append("""
## 工具使用原则
- 操作文件前，先用 read_file 确认内容
- 修改文件优先使用 write_file（精准写入）
- 使用 bash_exec 时，优先使用无副作用的只读命令
- 完成复杂任务后，考虑用 skill_manage 保存为 Skill
""")

        return "\n\n".join(parts)

    def _load_file(self, path: str, default: str = "") -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return default

    def _load_skills(self) -> str:
        """加载所有 Skill 的摘要（Level 0：只加载名称和描述）。"""
        if not os.path.exists(self.skills_dir):
            return ""

        skills = []
        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".md"):
                continue
            skill_path = os.path.join(self.skills_dir, filename)
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 提取 frontmatter 中的 name 和 description
                name = filename.replace(".md", "")
                desc = ""
                if content.startswith("---"):
                    fm_end = content.find("---", 3)
                    if fm_end > 0:
                        fm = content[3:fm_end]
                        for line in fm.split("\n"):
                            if line.startswith("description:"):
                                desc = line.split(":", 1)[1].strip()
                skills.append(f"- {name}: {desc}")
            except Exception:
                pass

        return "\n".join(skills) if skills else "（暂无 Skills）"


# ══════════════════════════════════════════════════════════════
# 上下文压缩
# ══════════════════════════════════════════════════════════════

class ContextCompressor:
    """
    当对话历史过长时，自动压缩早期消息。

    Hermes 的策略：
    - < 50% 窗口容量 → 完整保留
    - 50%-80% 容量 → 压缩早期工具调用结果
    - > 80% 容量 → 触发摘要压缩机制
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages

    def compress(self, messages: list[dict]) -> list[dict]:
        """压缩消息历史，保留最近的重要消息。"""
        if len(messages) <= self.max_messages:
            return messages

        # 策略：保留前 2 条（系统/初始上下文）+ 最近 N 条
        # 中间的消息压缩为摘要
        preserved_head = messages[:2]
        preserved_tail = messages[-(self.max_messages - 4):]

        # 中间部分生成摘要
        middle = messages[2:-len(preserved_tail)]
        summary = f"[摘要] 中间 {len(middle)} 轮对话已压缩。关键信息："

        # 提取工具调用摘要
        tool_calls_count = 0
        for msg in middle:
            if msg.get("tool_calls"):
                tool_calls_count += len(msg["tool_calls"])

        if tool_calls_count > 0:
            summary += f" 共 {tool_calls_count} 次工具调用。"

        summary_msg = {"role": "user", "content": summary}

        return preserved_head + [summary_msg] + preserved_tail


# ══════════════════════════════════════════════════════════════
# Hermes Agent 核心循环
# ══════════════════════════════════════════════════════════════

class HermesAgent:
    """
    Hermes Agent 核心：持久化 + 学习循环 + 多源 Prompt。

    与 Claude Code / Codex CLI 的关键差异：
    1. 加载历史：从 SQLite 恢复之前的对话，不是空会话
    2. 组装 Prompt：SOUL.md + MEMORY.md + Skills + 工具指引
    3. 学习：完成复杂任务后自动保存 Skill
    4. 持久化：每轮对话自动写入 SQLite
    """

    def __init__(self, session_key: str = "default"):
        self.session_key = session_key
        self.storage = SessionStorage(STATE_DB)
        self.prompt_builder = PromptBuilder(HERMES_HOME)
        self.compressor = ContextCompressor(max_messages=15)
        self.tool_registry = TOOL_REGISTRY
        self.tool_definitions = TOOL_DEFINITIONS

    def _init_files(self):
        """初始化默认的 SOUL.md 和 MEMORY.md。"""
        os.makedirs(HERMES_HOME, exist_ok=True)

        if not os.path.exists(SOUL_FILE):
            with open(SOUL_FILE, "w", encoding="utf-8") as f:
                f.write("你是一个 helpful、谨慎的 AI 助手。\n"
                        "你擅长编程、文件操作和任务自动化。\n"
                        "你会在完成任务后主动总结，并考虑是否需要保存为 Skill。")

        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write("# 记忆\n\n"
                        "## 用户偏好\n- 喜欢 Python\n- 偏好简洁的代码风格\n\n"
                        "## 项目上下文\n- 工作目录：当前目录\n")

    def run_conversation(self, user_input: str, max_turns: int = 10) -> str:
        """
        运行一次完整的对话循环。

        流程：
        1. 从 SQLite 加载历史消息
        2. 组装系统 prompt（SOUL + MEMORY + Skills）
        3. 调用 LLM
        4. 如果 tool_use → 执行工具 → 把结果喂回
        5. 如果 end_turn → 保存会话并返回
        6. 检查是否需要创建 Skill（学习循环）
        """
        self._init_files()

        # 1. 加载历史
        messages = self.storage.load_session(self.session_key)
        print(f"📝 从 SQLite 加载了 {len(messages)} 条历史消息")

        # 添加用户输入
        messages.append({"role": "user", "content": user_input})

        # 2. 组装系统 prompt
        system_prompt = self.prompt_builder.build()
        print(f"🧠 系统 prompt 已组装（SOUL + MEMORY + Skills）")

        for turn in range(max_turns):
            print(f"\n── Turn {turn + 1} {'─' * 45}")

            # 上下文压缩
            messages = self.compressor.compress(messages)

            # 3. 调用 LLM
            response = llm.call(system_prompt, messages, self.tool_definitions)

            # 保存 assistant 消息
            assistant_msg = {"role": "assistant"}
            if response.get("text"):
                assistant_msg["content"] = response["text"]
            if response.get("tool_calls"):
                assistant_msg["tool_calls"] = response["tool_calls"]
            self.storage.save_message(self.session_key, "assistant",
                                      response.get("text"),
                                      response.get("tool_calls"))

            # 4. 检查 stop_reason
            if response["stop_reason"] == "end_turn":
                print(f"✅ end_turn：{response.get('text', '完成')}")

                # 5. 学习：检查是否创建 Skill
                self._maybe_learn(messages)

                return response.get("text", "完成")

            elif response["stop_reason"] == "tool_use":
                tool_results = []
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    tool_input = tc["input"]

                    # 执行工具
                    if tool_name in self.tool_registry:
                        result = self.tool_registry[tool_name](**tool_input)
                    else:
                        result = f"[错误] 未知工具：{tool_name}"

                    print(f"   🔧 {tool_name} → {result.splitlines()[0]}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    })

                # 把工具结果加入消息历史
                messages.append(assistant_msg)
                messages.append({"role": "user", "content": tool_results})

                # 保存到数据库
                self.storage.save_message(self.session_key, "user",
                                          json.dumps(tool_results, ensure_ascii=False))

        return "[超时] 超过最大轮数"

    def _maybe_learn(self, messages: list):
        """
        学习循环：如果完成了复杂任务，考虑保存为 Skill。

        实际 Hermes 中，这是由 LLM 自己决定的（通过 skill_manage 工具）。
        这里简化为：如果工具调用次数 >= 3，就提示可以保存 Skill。
        """
        tool_call_count = sum(1 for m in messages if m.get("tool_calls"))
        if tool_call_count >= 3:
            print(f"\n🧠 [学习] 本次对话使用了 {tool_call_count} 次工具调用，"
                  f"已自动保存/更新 Skill")

    def search_memory(self, query: str) -> list[dict]:
        """搜索历史对话记忆。"""
        return self.storage.search_history(query, self.session_key)


# ══════════════════════════════════════════════════════════════
# 演示
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Hermes Agent 最小实现")
    print("核心特性：持久化 + 学习循环 + 多源 Prompt")
    print("=" * 60)
    print()

    # 初始化 Hermes 目录
    os.makedirs(HERMES_HOME, exist_ok=True)
    os.makedirs(SKILLS_DIR, exist_ok=True)

    print(f"📁 Hermes Home: {HERMES_HOME}")
    print(f"🗄️  数据库: {STATE_DB}")
    print(f"📚 Skills 目录: {SKILLS_DIR}")
    print(f"🧠 MEMORY: {MEMORY_FILE}")
    print(f"🎭 SOUL: {SOUL_FILE}")
    print()

    # 创建 Agent 实例
    agent = HermesAgent(session_key="demo_session")

    # 第一次对话：写 Fibonacci 代码
    print("🎯 用户：请写一个 Fibonacci 函数，保存到 /tmp/fibonacci.py")
    print()
    result1 = agent.run_conversation(
        "请写一个 fibonacci(n) 函数，保存到 /tmp/fibonacci.py。"
        "要求：用递归 + lru_cache，包含 main 打印 fibonacci(10)。"
    )
    print(f"\n🤖 Agent：{result1}")
    print()

    # 第二次对话：运行代码（验证持久化：历史已保存）
    print("🎯 用户：运行刚才写的代码，验证输出")
    print()
    result2 = agent.run_conversation("运行 /tmp/fibonacci.py，验证 fibonacci(10) 是否等于 55")
    print(f"\n🤖 Agent：{result2}")
    print()

    # 第三次对话：Agent 应该自动保存 Skill
    print("🎯 用户：再检查一下代码内容")
    print()
    result3 = agent.run_conversation("读取 /tmp/fibonacci.py 的内容")
    print(f"\n🤖 Agent：{result3}")
    print()

    # 展示持久化效果
    print("=" * 60)
    print("持久化验证：重新创建 Agent，加载同一会话")
    print("=" * 60)
    print()

    agent2 = HermesAgent(session_key="demo_session")
    history = agent2.storage.load_session("demo_session")
    print(f"📝 从 SQLite 恢复的历史消息数：{len(history)}")
    for i, msg in enumerate(history[:6]):
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content[:80]
        print(f"   [{i}] {msg['role']}: {content}...")
    print()

    # 展示 Skill 学习成果
    print("=" * 60)
    print("Skill 学习成果")
    print("=" * 60)
    print()

    skills = skill_manage("list")
    print(f"📚 已保存的 Skills：{skills}")
    print()

    if os.path.exists(os.path.join(SKILLS_DIR, "fibonacci-workflow.md")):
        with open(os.path.join(SKILLS_DIR, "fibonacci-workflow.md"), "r") as f:
            print("📝 fibonacci-workflow Skill 内容：")
            print(f.read())
    print()

    # 展示全文搜索
    print("=" * 60)
    print("记忆搜索：全文检索历史对话")
    print("=" * 60)
    print()

    results = agent.search_memory("fibonacci")
    print(f"🔍 搜索 'fibonacci' 找到 {len(results)} 条结果：")
    for r in results[:3]:
        print(f"   [{r['role']}] {r['content'][:100]}...")
    print()

    print("=" * 60)
    print("演示完成！")
    print()
    print("关键学习点：")
    print("  1. 观察 ~/.min_hermes/state.db —— 理解 SQLite 持久化")
    print("  2. 观察 ~/.min_hermes/skills/ —— 理解 Skill 自动创建")
    print("  3. 观察 ~/.min_hermes/MEMORY.md 和 SOUL.md —— 理解多源 Prompt")
    print("  4. 重新运行脚本 —— 验证跨会话记忆恢复")
    print("=" * 60)
