# 运行: python 03_react_agent.py
# 依赖: pip install anthropic
# 说明: 实现一个 ReAct（Reasoning + Acting）Agent
#
# ReAct 是什么？
# 类比：就像一个特别能干的助手，遇到复杂任务时，它会：
#   1. 先"想一想"（Thought）：分析当前情况，决定下一步
#   2. 再"做一做"（Action）：调用某个工具
#   3. 看看结果（Observation）：观察工具返回什么
#   4. 重复以上步骤，直到任务完成
# 这就是 ReAct = Reasoning（推理）+ Acting（行动）的循环

import anthropic    # Claude SDK
import json         # JSON 处理
import os           # 文件操作
import math         # 数学函数（给 calculate 工具用）

# ============================================================
# 第一部分：工具实现（Agent 的"手脚"）
# ============================================================

def search_web(query: str) -> str:
    """
    模拟网络搜索
    真实版本：调用 Bing/Google Search API、Tavily、Serper 等
    现在用 Mock 数据演示流程

    参数：query = 搜索关键词
    返回：搜索结果字符串
    """
    print(f"    🔍 [search_web] 搜索: {query!r}")

    # Mock 搜索结果数据库
    mock_results = {
        "python asyncio": (
            "Python asyncio 是标准库中的异步 I/O 框架。"
            "核心概念：事件循环（event loop）、协程（coroutine）、任务（Task）。"
            "使用 async def 定义协程，await 等待异步操作完成。"
        ),
        "claude api": (
            "Claude API 由 Anthropic 提供，支持文本生成、工具调用、视觉理解。"
            "主要端点：/v1/messages。最新模型：claude-opus-4-5, claude-sonnet-4-5。"
            "定价按 token 计费，input/output 分开计算。"
        ),
        "react agent": (
            "ReAct（Reason+Act）是 2022 年 Google 提出的 Agent 框架。"
            "论文：'ReAct: Synergizing Reasoning and Acting in Language Models'。"
            "核心思想：让 LLM 交替进行推理（Thought）和行动（Action）。"
        ),
        "大语言模型": (
            "大语言模型（LLM）是基于 Transformer 架构的神经网络模型。"
            "代表作：GPT-4、Claude、Gemini、Llama。"
            "主要能力：文本理解、生成、推理、代码编写。"
        ),
    }

    # 查找最匹配的结果（简单关键词匹配）
    query_lower = query.lower()
    for key, result in mock_results.items():
        if any(word in query_lower for word in key.split()):
            result_text = f"搜索 '{query}' 的结果：\n{result}"
            print(f"    ✓ 找到相关结果（{len(result_text)} 字符）")
            return result_text

    # 没有匹配结果
    fallback = f"搜索 '{query}' 未找到明确结果。建议换个关键词重试。"
    print(f"    ✗ 未找到结果，返回默认提示")
    return fallback


def read_file(path: str) -> str:
    """
    读取本地文件内容
    支持读取文本文件（.txt, .md, .py 等）

    参数：path = 文件路径（相对或绝对路径）
    返回：文件内容字符串，或错误信息
    """
    print(f"    📂 [read_file] 读取: {path!r}")

    try:
        # 展开用户目录符号（~ 代表 home 目录）
        expanded_path = os.path.expanduser(path)
        with open(expanded_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"    ✓ 读取成功（{len(content)} 字符）")
        # 如果文件太长，截断到前 2000 字符，避免 token 爆炸
        if len(content) > 2000:
            content = content[:2000] + "\n\n[文件过长，已截断到前 2000 字符]"
        return content

    except FileNotFoundError:
        error = f"错误：文件不存在 '{path}'"
        print(f"    ✗ {error}")
        return error

    except PermissionError:
        error = f"错误：没有权限读取 '{path}'"
        print(f"    ✗ {error}")
        return error

    except Exception as e:
        error = f"错误：读取文件失败 - {str(e)}"
        print(f"    ✗ {error}")
        return error


def write_file(path: str, content: str) -> str:
    """
    写入文件（如果文件不存在会自动创建，如果存在会覆盖）

    参数：
        path    = 文件路径
        content = 要写入的内容
    返回：成功或失败信息
    """
    print(f"    💾 [write_file] 写入: {path!r}（{len(content)} 字符）")

    try:
        expanded_path = os.path.expanduser(path)
        # 如果父目录不存在，自动创建
        parent_dir = os.path.dirname(expanded_path)
        if parent_dir:  # 有父目录时才创建
            os.makedirs(parent_dir, exist_ok=True)

        with open(expanded_path, "w", encoding="utf-8") as f:
            f.write(content)

        success = f"成功：文件已写入 '{path}'（{len(content)} 字符）"
        print(f"    ✓ {success}")
        return success

    except PermissionError:
        error = f"错误：没有权限写入 '{path}'"
        print(f"    ✗ {error}")
        return error

    except Exception as e:
        error = f"错误：写入文件失败 - {str(e)}"
        print(f"    ✗ {error}")
        return error


def calculate(expr: str) -> str:
    """
    执行数学计算，支持基础运算和 math 函数

    参数：expr = 数学表达式字符串，例如 "2 ** 10" 或 "math.sqrt(16)"
    返回：计算结果字符串
    """
    print(f"    🔢 [calculate] 计算: {expr!r}")

    try:
        # 安全的计算环境：只允许 math 模块和基本函数
        # 不允许 import、exec、open 等危险操作
        safe_globals = {
            "math": math,           # 允许 math.sqrt(), math.pi 等
            "abs": abs,             # 绝对值
            "round": round,         # 四舍五入
            "min": min,             # 最小值
            "max": max,             # 最大值
            "sum": sum,             # 求和
            "__builtins__": {}      # 禁用所有内置函数（安全措施）
        }
        result = eval(expr, safe_globals)
        result_str = f"{expr} = {result}"
        print(f"    ✓ 结果: {result_str}")
        return result_str

    except ZeroDivisionError:
        error = "错误：除数不能为零"
        print(f"    ✗ {error}")
        return error

    except SyntaxError:
        error = f"错误：表达式语法有误 '{expr}'"
        print(f"    ✗ {error}")
        return error

    except Exception as e:
        error = f"错误：计算失败 - {str(e)}"
        print(f"    ✗ {error}")
        return error


# ============================================================
# 第二部分：工具的 JSON Schema 定义（告诉 Claude 有哪些工具可用）
# ============================================================

REACT_TOOLS = [
    {
        "name": "search_web",
        "description": "在网络上搜索信息。当需要查找外部知识、最新信息或不确定的事实时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，越具体越好，例如：'Python asyncio 教程' 而不是 'Python'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "读取本地文件的内容。当需要查看已有文件、读取数据或检查文档时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，例如：'/tmp/notes.txt' 或 '~/Desktop/report.md'"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "将内容写入本地文件。当需要保存结果、创建报告或记录信息时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要写入的文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入文件的内容"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "calculate",
        "description": "执行数学计算。当需要精确计算、处理数字或执行数学公式时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "数学表达式，例如：'2 ** 10'、'math.sqrt(144)'、'(100 - 30) * 0.8'"
                }
            },
            "required": ["expr"]
        }
    }
]

# 工具分发表（工具名 → 函数）
TOOL_MAP = {
    "search_web": search_web,
    "read_file": read_file,
    "write_file": write_file,
    "calculate": calculate,
}


# ============================================================
# 第三部分：ReActAgent 类
# ============================================================

class ReActAgent:
    """
    ReAct Agent 实现

    工作流程：
    ┌─────────────────────────────────────────────────┐
    │  用户任务                                        │
    │      ↓                                           │
    │  Claude 分析 → Thought（我需要先搜索一下...）    │
    │      ↓                                           │
    │  Action（调用 search_web）                       │
    │      ↓                                           │
    │  Observation（搜索结果是...）                    │
    │      ↓                                           │
    │  Claude 再次分析 → Thought（有了信息，现在我...）│
    │      ↓                                           │
    │  Action（调用 write_file 写入结果）              │
    │      ↓                                           │
    │  Observation（写入成功）                         │
    │      ↓                                           │
    │  Claude 判断任务完成 → 输出最终答案              │
    └─────────────────────────────────────────────────┘
    """

    MAX_STEPS = 10  # 最大执行步数，防止无限循环（保险措施）

    def __init__(self, model: str = "claude-opus-4-5"):
        """
        初始化 Agent

        参数：model = 使用的 Claude 模型名称
        """
        self.client = anthropic.Anthropic()  # 创建 Claude 客户端
        self.model = model
        self.trace = []      # 执行轨迹记录（每步的思考、行动、结果）
        self.messages = []   # 对话历史

    def _record_trace(self, step: int, event_type: str, content: str):
        """
        记录执行轨迹
        类比：就像飞机的黑匣子，记录每一步发生了什么

        参数：
            step       = 当前步骤编号
            event_type = 事件类型（"thought"、"action"、"observation"、"final"）
            content    = 事件内容
        """
        entry = {
            "step": step,
            "type": event_type,
            "content": content
        }
        self.trace.append(entry)

        # 用不同符号打印，直观看出是哪种事件
        icons = {
            "thought": "💭",
            "action": "⚡",
            "observation": "👁️",
            "final": "✅",
            "error": "❌"
        }
        icon = icons.get(event_type, "•")
        print(f"\n  [{step}] {icon} {event_type.upper()}")
        # 内容太长时截断显示
        display = content[:200] + "..." if len(content) > 200 else content
        print(f"      {display}")

    def run(self, task: str) -> dict:
        """
        执行任务的主入口方法

        参数：task = 用户描述的任务字符串
        返回：包含最终结果和执行轨迹的字典

        返回格式：
        {
            "success": True/False,
            "result": "最终答案文字",
            "steps_taken": 3,
            "trace": [每步记录的列表]
        }
        """
        print(f"\n{'='*60}")
        print(f"🎯 任务：{task}")
        print(f"{'='*60}")

        # 清空之前的状态（支持复用同一个 Agent 对象）
        self.trace = []
        self.messages = []

        # 构建系统提示，告诉 Claude 它是一个 ReAct Agent
        system_prompt = (
            "你是一个 ReAct Agent，通过'思考→行动→观察'循环来完成复杂任务。\n"
            "每次回复前，先用简短的思考分析当前情况和下一步计划。\n"
            "灵活使用提供的工具来收集信息、计算、读写文件。\n"
            "当任务完成时，给出清晰的最终总结。\n"
            "始终用中文思考和回复。"
        )

        # 用户任务作为第一条消息
        self.messages.append({"role": "user", "content": task})

        step = 0  # 步骤计数器

        # ---- 主循环：Thought → Action → Observation ----
        while step < self.MAX_STEPS:
            step += 1
            print(f"\n--- Step {step}/{self.MAX_STEPS} ---")

            # 调用 Claude
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=system_prompt,
                    tools=REACT_TOOLS,
                    messages=self.messages
                )
            except Exception as e:
                # API 调用失败（网络问题、API key 问题等）
                error_msg = f"API 调用失败：{str(e)}"
                self._record_trace(step, "error", error_msg)
                return {
                    "success": False,
                    "result": error_msg,
                    "steps_taken": step,
                    "trace": self.trace
                }

            # 把 Claude 的回复加入历史
            self.messages.append({"role": "assistant", "content": response.content})

            # ---- 解析 Claude 的回复 ----
            has_tool_call = False  # 标记这轮是否有工具调用

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    # Claude 的思考文字（Thought 阶段）
                    self._record_trace(step, "thought", block.text.strip())

                elif block.type == "tool_use":
                    # Claude 决定调用工具（Action 阶段）
                    has_tool_call = True
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    action_desc = f"调用 {tool_name}({json.dumps(tool_input, ensure_ascii=False)})"
                    self._record_trace(step, "action", action_desc)

                    # 执行工具
                    if tool_name in TOOL_MAP:
                        try:
                            result = TOOL_MAP[tool_name](**tool_input)
                        except Exception as e:
                            result = f"工具执行异常：{str(e)}"
                    else:
                        result = f"工具 '{tool_name}' 未注册"

                    # 记录观察结果（Observation 阶段）
                    self._record_trace(step, "observation", str(result))

                    # 把工具结果回传给 Claude
                    # 必须用 "user" 角色，content 是 tool_result 类型
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": str(result)
                            }
                        ]
                    })

            # ---- 判断是否结束 ----
            if response.stop_reason == "end_turn" and not has_tool_call:
                # Claude 说完话，没有调用工具，说明任务完成了
                # 提取最终的文字回复
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                self._record_trace(step, "final", final_text)

                print(f"\n{'='*60}")
                print(f"任务完成（共 {step} 步）")
                print(f"{'='*60}")

                return {
                    "success": True,
                    "result": final_text,
                    "steps_taken": step,
                    "trace": self.trace
                }

        # ---- 达到最大步数，强制结束 ----
        timeout_msg = f"已达到最大步数限制（{self.MAX_STEPS} 步），任务强制结束"
        self._record_trace(step, "error", timeout_msg)
        print(f"\n⚠️  {timeout_msg}")

        return {
            "success": False,
            "result": timeout_msg,
            "steps_taken": step,
            "trace": self.trace
        }

    def print_trace(self):
        """
        打印完整的执行轨迹（方便事后复盘）
        类比：就像看一部电影的分镜脚本，每步都看得清清楚楚
        """
        print("\n" + "=" * 60)
        print("📋 完整执行轨迹")
        print("=" * 60)
        for entry in self.trace:
            print(f"\nStep {entry['step']} | {entry['type'].upper()}")
            print(f"  {entry['content']}")


# ============================================================
# 第四部分：演示主程序
# ============================================================

def main():
    """
    演示 ReAct Agent 执行多步骤任务
    """
    print("🤖 ReAct Agent 演示\n")

    agent = ReActAgent()

    # ---- 任务1：搜索信息并做计算 ----
    print("\n" + "█"*60)
    print("任务1：搜索 + 计算组合任务")
    print("█"*60)

    result1 = agent.run(
        "帮我搜索一下 Python asyncio 的基本介绍，"
        "然后计算 2 的 10 次方等于多少，最后给我一个简短的总结。"
    )

    print(f"\n最终结果：{result1['result'][:300]}")
    print(f"成功：{result1['success']}，共执行 {result1['steps_taken']} 步")

    # ---- 任务2：搜索并写入文件 ----
    print("\n" + "█"*60)
    print("任务2：搜索信息并保存到文件")
    print("█"*60)

    result2 = agent.run(
        "搜索 Claude API 的基本信息，"
        "然后把这些信息整理后写入文件 /tmp/claude_notes.txt，"
        "最后告诉我文件写入是否成功。"
    )

    print(f"\n最终结果：{result2['result'][:300]}")
    print(f"成功：{result2['success']}，共执行 {result2['steps_taken']} 步")

    # 验证文件是否真的写入了
    if os.path.exists("/tmp/claude_notes.txt"):
        print("\n✓ 验证：/tmp/claude_notes.txt 文件确实存在！")
        with open("/tmp/claude_notes.txt", "r") as f:
            preview = f.read()[:200]
        print(f"  文件预览：{preview}...")

    # ---- 打印统计 ----
    print("\n" + "=" * 60)
    print("📊 运行统计")
    print("=" * 60)
    print(f"任务1：{'成功' if result1['success'] else '失败'}，{result1['steps_taken']} 步")
    print(f"任务2：{'成功' if result2['success'] else '失败'}，{result2['steps_taken']} 步")


if __name__ == "__main__":
    main()
