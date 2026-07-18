"""
LangChain Agent 完整示例
=========================
本文件演示 LangChain Agent 的核心用法，包括：
1. @tool 装饰器定义工具
2. BaseTool 子类（复杂工具）
3. create_react_agent（内置 ReAct 策略）
4. AgentExecutor 执行器
5. ConversationBufferMemory 对话记忆
6. 流式 Agent 输出
7. LangSmith 追踪（LANGCHAIN_TRACING_V2=true）
8. 内置工具使用（Calculator、PythonREPL 等）

安装依赖：
    pip install langchain langchain-openai langchain-community langchain-experimental

环境变量（可选）：
    OPENAI_API_KEY=sk-...          # LLM 调用密钥
    LANGCHAIN_TRACING_V2=true      # 开启 LangSmith 追踪
    LANGCHAIN_API_KEY=ls__...      # LangSmith API 密钥
    LANGCHAIN_PROJECT=my-project   # LangSmith 项目名
"""

import os
from typing import Any, Optional, Type

# ── 1. @tool 装饰器 ──────────────────────────────────────────────────────────
# @tool 是最简单的工具定义方式：写一个普通函数，加上装饰器就变成 Agent 可用的工具。
# 函数的 docstring 会被 LLM 读取，用来决定"什么时候该调用这个工具"，所以要写清楚。

from langchain.tools import tool


@tool
def add_numbers(expression: str) -> str:
    """
    对两个整数求和。
    输入格式："数字1,数字2"，例如 "3,5"。
    返回两数之和。
    """
    try:
        parts = expression.split(",")
        a, b = int(parts[0].strip()), int(parts[1].strip())
        result = a + b
        return f"{a} + {b} = {result}"
    except Exception as e:
        return f"计算出错，请检查输入格式（正确示例：3,5）。错误：{e}"


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气。
    输入城市名称（中文或英文均可），返回模拟天气信息。
    """
    # 实际项目中这里会调用真实的天气 API，此处用模拟数据演示
    mock_weather = {
        "北京": "晴天，气温 28°C，湿度 40%，东南风 3 级",
        "上海": "多云，气温 32°C，湿度 75%，东风 2 级",
        "广州": "阵雨，气温 35°C，湿度 85%，南风 1 级",
    }
    weather = mock_weather.get(city, f"{city} 的天气数据暂不可用（仅支持北京/上海/广州）")
    return weather


@tool
def search_knowledge(query: str) -> str:
    """
    在本地知识库中搜索相关信息。
    输入搜索关键词，返回匹配到的知识条目。
    """
    # 实际项目中可接入向量数据库（如 Chroma、FAISS）做语义搜索
    knowledge_base = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的 Python/JS 框架，提供链、Agent、记忆等核心抽象。",
        "agent": "Agent 是能够自主决策、调用工具、完成多步任务的 AI 系统，核心是 LLM + 工具 + 循环推理。",
        "react": "ReAct 是一种 Agent 策略：先 Reasoning（推理），再 Acting（行动），交替进行直到得出答案。",
        "memory": "LangChain Memory 组件用于在多轮对话中保存历史消息，让 LLM 能记住上下文。",
    }
    for keyword, content in knowledge_base.items():
        if keyword.lower() in query.lower():
            return content
    return f"未找到与 '{query}' 相关的知识条目。"


# ── 2. BaseTool 子类（复杂工具） ─────────────────────────────────────────────
# 当工具逻辑较复杂，需要初始化参数、状态管理或异步支持时，继承 BaseTool 更灵活。

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class FileReaderInput(BaseModel):
    """FileReaderTool 的输入参数 Schema，使用 Pydantic 定义。"""

    file_path: str = Field(description="要读取的文件路径（相对或绝对路径）")
    max_lines: int = Field(default=10, description="最多读取的行数，默认 10 行")


class FileReaderTool(BaseTool):
    """
    读取本地文本文件内容的工具。

    继承 BaseTool 的好处：
    - 可以在 __init__ 中注入依赖（如数据库连接、配置等）
    - 支持 args_schema 做严格的输入校验
    - 可以覆写 _arun 实现异步版本
    """

    name: str = "file_reader"
    description: str = (
        "读取本地文件内容。当需要查看某个文件的内容时使用此工具。"
        "输入文件路径和可选的最大行数。"
    )
    # 绑定输入 Schema，LangChain 会自动校验 Agent 传来的参数
    args_schema: Type[BaseModel] = FileReaderInput

    # BaseTool 要求实现 _run 方法（同步版本）
    def _run(self, file_path: str, max_lines: int = 10) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[:max_lines]
            content = "".join(lines)
            return f"文件前 {max_lines} 行内容：\n{content}"
        except FileNotFoundError:
            return f"文件不存在：{file_path}"
        except Exception as e:
            return f"读取文件出错：{e}"

    # 覆写 _arun 实现异步版本（可选，不实现时异步调用会回退到同步版本）
    async def _arun(self, file_path: str, max_lines: int = 10) -> str:
        # 实际项目中可使用 aiofiles 做真正的异步文件读取
        return self._run(file_path, max_lines)


# ── 3. 内置工具 ──────────────────────────────────────────────────────────────
# LangChain 提供了一些开箱即用的工具，无需自己实现。

def get_builtin_tools() -> list:
    """
    加载常用内置工具。

    注意：
    - LLMMathChain 需要一个 LLM 实例来解析数学表达式
    - PythonREPLTool 会在本地执行 Python 代码，生产环境请谨慎使用
    - WikipediaQueryRun 需要安装 wikipedia 包：pip install wikipedia
    """
    tools = []

    # PythonREPL：让 Agent 可以执行 Python 代码（强大但有安全风险）
    try:
        from langchain_experimental.tools import PythonREPLTool
        python_tool = PythonREPLTool()
        tools.append(python_tool)
        print("[内置工具] PythonREPLTool 已加载")
    except ImportError:
        print("[内置工具] PythonREPLTool 未安装（需要 langchain-experimental）")

    return tools


# ── 4. 构建 Agent（create_react_agent + AgentExecutor） ──────────────────────
# create_react_agent 将 LLM、工具列表、提示词模板组合成一个 ReAct Agent。
# AgentExecutor 是运行 Agent 的"引擎"，负责循环调用 Agent 直到任务完成。

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI


def build_agent(tools: list, verbose: bool = True) -> AgentExecutor:
    """
    构建一个基础的 ReAct Agent。

    参数：
        tools    - 工具列表，Agent 可以从中选择并调用
        verbose  - 是否打印 Agent 的思考过程（调试时非常有用）

    返回：
        AgentExecutor 实例
    """
    # 初始化 LLM（这里用 GPT-4o-mini，性价比高）
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,          # 设为 0 让推理更确定性，减少随机性
        streaming=True,         # 开启流式，配合后面的流式输出使用
    )

    # 从 LangChain Hub 拉取标准 ReAct 提示词模板
    # 这个模板告诉 LLM：你有哪些工具、如何格式化思考过程和行动
    # 模板内容大致是：
    #   "你是一个 Agent，有以下工具：{tools}
    #    按 Thought/Action/Observation 格式输出..."
    prompt = hub.pull("hwchase17/react")

    # 组合成 Agent（注意：这里只是定义，还没开始执行）
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # AgentExecutor 包裹 Agent，提供：
    # - max_iterations：防止无限循环（默认 15）
    # - handle_parsing_errors：LLM 输出格式错误时优雅处理而不崩溃
    # - verbose：打印每一步的 Thought/Action/Observation
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return executor


# ── 5. ConversationBufferMemory 对话记忆 ─────────────────────────────────────
# Memory 让 Agent 在多轮对话中记住之前说过的话。
# ConversationBufferMemory 是最简单的实现：把所有历史消息原样保存在内存里。

from langchain.memory import ConversationBufferMemory


def build_agent_with_memory(tools: list) -> AgentExecutor:
    """
    构建带有对话记忆的 Agent。

    与基础 Agent 的区别：
    - 每次对话结束后，问题和答案都会存入 memory
    - 下次对话时，历史记录会附加到提示词中，LLM 能记住上下文
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ConversationBufferMemory 配置说明：
    # - memory_key：在提示词模板中引用历史记录的变量名
    # - return_messages：True 时返回 Message 对象列表（适合 ChatModel）
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # 带记忆的 Agent 需要用支持 chat_history 变量的提示词模板
    # hwchase17/react-chat 是官方提供的带记忆版本
    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,          # 传入记忆组件
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return executor


# ── 6. 流式 Agent 输出 ───────────────────────────────────────────────────────
# 流式输出让用户不必等待 Agent 完全结束，而是看到逐字生成的过程，体验更好。

def run_agent_streaming(executor: AgentExecutor, query: str) -> None:
    """
    以流式方式运行 Agent，实时打印每个 token。

    LangChain 的流式 Agent 通过 stream() 方法实现，
    每次 yield 一个事件字典，包含 Agent 的中间步骤或最终答案。
    """
    print(f"\n[流式执行] 问题：{query}")
    print("-" * 50)

    # stream() 返回一个生成器，每次 yield 一个事件
    for chunk in executor.stream({"input": query}):
        # chunk 可能包含以下键：
        # - "actions"：Agent 决定调用某个工具（中间步骤）
        # - "steps"：工具执行完毕，返回观察结果
        # - "output"：最终答案

        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"\n[思考] 决定调用工具: {action.tool}")
                print(f"[参数] {action.tool_input}")

        elif "steps" in chunk:
            for step in chunk["steps"]:
                print(f"\n[观察] 工具返回: {step.observation}")

        elif "output" in chunk:
            print(f"\n[最终答案] {chunk['output']}")

    print("-" * 50)


# ── 7. LangSmith 追踪配置 ────────────────────────────────────────────────────
# LangSmith 是 LangChain 官方的可观测性平台，可以记录每次 LLM 调用的输入输出、
# 工具调用链路、耗时、费用等，方便调试和性能分析。

def setup_langsmith_tracing() -> None:
    """
    配置 LangSmith 追踪。

    只需设置以下环境变量，LangChain 会自动将所有调用上报到 LangSmith：
        LANGCHAIN_TRACING_V2=true        - 开启追踪
        LANGCHAIN_API_KEY=ls__...        - 你的 LangSmith API 密钥
        LANGCHAIN_PROJECT=my-project     - 追踪数据归属的项目名

    查看追踪数据：https://smith.langchain.com
    """
    # 检查是否已通过环境变量配置
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print(f"[LangSmith] 追踪已开启，项目：{project}")
        print("[LangSmith] 访问 https://smith.langchain.com 查看追踪数据")
    else:
        print("[LangSmith] 追踪未开启。如需开启，请设置：")
        print("  export LANGCHAIN_TRACING_V2=true")
        print("  export LANGCHAIN_API_KEY=ls__your_key")
        print("  export LANGCHAIN_PROJECT=my-project")

    # 也可以在代码中直接设置（不推荐，密钥不应硬编码）：
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = "ls__your_key"


# ── 8. 主函数：整合所有功能演示 ──────────────────────────────────────────────

def demo_basic_tools() -> None:
    """演示用 @tool 装饰器定义的工具（不依赖 LLM，可独立测试）。"""
    print("\n=== 演示 @tool 装饰器定义的工具 ===")

    # 直接调用工具（绕过 Agent，用于单元测试）
    result = add_numbers.invoke("3,7")
    print(f"add_numbers('3,7') => {result}")

    result = get_weather.invoke("北京")
    print(f"get_weather('北京') => {result}")

    result = search_knowledge.invoke("什么是 ReAct")
    print(f"search_knowledge('什么是 ReAct') => {result}")

    # 查看工具元信息（LLM 会读取这些信息决定何时调用工具）
    print(f"\n工具名称: {add_numbers.name}")
    print(f"工具描述: {add_numbers.description}")


def demo_base_tool() -> None:
    """演示 BaseTool 子类的使用（不依赖 LLM，可独立测试）。"""
    print("\n=== 演示 BaseTool 子类 ===")

    reader = FileReaderTool()

    # 读取一个不存在的文件，测试错误处理
    result = reader.invoke({"file_path": "/tmp/not_exist.txt", "max_lines": 5})
    print(f"读取不存在的文件: {result}")

    # 创建临时文件并读取
    tmp_path = "/tmp/langchain_demo.txt"
    with open(tmp_path, "w") as f:
        f.write("第1行：LangChain 是构建 LLM 应用的框架\n")
        f.write("第2行：Agent 可以自主调用工具完成任务\n")
        f.write("第3行：Memory 让对话有上下文记忆\n")

    result = reader.invoke({"file_path": tmp_path, "max_lines": 2})
    print(f"读取临时文件（前2行）:\n{result}")


def demo_full_agent() -> None:
    """
    完整 Agent 演示（需要 OPENAI_API_KEY 环境变量）。

    如果没有 API Key，这个函数会打印提示信息后跳过。
    """
    print("\n=== 完整 Agent 演示 ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("[跳过] 未设置 OPENAI_API_KEY，跳过需要 LLM 的演示。")
        print("  设置方法：export OPENAI_API_KEY=sk-...")
        return

    # 检查 LangSmith 配置
    setup_langsmith_tracing()

    # 组合工具列表
    tools = [
        add_numbers,        # @tool 装饰器定义的工具
        get_weather,        # @tool 装饰器定义的工具
        search_knowledge,   # @tool 装饰器定义的工具
        FileReaderTool(),   # BaseTool 子类定义的工具
    ]

    # 尝试加载内置工具
    tools.extend(get_builtin_tools())

    print(f"\n已加载 {len(tools)} 个工具：{[t.name for t in tools]}")

    # --- 基础 Agent（单轮问答）---
    print("\n--- 基础 Agent（单轮问答）---")
    executor = build_agent(tools, verbose=True)

    # Agent 会自动选择合适的工具来回答问题
    response = executor.invoke({"input": "北京今天天气怎么样？"})
    print(f"\n最终答案：{response['output']}")

    # --- 流式 Agent ---
    print("\n--- 流式 Agent 输出 ---")
    run_agent_streaming(executor, "请帮我计算 42 加上 58 等于多少？")

    # --- 带记忆的 Agent（多轮对话）---
    print("\n--- 带记忆的 Agent（多轮对话）---")
    memory_executor = build_agent_with_memory(tools)

    # 第一轮：问天气
    r1 = memory_executor.invoke({"input": "上海今天天气如何？"})
    print(f"第一轮答案：{r1['output']}")

    # 第二轮：引用上轮对话（Agent 应该记得"上海天气"的上下文）
    r2 = memory_executor.invoke({"input": "那广州呢？和前面那个城市比怎么样？"})
    print(f"第二轮答案：{r2['output']}")


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agent 完整示例")
    print("=" * 60)

    # 步骤1：演示工具定义（无需 API Key）
    demo_basic_tools()

    # 步骤2：演示 BaseTool 子类（无需 API Key）
    demo_base_tool()

    # 步骤3：完整 Agent 演示（需要 OPENAI_API_KEY）
    demo_full_agent()

    print("\n演示完成。")
    print("提示：设置 OPENAI_API_KEY 后可运行完整 Agent 演示。")
