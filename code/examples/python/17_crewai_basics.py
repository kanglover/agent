"""
CrewAI 多 Agent 系统基础
========================

本文件演示 CrewAI 框架的核心用法：
- Agent / Task / Crew 三元组
- 顺序执行 vs 层级执行
- 配置 Claude 作为 LLM
- 用 @tool 装饰器自定义工具
- 演示：研究报告自动生成（搜索员 + 写手 + 编辑）

安装依赖：
    pip install crewai crewai-tools anthropic
"""

# ============================================================
# 1. 安装说明（运行时打印，方便新手确认环境）
# ============================================================

INSTALL_NOTES = """
【安装步骤】
1. pip install crewai          # 核心框架
2. pip install crewai-tools    # 官方工具集（搜索、文件读写等）
3. pip install anthropic       # Claude SDK（若使用 Claude 作为 LLM）

环境变量（二选一）：
  export ANTHROPIC_API_KEY="sk-ant-..."   # 使用 Claude
  export OPENAI_API_KEY="sk-..."          # 使用 OpenAI（默认）
"""

import os
import json
import textwrap
from datetime import datetime
from typing import Optional

# ---- 延迟导入，避免未安装时直接崩溃 ----
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from langchain_anthropic import ChatAnthropic
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("[警告] crewai 未安装，仅展示代码结构，不实际运行。")
    print("       运行：pip install crewai langchain-anthropic")


# ============================================================
# 2. 配置 Claude 作为 LLM
# ============================================================

def get_claude_llm(model: str = "claude-opus-4-5", temperature: float = 0.3):
    """
    返回 Claude LLM 实例，供 Agent 使用。

    CrewAI 默认用 OpenAI，替换为 Claude 只需传入 llm 参数即可。
    """
    if not CREWAI_AVAILABLE:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未找到 ANTHROPIC_API_KEY，请先设置环境变量：\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        max_tokens=4096,
    )


# ============================================================
# 3. 自定义工具（@tool 装饰器 / BaseTool 两种写法）
# ============================================================

# --- 写法一：函数装饰器（快速） ---
if CREWAI_AVAILABLE:
    from crewai.tools import tool

    @tool("网络搜索工具")
    def web_search(query: str) -> str:
        """
        模拟网络搜索，返回与 query 相关的摘要信息。
        生产环境请替换为真实 API（如 Tavily、SerpAPI）。
        """
        # 模拟搜索结果（真实场景接入搜索 API）
        mock_results = {
            "AI": "人工智能（AI）2025年持续高速发展，大模型参数量突破万亿……",
            "CrewAI": "CrewAI 是一个多 Agent 协作框架，支持顺序/层级两种执行模式……",
            "Claude": "Claude 是 Anthropic 推出的 AI 助手，以安全性和长上下文著称……",
        }
        for keyword, result in mock_results.items():
            if keyword.lower() in query.lower():
                return f"[搜索结果] 关键词：{query}\n{result}"
        return f"[搜索结果] 关键词：{query}\n暂无相关内容，建议换个关键词重试。"

    @tool("文件保存工具")
    def save_to_file(content: str, filename: str = "output.md") -> str:
        """将内容保存到本地文件，返回保存路径。"""
        output_dir = "/tmp/crewai_outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已保存：{filepath}"


# --- 写法二：继承 BaseTool（复杂逻辑推荐） ---
if CREWAI_AVAILABLE:
    class WordCountTool(BaseTool):
        """统计文本字数的工具，展示 BaseTool 继承写法。"""

        name: str = "字数统计工具"
        description: str = "统计给定文本的中英文字数，返回统计报告。"

        def _run(self, text: str) -> str:
            chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
            english_words = len([w for w in text.split() if w.isascii()])
            total_chars = len(text)
            return (
                f"【字数统计】\n"
                f"  总字符数：{total_chars}\n"
                f"  中文字数：{chinese_chars}\n"
                f"  英文单词数：{english_words}"
            )


# ============================================================
# 4. Agent 定义
# ============================================================

def create_agents(llm=None):
    """
    创建三个 Agent：搜索员、写手、编辑。
    每个 Agent 有独立的 role / goal / backstory 和专属工具。
    """
    if not CREWAI_AVAILABLE:
        print("[跳过] crewai 未安装，无法创建 Agent")
        return None, None, None

    # Agent 1：搜索员（负责收集资料）
    researcher = Agent(
        role="资深研究员",
        goal="收集关于 {topic} 的最新、最准确信息，整理成结构化摘要",
        backstory=textwrap.dedent("""
            你是一位有 10 年经验的信息研究专家，擅长从海量数据中
            提炼关键信息。你对信息的准确性要求极高，从不捏造内容。
        """),
        tools=[web_search],          # 配备搜索工具
        llm=llm,
        verbose=True,                # 打印推理过程
        allow_delegation=False,      # 不允许转包给其他 Agent
        max_iter=3,                  # 最多尝试 3 轮
    )

    # Agent 2：写手（负责撰写报告）
    writer = Agent(
        role="技术写手",
        goal="根据研究员提供的资料，撰写一篇通俗易懂、结构清晰的报告",
        backstory=textwrap.dedent("""
            你是一位科技媒体的金牌作者，擅长把复杂技术用人人都能
            看懂的语言表达出来。你的文章逻辑清晰、案例丰富、读来轻松。
        """),
        tools=[word_count_tool := WordCountTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # Agent 3：编辑（负责审校和优化）
    editor = Agent(
        role="资深编辑",
        goal="审核写手的报告，修正错误、优化表达，确保报告质量达到发布标准",
        backstory=textwrap.dedent("""
            你是出版社的首席编辑，有严苛的质量标准。你会检查事实准确性、
            逻辑连贯性和语言流畅度，并给出具体的修改建议。
        """),
        tools=[save_to_file],        # 编辑完成后保存文件
        llm=llm,
        verbose=True,
        allow_delegation=True,       # 允许将局部任务转包给其他 Agent
    )

    return researcher, writer, editor


# ============================================================
# 5. Task 定义
# ============================================================

def create_tasks(researcher, writer, editor, topic: str):
    """
    为三个 Agent 各创建一个 Task。
    Task 之间通过 context 参数传递依赖关系。
    """
    if not CREWAI_AVAILABLE:
        return None, None, None

    # Task 1：搜索资料
    research_task = Task(
        description=textwrap.dedent(f"""
            请搜索关于「{topic}」的相关资料，重点涵盖：
            1. 核心概念和定义
            2. 主要特点和优势
            3. 典型应用场景
            4. 当前发展现状
            请将搜索结果整理成有条理的摘要，不少于 200 字。
        """),
        expected_output="结构化的研究摘要，包含以上 4 个方面的信息",
        agent=researcher,
    )

    # Task 2：撰写报告（依赖 Task 1 的输出）
    writing_task = Task(
        description=textwrap.dedent(f"""
            根据研究员提供的资料，撰写一篇关于「{topic}」的入门报告：
            - 标题醒目，引人入胜
            - 正文分为：背景介绍 / 核心内容 / 应用场景 / 总结展望
            - 语言通俗，避免不必要的术语
            - 字数控制在 400-600 字
            最后用字数统计工具检查字数。
        """),
        expected_output="完整的 Markdown 格式报告，字数在 400-600 字之间",
        agent=writer,
        context=[research_task],    # 依赖研究任务的输出
    )

    # Task 3：审校并保存（依赖 Task 2 的输出）
    editing_task = Task(
        description=textwrap.dedent(f"""
            审校写手提交的报告，完成以下工作：
            1. 检查是否有事实性错误
            2. 优化不通顺的表达
            3. 确认结构是否完整（标题/各章节/总结）
            4. 在报告末尾添加「审校说明」段落，注明修改要点
            5. 用文件保存工具将最终版本保存为 {topic.replace(' ', '_')}_report.md
        """),
        expected_output="经过审校的完整报告，含审校说明，并已保存到文件",
        agent=editor,
        context=[writing_task],
    )

    return research_task, writing_task, editing_task


# ============================================================
# 6. 顺序执行（Sequential）
# ============================================================

def run_sequential(topic: str = "CrewAI 框架"):
    """
    顺序执行模式：Task 1 → Task 2 → Task 3，像流水线一样依次完成。
    适合：有明确依赖关系的线性流程。
    """
    print("\n" + "=" * 60)
    print("【顺序执行模式】Sequential Process")
    print("=" * 60)

    llm = get_claude_llm()
    researcher, writer, editor = create_agents(llm)
    t1, t2, t3 = create_tasks(researcher, writer, editor, topic)

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[t1, t2, t3],
        process=Process.sequential,  # 顺序执行
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result


# ============================================================
# 7. 层级执行（Hierarchical）
# ============================================================

def run_hierarchical(topic: str = "CrewAI 框架"):
    """
    层级执行模式：自动创建一个「管理者 Agent」，
    管理者负责任务分配和协调，下属 Agent 专注执行。
    适合：任务复杂、需要灵活调度的场景。
    """
    print("\n" + "=" * 60)
    print("【层级执行模式】Hierarchical Process")
    print("=" * 60)

    llm = get_claude_llm()
    researcher, writer, editor = create_agents(llm)
    t1, t2, t3 = create_tasks(researcher, writer, editor, topic)

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[t1, t2, t3],
        process=Process.hierarchical,   # 层级执行
        manager_llm=llm,                # 管理者使用的 LLM
        verbose=True,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result


# ============================================================
# 8. 输出解析和保存
# ============================================================

def parse_and_save_result(result, output_path: Optional[str] = None):
    """
    解析 Crew 的执行结果并保存到本地文件。

    result.raw          -> 最终输出的原始文本
    result.token_usage  -> Token 消耗统计
    result.tasks_output -> 每个 Task 的输出列表
    """
    print("\n" + "=" * 60)
    print("【输出解析】")
    print("=" * 60)

    # 提取最终文本
    final_text = result.raw if hasattr(result, "raw") else str(result)
    print(f"最终输出（前 300 字）：\n{final_text[:300]}...")

    # Token 用量
    if hasattr(result, "token_usage") and result.token_usage:
        usage = result.token_usage
        print(f"\nToken 用量：")
        print(f"  输入 tokens：{getattr(usage, 'prompt_tokens', 'N/A')}")
        print(f"  输出 tokens：{getattr(usage, 'completion_tokens', 'N/A')}")

    # 各 Task 输出
    if hasattr(result, "tasks_output") and result.tasks_output:
        print(f"\n各 Task 输出摘要：")
        for i, task_out in enumerate(result.tasks_output, 1):
            summary = str(task_out)[:100].replace("\n", " ")
            print(f"  Task {i}：{summary}...")

    # 保存到文件
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/tmp/crewai_outputs/final_{ts}.md"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# CrewAI 报告\n\n生成时间：{datetime.now()}\n\n")
        f.write(final_text)

    print(f"\n完整报告已保存：{output_path}")
    return output_path


# ============================================================
# 9. 纯结构展示（无需 API Key 也能运行）
# ============================================================

def show_structure_demo():
    """
    不依赖 API Key，打印 CrewAI 核心概念和代码结构说明。
    适合第一次了解框架时快速建立直觉。
    """
    print("\n" + "=" * 60)
    print("【CrewAI 核心概念速览】")
    print("=" * 60)

    concepts = {
        "Agent（智能体）": "有角色、目标、背景故事和工具的 AI 成员，类比公司里的员工",
        "Task（任务）":    "分配给某个 Agent 的具体工作，包含描述、期望输出和依赖关系",
        "Crew（团队）":    "把多个 Agent 和 Task 组合在一起，决定执行方式（顺序/层级）",
        "Tool（工具）":    "Agent 可以调用的外部能力，如搜索、写文件、计算等",
        "Process（流程）": "Sequential=流水线；Hierarchical=有管理者的树状调度",
    }

    for name, desc in concepts.items():
        print(f"\n  {name}")
        print(f"    └─ {desc}")

    print("\n" + "=" * 60)
    print("【执行模式对比】")
    print("=" * 60)
    print("""
  顺序执行（Sequential）
    搜索员 → 写手 → 编辑
    特点：简单、可预期、适合线性流程

  层级执行（Hierarchical）
          管理者
         /  |  \\
      搜索员 写手 编辑
    特点：灵活、自动调度、适合复杂任务
    代价：多消耗一个管理者的 LLM 调用
    """)

    print("=" * 60)
    print("【@tool 装饰器示例（可复制使用）】")
    print("=" * 60)
    code_snippet = '''
from crewai.tools import tool

@tool("天气查询工具")
def get_weather(city: str) -> str:
    """查询指定城市的天气，返回温度和天气描述。"""
    # 替换为真实 API 调用
    return f"{city}：晴，25°C，空气质量优"
    '''
    print(code_snippet)


# ============================================================
# 10. 主入口
# ============================================================

def main():
    print(INSTALL_NOTES)
    show_structure_demo()

    if not CREWAI_AVAILABLE:
        print("\n[提示] 安装 crewai 后重新运行，即可执行真实的多 Agent 任务。")
        return

    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not has_key:
        print("\n[提示] 未检测到 API Key，跳过实际运行。")
        print("  设置后重新运行：export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    # 选择执行模式（默认顺序执行）
    mode = os.getenv("CREWAI_MODE", "sequential").lower()
    topic = os.getenv("CREWAI_TOPIC", "CrewAI 多 Agent 框架")

    print(f"\n执行模式：{mode}，研究主题：{topic}")

    if mode == "hierarchical":
        result = run_hierarchical(topic)
    else:
        result = run_sequential(topic)

    parse_and_save_result(result)


if __name__ == "__main__":
    main()
