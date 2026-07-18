"""
LangGraph 实现 ReAct Agent
==========================
演示如何用 LangGraph 手工搭建一个 ReAct（Reason + Act）Agent，
并与 LangChain 内置的 create_react_agent 做对比。

核心概念速查
-----------
- ReAct = Reason（推理） + Act（行动）循环
- State  = 贯穿整张图的"共享记事本"
- Node   = 图中的一个处理节点（函数）
- Edge   = 节点之间的连接，条件边决定走哪条路
- Checkpoint（检查点）= 把 State 持久化，支持恢复 / 回放
- Human-in-the-Loop = 图在执行中暂停，等人类审核再继续

依赖安装（任选其一）：
    pip install langgraph langchain-anthropic
    uv add langgraph langchain-anthropic
"""

# ──────────────────────────────────────────────
# 标准库
# ──────────────────────────────────────────────
import json
import os
from typing import Annotated, Literal, TypedDict

# ──────────────────────────────────────────────
# LangGraph 核心
# ──────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages  # 消息列表的 reducer：追加而非覆盖
from langgraph.checkpoint.memory import MemorySaver  # 内存检查点（生产用 SqliteSaver/PostgresSaver）
from langgraph.prebuilt import ToolNode              # 官方工具执行节点（省去手写循环）

# ──────────────────────────────────────────────
# LangChain / Anthropic
# ──────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ══════════════════════════════════════════════
# 1. State 设计
#    State 是整张图共享的"记事本"，每个节点都能读写它。
# ══════════════════════════════════════════════

class AgentState(TypedDict):
    # messages：对话历史；add_messages 是 reducer，保证每次只追加而不覆盖
    messages: Annotated[list, add_messages]

    # steps：已执行的推理步骤数，用来防止死循环
    steps: int

    # final_answer：Agent 认为已经得到最终答案时写入此字段
    final_answer: str


# ══════════════════════════════════════════════
# 2. 工具定义
#    用 @tool 装饰器声明，LangGraph 会自动把 docstring 当作工具描述传给 LLM
# ══════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """
    安全的数学计算器。
    支持加减乘除、幂运算、括号。
    示例：calculator("(3 + 5) * 2 ** 3")
    """
    # 只允许数字和基本运算符，防止代码注入
    allowed = set("0123456789+-*/().** ")
    if not all(c in allowed for c in expression):
        return f"错误：表达式含有不允许的字符：{expression}"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气（模拟数据，实际项目替换为真实 API）。
    示例：get_weather("北京")
    """
    weather_db = {
        "北京": "晴，25°C，东风 3 级",
        "上海": "多云，22°C，东南风 2 级",
        "广州": "小雨，28°C，南风 4 级",
        "深圳": "阴，27°C，偏南风 3 级",
    }
    return weather_db.get(city, f"暂无 {city} 的天气数据")


@tool
def search_knowledge(query: str) -> str:
    """
    检索本地知识库（模拟）。
    示例：search_knowledge("LangGraph 是什么")
    """
    kb = {
        "langgraph": "LangGraph 是 LangChain 团队开发的图编排框架，专门用于构建有状态的 Agent。",
        "react": "ReAct = Reasoning + Acting，让 LLM 交替进行推理和工具调用。",
        "agent":  "Agent 是能自主决策、调用工具、循环执行直到完成目标的 AI 程序。",
    }
    key = query.lower()
    for k, v in kb.items():
        if k in key:
            return v
    return f"未找到与 '{query}' 相关的条目"


# 把工具放进列表，便于后续统一绑定
TOOLS = [calculator, get_weather, search_knowledge]

# ══════════════════════════════════════════════
# 3. LLM 初始化并绑定工具
# ══════════════════════════════════════════════

def build_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    llm = ChatAnthropic(
        model="claude-opus-4-5",
        api_key=api_key,
        temperature=0,
        max_tokens=1024,
    )
    return llm.bind_tools(TOOLS)  # 告诉 LLM 它可以调用哪些工具


LLM = build_llm()

# 系统提示：告诉 LLM 它是一个 ReAct Agent
SYSTEM_PROMPT = SystemMessage(content="""
你是一个 ReAct Agent，按照以下循环工作：
1. 思考（Thought）：分析用户问题，决定是否需要调用工具
2. 行动（Action）：如果需要，调用合适的工具
3. 观察（Observation）：查看工具返回结果
4. 重复直到可以给出最终答案

可用工具：calculator（数学计算）、get_weather（查天气）、search_knowledge（检索知识）
当你认为已经得到完整答案时，直接回复用户，不要再调用工具。
""".strip())


# ══════════════════════════════════════════════
# 4. llm_node：调用 LLM，决定下一步
# ══════════════════════════════════════════════

def llm_node(state: AgentState) -> AgentState:
    """
    推理节点：把当前对话历史发给 LLM，得到回复。
    LLM 可能返回：
      - 包含 tool_calls 的 AIMessage → 需要执行工具
      - 普通文本 AIMessage → 最终答案，结束循环
    """
    messages = [SYSTEM_PROMPT] + state["messages"]
    response: AIMessage = LLM.invoke(messages)

    # 如果没有工具调用，说明 LLM 认为已有最终答案
    final = "" if response.tool_calls else response.content

    return {
        "messages": [response],
        "steps": state.get("steps", 0) + 1,
        "final_answer": final,
    }


# ══════════════════════════════════════════════
# 5. tool_node：执行工具
#    使用 LangGraph 内置的 ToolNode，它会自动：
#    - 解析 AIMessage 中的 tool_calls
#    - 依次调用对应的工具函数
#    - 把结果包装成 ToolMessage 追加到 messages
# ══════════════════════════════════════════════

tool_executor = ToolNode(TOOLS)

# ══════════════════════════════════════════════
# 6. 条件边：should_continue
#    根据 State 决定下一个节点是"执行工具"还是"结束"
# ══════════════════════════════════════════════

MAX_STEPS = 10  # 最多循环次数，防止无限递归

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    条件边函数，返回值是下一个节点的名字：
    - "tools"   → 继续执行工具（LLM 发出了工具调用请求）
    - "__end__" → 结束图（已有最终答案或超出步骤限制）
    """
    last_message = state["messages"][-1]

    # 超出步骤限制，强制结束
    if state.get("steps", 0) >= MAX_STEPS:
        print(f"[警告] 已达到最大步骤数 {MAX_STEPS}，强制结束")
        return "__end__"

    # LLM 发出了工具调用请求
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 没有工具调用 → 最终答案，结束
    return "__end__"


# ══════════════════════════════════════════════
# 7. 构建图：把节点和边拼接成完整的 ReAct 循环
# ══════════════════════════════════════════════

def build_graph(checkpointer=None, human_in_loop: bool = False):
    """
    构建 ReAct Agent 图。

    参数：
      checkpointer   - 检查点对象，传入则启用持久化对话
      human_in_loop  - True 时在执行工具前暂停，等待人类审核
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_executor)

    # 入口：从 START 进入 llm 节点
    graph.add_edge(START, "llm")

    # 条件边：llm 节点执行完后，根据 should_continue 决定去哪
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools":   "tools",   # 有工具调用 → 去 tools 节点
            "__end__": END,       # 无工具调用 → 结束
        },
    )

    # 工具执行完后，无条件回到 llm 节点（ReAct 核心循环）
    graph.add_edge("tools", "llm")

    # Human-in-the-Loop：在进入 tools 节点之前暂停
    # 此时图会抛出 GraphInterrupt，外部代码可以审核 / 修改后再 resume
    interrupt_before = ["tools"] if human_in_loop else []

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


# ══════════════════════════════════════════════
# 8. 流式输出：逐节点打印执行过程
# ══════════════════════════════════════════════

def stream_agent(app, user_input: str, thread_id: str = "demo-thread"):
    """
    用 .stream() 以流式方式运行 Agent，
    每当一个节点执行完毕就立即打印它的输出。
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "steps": 0,
        "final_answer": "",
    }

    print(f"\n{'='*60}")
    print(f"用户：{user_input}")
    print(f"{'='*60}")

    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n[节点: {node_name}]")

            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  → 调用工具: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
                        elif msg.content:
                            print(f"  → AI 回复: {msg.content}")

                    elif isinstance(msg, ToolMessage):
                        print(f"  → 工具结果 [{msg.name}]: {msg.content}")

            if node_output.get("final_answer"):
                print(f"\n最终答案: {node_output['final_answer']}")

    print(f"\n{'='*60}\n")


# ══════════════════════════════════════════════
# 9. Human-in-the-Loop 演示
# ══════════════════════════════════════════════

def demo_human_in_loop(user_input: str, thread_id: str = "hitl-thread"):
    """
    演示 Human-in-the-Loop：
    1. 图在调用工具前自动暂停（interrupt_before=["tools"]）
    2. 打印待执行的工具调用，让用户确认
    3. 用户输入 y 继续，n 中止
    """
    memory = MemorySaver()
    app = build_graph(checkpointer=memory, human_in_loop=True)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"[Human-in-the-Loop 模式] 用户：{user_input}")
    print(f"{'='*60}")

    # 第一次运行，会在 tools 节点前暂停
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "steps": 0,
        "final_answer": "",
    }

    # stream 直到暂停（GraphInterrupt）
    pending_tool_calls = []
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        pending_tool_calls = msg.tool_calls
                        for tc in msg.tool_calls:
                            print(f"\n[待审核工具调用] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

    if pending_tool_calls:
        approval = input("\n是否允许执行以上工具调用？(y/n): ").strip().lower()
        if approval != "y":
            print("用户拒绝，Agent 终止。")
            return

        # 用户批准 → 从检查点恢复并继续执行（传入 None 表示不修改 State）
        print("\n用户已批准，继续执行...\n")
        for event in app.stream(None, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                print(f"[节点: {node_name}]")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, ToolMessage):
                            print(f"  → 工具结果 [{msg.name}]: {msg.content}")
                        elif isinstance(msg, AIMessage) and msg.content:
                            print(f"  → 最终答案: {msg.content}")

    print(f"\n{'='*60}\n")


# ══════════════════════════════════════════════
# 10. 对比：LangChain 内置的 create_react_agent
# ══════════════════════════════════════════════

def compare_with_prebuilt(user_input: str):
    """
    使用 LangGraph 的 create_react_agent（官方预置，更简洁）
    与上面手工搭建的版本做对比。

    手工搭建 vs create_react_agent：
    ┌──────────────────┬────────────────────┬──────────────────────┐
    │ 特性              │ 手工搭建            │ create_react_agent   │
    ├──────────────────┼────────────────────┼──────────────────────┤
    │ 代码量            │ 多（~100 行）       │ 少（3 行）            │
    │ 自定义 State      │ 完全自由            │ 只能用内置 MessagesState│
    │ 自定义节点逻辑    │ 完全自由            │ 受限                  │
    │ 条件边            │ 自定义              │ 内置                  │
    │ Human-in-the-Loop │ 支持               │ 支持（参数传入）       │
    │ 适用场景          │ 复杂 / 定制 Agent  │ 快速原型 / 标准场景   │
    └──────────────────┴────────────────────┴──────────────────────┘
    """
    try:
        from langgraph.prebuilt import create_react_agent  # noqa: PLC0415
    except ImportError:
        print("[跳过] create_react_agent 未找到，请升级 langgraph >= 0.2")
        return

    llm_plain = ChatAnthropic(
        model="claude-opus-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
        temperature=0,
        max_tokens=1024,
    )

    # 一行创建 Agent（内部自动完成 State / 节点 / 边的构建）
    prebuilt_app = create_react_agent(
        model=llm_plain,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "prebuilt-demo"}}
    print(f"\n{'='*60}")
    print(f"[create_react_agent 对比] 用户：{user_input}")
    print(f"{'='*60}")

    result = prebuilt_app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    last_msg = result["messages"][-1]
    print(f"回复：{last_msg.content}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════
# 11. 主程序：把所有演示串起来
# ══════════════════════════════════════════════

def main():
    # ── 11-A. 基础演示（无检查点） ──
    print("\n【演示 A：基础 ReAct Agent（无持久化）】")
    basic_app = build_graph()
    stream_agent(basic_app, "帮我计算 (123 + 456) * 789 是多少？", thread_id="basic-1")

    # ── 11-B. 多工具调用 ──
    print("\n【演示 B：多工具协作】")
    stream_agent(basic_app, "北京今天天气怎么样？顺便告诉我 LangGraph 是什么。", thread_id="basic-2")

    # ── 11-C. MemorySaver 检查点（同一 thread_id 保留对话上下文） ──
    print("\n【演示 C：MemorySaver 持久化对话（多轮）】")
    memory = MemorySaver()
    memory_app = build_graph(checkpointer=memory)

    # 第一轮
    stream_agent(memory_app, "我叫小明，帮我算 100 * 200。", thread_id="memory-abc")
    # 第二轮：Agent 会记住上文，知道用户叫小明
    stream_agent(memory_app, "刚才那道题的结果再乘以 3 是多少？", thread_id="memory-abc")

    # ── 11-D. Human-in-the-Loop ──
    # 注意：此处会触发 input()，在非交互环境中请注释掉
    # print("\n【演示 D：Human-in-the-Loop】")
    # demo_human_in_loop("查一下上海的天气")

    # ── 11-E. 对比 create_react_agent ──
    print("\n【演示 E：对比 LangChain create_react_agent】")
    compare_with_prebuilt("深圳天气怎么样？")


if __name__ == "__main__":
    main()
