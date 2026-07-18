"""
03 - 工具调用 Agent：让图会调工具
=================================
【这章学什么】
  - 用 LangGraph 搭一个完整的 ReAct Agent（会调工具的智能体）
  - ToolNode：专门执行工具的内置节点
  - tools_condition：判断"模型要不要调工具"的条件函数
  - 用 messages 列表做 State（Agent 的标准做法）

【为什么学它】
  第05章(langchain) 学过工具调用：模型决定调工具 → 执行 → 回传 → 模型再答。
  那时是手写循环。LangGraph 把这套模式做成了标准结构：
    用户问 → 模型(可能要工具) → 工具节点(执行) → 回到模型 → ... → 结束
  这就是 ReAct Agent 的标准图谱。学会它，你就能搭"会办事"的智能体了。

【类比】
  把第02章的"循环"和"条件边"用到工具上：
  - 模型节点 = 决策者（要不要用工具？用哪个？）
  - 工具节点 = 执行者（替决策者跑腿）
  - 条件边 = 红绿灯（决策者说要用工具 → 去执行；说不用 → 结束）
  循环：执行完工具回到决策者，决策者再判断还要不要用工具。

【运行】
  cd code
  uv run python learn_langgraph/03_tool_agent.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from typing import Annotated
from typing_extensions import TypedDict
import operator

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ============================================================
# 第1步：定义工具（跟 langchain 第05章一样）
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。输入城市名（如"北京"）。"""
    mock = {"北京": "晴 28°C", "上海": "多云 25°C", "广州": "雷阵雨 30°C"}
    return mock.get(city, f"{city}：暂无天气数据")


@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式，例如 '12 * 8' 或 '100 / 4'。"""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算失败：{e}"


tools = [get_weather, calculate]


# ============================================================
# 第2步：定义 State（用 messages 列表，Agent 标准）
# ============================================================
# Agent 的状态通常就是一个消息列表，记录整个对话/思考过程。
# Annotated[..., operator.add] 表示新消息"追加"而不是"覆盖"。
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ============================================================
# 第3步：定义节点
# ============================================================
# 模型节点：把消息发给"绑了工具的模型"
def call_model(state: AgentState) -> dict:
    """模型节点：决定要不要调工具、调哪个。"""
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    # 返回新消息，会被追加到 messages（因为 operator.add）
    return {"messages": [response]}


# 工具节点：用内置 ToolNode，自动执行模型请求的工具
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)


# ============================================================
# 第4步：建图（经典 ReAct 结构）
# ============================================================
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

b = StateGraph(AgentState)

b.add_node("agent", call_model)       # 决策节点
b.add_node("tools", tool_node)        # 执行节点

b.add_edge(START, "agent")            # 先进决策节点

# 关键：条件边。tools_condition 是现成函数：
#   如果模型回复里有 tool_calls → 返回 "tools"（去执行工具）
#   如果没有 → 返回 END（直接结束）
b.add_conditional_edges(
    "agent",
    tools_condition,   # 现成的判断函数，不用自己写
    {"tools": "tools", END: END},
)

# 工具执行完 → 回到 agent（形成循环，让模型看到结果后继续判断）
b.add_edge("tools", "agent")

agent = b.compile()


# ============================================================
# 第5步：运行 Agent
# ============================================================
def ask(question: str):
    print(f"\n❓ {question}")
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    # 最后一条消息就是最终回复
    print(f"💡 {result['messages'][-1].content}")
    # 打印中间经历了哪些步骤（看 agent 调了几次工具）
    print(f"   （共 {len(result['messages'])} 条消息）")


print("=" * 60)
print("LangGraph ReAct Agent 演示")
print("=" * 60)

ask("北京今天天气怎么样？")          # 应调 get_weather
ask("帮我算一下 256 乘以 16")        # 应调 calculate
ask("用一句话解释什么是大语言模型")   # 不调工具，直接答


# ============================================================
# 第6步：用 stream 看推理过程（理解循环）
# ============================================================
print("\n" + "=" * 60)
print("用 stream 看 Agent 的推理过程")
print("=" * 60)
print("问题：上海天气如何？")
print("-" * 40)
for chunk in agent.stream({"messages": [HumanMessage(content="上海天气如何？")]}, stream_mode="updates"):
    for node, update in chunk.items():
        if node == "agent":
            msg = update["messages"][-1]
            if getattr(msg, "tool_calls", None):
                print(f"  [agent] 决定调工具: {[tc['name'] for tc in msg.tool_calls]}")
            else:
                print(f"  [agent] 给出最终回答")
        elif node == "tools":
            for m in update["messages"]:
                print(f"  [tools] 执行结果: {m.content}")


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. Agent State 通常就是 messages 列表（用 operator.add 追加）")
print("2. 两个节点：agent(模型决策) + tools(ToolNode 执行)")
print("3. tools_condition 是现成的条件函数：有 tool_calls→去tools，没有→结束")
print("4. tools → agent 的边形成循环：执行完工具回模型，模型继续判断")
print("5. 这就是 ReAct Agent 的标准图谱，理解了它=理解了 Agent 本质")
print("6. 下一章：给 Agent 加记忆（多轮对话 + 持久化）")
print("=" * 60)
