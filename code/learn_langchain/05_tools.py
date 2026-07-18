"""
05 - 工具调用：让模型会用工具，而不是只会说话
==============================================
【这章学什么】
  - bind_tools：给模型"挂"上工具
  - 工具调用循环：模型决定调工具 → 我们执行 → 结果回传模型 → 最终回答
  - 这是 Agent（智能体）的基础：Agent = 模型 + 工具 + 循环

【为什么学它】
  模型本身只会"说话"，不会"做事"。比如你问"北京天气如何"，它只能猜，
  因为它没法上网查。但如果你给它一个"查天气的工具"，它能决定：
  "这个问题我需要调用 get_weather 工具，参数是北京"——然后我们把工具执行结果
  喂回给它，它就能给出真实答案了。

  这就是从"聊天机器人"到"会办事的智能体"的关键一步。

【类比】
  - 普通模型 = 一个只会嘴上说说的顾问（"北京大概挺热的吧"）
  - 带工具的模型 = 同一个顾问，但桌上放了电话、计算器（"我查一下…北京 28°C 晴"）
  bind_tools 就是"把工具摆到它桌上"。但它不会自己打电话——你(代码)要替它拨。

【关键：工具调用循环】
  模型说"我要调工具X(参数Y)"  →  你执行工具X  →  把结果告诉模型  →  模型据此回答
  这四步缺一不可，bind_tools 只做了第一步的"声明"。

【运行】
  cd code
  uv run python learn_langchain/05_tools.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)


# ── 1. 定义工具：普通函数 + @tool 装饰器 ────────────────────
# @tool 把一个普通函数变成"模型能看懂的工具"。
# ⚠️ 函数的 docstring（三引号注释）非常重要——模型就是靠读它来判断
#    "什么时候该用这个工具、参数怎么填"。所以要写得清楚。
print("=" * 60)
print("1. 定义工具")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。输入城市名（如"北京"）。"""
    # 演示用模拟数据；真实项目这里调天气 API
    mock = {"北京": "晴 28°C", "上海": "多云 25°C", "广州": "雷阵雨 30°C"}
    return mock.get(city, f"{city}：暂无天气数据")


@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式，例如 '12 * 8' 或 '100 / 4'。"""
    # ⚠️ 演示用 eval，生产环境有安全风险，可用 ast.literal_eval 或 sympy
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算失败：{e}"


print(f"工具1：{get_weather.name} —— {get_weather.description}")
print(f"工具2：{calculate.name} —— {calculate.description}")
print()


# ── 2. 把工具"挂"到模型上 ──────────────────────────────────
# bind_tools 之后，模型回复时可能会带上 tool_calls（工具调用请求）
llm_with_tools = llm.bind_tools([get_weather, calculate])


# ── 3. 工具调用循环（重点！）──────────────────────────────
def ask(question: str):
    """完整的工具调用流程。"""
    print(f"\n❓ 问题：{question}")

    # 第1步：把问题给"带工具的模型"，看它要不要调工具
    messages = [HumanMessage(content=question)]
    response = llm_with_tools.invoke(messages)

    # 如果模型没要工具，直接就回答了
    if not response.tool_calls:
        print(f"💡 直接回答：{response.content}")
        return

    # 第2步：模型要调工具了。逐个执行它请求的工具
    messages.append(response)  # 记住模型的决定，留作上下文
    tools_map = {"get_weather": get_weather, "calculate": calculate}

    for tc in response.tool_calls:
        name = tc["name"]
        args = tc["args"]
        print(f"🔧 模型决定调工具：{name}({args})")

        # 执行对应工具
        observation = tools_map[name].invoke(args)
        print(f"   工具返回：{observation}")

        # 第3步：把工具结果作为 ToolMessage 回传给模型
        # tool_call_id 用来"对账"——告诉模型这是哪个工具调用的结果
        messages.append(ToolMessage(content=str(observation), tool_call_id=tc["id"]))

    # 第4步：模型看到工具结果后，给出最终回答
    final = llm_with_tools.invoke(messages)
    print(f"💡 最终回答：{final.content}")


print("=" * 60)
print("2. 工具调用演示")
print("=" * 60)

ask("北京今天天气怎么样？")        # → 应调 get_weather
ask("帮我算一下 256 乘以 16")       # → 应调 calculate
ask("用一句话解释什么是大语言模型")  # → 不需工具，直接答


# ── 小结 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结：")
print("1. @tool 把普通函数变成工具；docstring 是模型选工具的依据，要写清楚")
print("2. bind_tools 把工具挂到模型上；回复里可能带 tool_calls")
print("3. 工具调用循环：模型请求 → 执行工具 → 结果回传 → 模型作答（4步缺一不可）")
print("4. Agent 本质就是把这个循环自动化、多轮化（下一章学记忆，再后面就是 Agent）")
print("=" * 60)
