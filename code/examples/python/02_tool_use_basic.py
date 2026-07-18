# 运行: python 02_tool_use_basic.py
# 依赖: pip install anthropic
# 说明: 演示如何给 Claude 配备"工具"，让它能查天气、算数、搜笔记
#
# 核心概念：Tool Use（工具调用）
# 类比：Claude 本身只会"动脑"，工具就是它的"手"
#       你告诉 Claude "你有一双手，左手能查天气，右手能算数"
#       它自己决定什么时候用哪只手，你负责真正去执行

import anthropic  # Anthropic SDK
import json       # 用于解析/序列化 JSON 数据

# ============================================================
# 第一部分：工具定义（用 JSON Schema 描述工具）
# ============================================================
# JSON Schema 是一种"说明书"格式，告诉 Claude：
#   - 这个工具叫什么名字
#   - 它能做什么（description）
#   - 调用它需要传哪些参数（parameters）
# 类比：就像菜单，告诉顾客有哪些菜、每道菜的食材

TOOLS = [
    # ---- 工具1：查询天气 ----
    {
        "name": "get_weather",       # 工具名，Claude 会用这个名字来"叫"这个工具
        "description": (             # description 很重要！Claude 靠这个判断什么时候用这个工具
            "查询指定城市的当前天气信息。"
            "返回温度、天气状况和湿度。"
        ),
        "input_schema": {            # input_schema 描述这个工具接受什么参数
            "type": "object",        # type: "object" 表示参数是一个对象（字典）
            "properties": {          # properties 列出所有可用的参数字段
                "city": {
                    "type": "string",       # type: "string" 表示这个参数是字符串
                    "description": "城市名称，例如：北京、上海、广州"
                    # description 帮助 Claude 理解这个参数填什么
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    # enum 表示这个参数只能是列表中的值之一
                    "description": "温度单位：celsius（摄氏度）或 fahrenheit（华氏度）"
                }
            },
            "required": ["city"]     # required 列出必填参数，不填就报错
            # unit 没有在 required 里，所以是可选的
        }
    },

    # ---- 工具2：数学计算 ----
    {
        "name": "calculate",
        "description": (
            "执行数学计算。支持加减乘除、幂运算、括号等基本运算。"
            "当用户需要精确计算时使用，避免 Claude 自己心算出错。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式字符串，例如：'2 + 3 * 4' 或 '(100 - 20) / 4'"
                }
            },
            "required": ["expression"]   # expression 是必填的，没有就没法算
        }
    },

    # ---- 工具3：搜索本地笔记 ----
    {
        "name": "search_notes",
        "description": (
            "在本地笔记文件中搜索相关内容。"
            "当用户询问之前记录的信息或需要查找笔记时使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如：'Python 异步' 或 'API 调用方法'"
                },
                "max_results": {
                    "type": "integer",          # integer 表示整数类型
                    "description": "最多返回多少条结果，默认为 5",
                    "default": 5                # default 提示 Claude 不填时用什么默认值
                }
            },
            "required": ["query"]
        }
    }
]


# ============================================================
# 第二部分：工具的 Mock 实现（假的实现，用于演示）
# ============================================================
# 真实项目中，这里会调用真实的天气 API、计算引擎、数据库
# 现在用 mock（假数据）代替，这样不需要真实 API key 也能跑

def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    查询天气的 Mock 实现
    真实版本会调用：OpenWeatherMap API、高德天气 API 等
    """
    print(f"  [工具执行] get_weather(city={city!r}, unit={unit!r})")

    # Mock 数据：假装真的查了一下
    mock_data = {
        "北京": {"temp": 28, "condition": "晴天", "humidity": 45},
        "上海": {"temp": 32, "condition": "多云", "humidity": 78},
        "广州": {"temp": 35, "condition": "雷阵雨", "humidity": 90},
    }

    # 查不到的城市返回默认数据
    weather = mock_data.get(city, {"temp": 25, "condition": "未知", "humidity": 60})

    # 如果要华氏度，做一下换算
    if unit == "fahrenheit":
        weather["temp"] = weather["temp"] * 9 / 5 + 32
        temp_unit = "°F"
    else:
        temp_unit = "°C"

    result = {
        "city": city,
        "temperature": f"{weather['temp']}{temp_unit}",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%"
    }
    print(f"  [工具结果] {result}")
    return result


def calculate(expression: str) -> dict:
    """
    数学计算的 Mock 实现
    使用 Python 内置的 eval()（注意：真实生产环境要用更安全的解析器）
    """
    print(f"  [工具执行] calculate(expression={expression!r})")

    try:
        # eval() 可以执行字符串形式的 Python 表达式
        # 类比：就像计算器，你输入 "2+3" 它输出 5
        result_value = eval(expression)
        result = {
            "expression": expression,
            "result": result_value,
            "success": True
        }
    except Exception as e:
        # 如果表达式有语法错误，返回错误信息
        result = {
            "expression": expression,
            "error": str(e),
            "success": False
        }

    print(f"  [工具结果] {result}")
    return result


def search_notes(query: str, max_results: int = 5) -> dict:
    """
    搜索笔记的 Mock 实现
    真实版本会搜索本地文件系统或数据库
    """
    print(f"  [工具执行] search_notes(query={query!r}, max_results={max_results})")

    # Mock 笔记数据库
    mock_notes = [
        {"id": 1, "title": "Python 异步编程入门", "snippet": "asyncio 是 Python 的异步框架，使用 async/await 语法..."},
        {"id": 2, "title": "Claude API 调用方法", "snippet": "使用 anthropic.Anthropic() 创建客户端，调用 messages.create()..."},
        {"id": 3, "title": "Tool Use 工具调用", "snippet": "通过 tools 参数传入工具定义，Claude 会在需要时调用工具..."},
        {"id": 4, "title": "Prompt 写作技巧", "snippet": "好的 Prompt 要明确角色、任务、格式和约束条件..."},
        {"id": 5, "title": "流式输出 Streaming", "snippet": "使用 stream() 上下文管理器，通过 text_stream 迭代获取文字块..."},
    ]

    # 简单的关键词匹配搜索
    query_lower = query.lower()
    matched = [
        note for note in mock_notes
        if query_lower in note["title"].lower() or query_lower in note["snippet"].lower()
    ]

    # 限制返回数量
    matched = matched[:max_results]

    result = {
        "query": query,
        "total_found": len(matched),
        "results": matched
    }
    print(f"  [工具结果] 找到 {len(matched)} 条结果")
    return result


# ============================================================
# 第三部分：工具分发器（把工具名映射到实际函数）
# ============================================================

# 工具注册表：key = 工具名，value = 对应的 Python 函数
# 类比：就像公司的分机号码表，"转1号" = 接线员，"转2号" = 技术支持
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_notes": search_notes,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    执行工具调用，返回结果的 JSON 字符串

    参数：
        tool_name  = Claude 选择的工具名
        tool_input = Claude 传入的参数（字典格式）
    返回：
        工具执行结果，转为 JSON 字符串，方便回传给 Claude
    """
    if tool_name not in TOOL_REGISTRY:
        # 工具不存在，返回错误
        error_result = {"error": f"工具 '{tool_name}' 不存在"}
        print(f"  [错误] {error_result}")
        return json.dumps(error_result, ensure_ascii=False)

    try:
        # 动态调用对应的函数，** 是字典解包，把字典展开为关键字参数
        # 类比：func(**{"a": 1, "b": 2}) 等价于 func(a=1, b=2)
        tool_func = TOOL_REGISTRY[tool_name]
        result = tool_func(**tool_input)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        # 工具执行出错，返回错误信息而不是崩溃
        error_result = {"error": f"工具执行失败: {str(e)}"}
        print(f"  [执行错误] {error_result}")
        return json.dumps(error_result, ensure_ascii=False)


# ============================================================
# 第四部分：完整的 Tool Use 调用流程
# ============================================================

def run_with_tools(user_message: str):
    """
    完整的工具调用流程演示

    流程图：
    用户提问
      ↓
    Claude 分析（可能决定调用工具）
      ↓
    如果 stop_reason == "tool_use"：
      → 解析 Claude 想调用哪些工具
      → 本地执行这些工具
      → 把结果回传给 Claude
      → Claude 再次分析，生成最终回复
    如果 stop_reason == "end_turn"：
      → 直接输出回复（不需要工具）
    """
    print("\n" + "=" * 60)
    print(f"用户提问：{user_message}")
    print("=" * 60)

    client = anthropic.Anthropic()

    # 初始化对话历史
    messages = [{"role": "user", "content": user_message}]

    # 循环处理，直到 Claude 给出最终答案（stop_reason == "end_turn"）
    # 为什么要循环？因为工具调用可能发生多次（链式工具调用）
    step = 0
    while True:
        step += 1
        print(f"\n--- 第 {step} 轮 API 调用 ---")

        # 调用 Claude，传入工具列表
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,            # 把工具列表告诉 Claude
            messages=messages       # 发送完整对话历史
        )

        print(f"  stop_reason = {response.stop_reason}")
        print(f"  content blocks 数量 = {len(response.content)}")

        # 把 Claude 的回复（可能包含工具调用请求）加入历史
        # 注意：这里要把整个 content 列表加进去，不只是文字
        messages.append({"role": "assistant", "content": response.content})

        # ---- 情况1：Claude 想调用工具 ----
        if response.stop_reason == "tool_use":
            print("  Claude 要调用工具，开始处理...")

            # 收集所有工具调用结果
            tool_results = []

            # response.content 可能包含多个 block
            # 有些是 TextBlock（思考过程文字），有些是 ToolUseBlock（工具调用请求）
            for block in response.content:
                if block.type == "tool_use":
                    # 这是一个工具调用请求
                    print(f"\n  Claude 要调用工具：{block.name}")
                    print(f"  工具参数：{json.dumps(block.input, ensure_ascii=False)}")

                    # 执行工具
                    result_str = execute_tool(block.name, block.input)

                    # 把工具结果打包成 tool_result 格式
                    # tool_use_id 要和请求里的 block.id 对应，Claude 靠这个匹配结果
                    tool_results.append({
                        "type": "tool_result",      # 固定写法，表示这是工具返回值
                        "tool_use_id": block.id,    # 对应之前的工具调用 ID
                        "content": result_str       # 工具执行结果（字符串格式）
                    })

                elif block.type == "text" and block.text:
                    # Claude 在调用工具前可能会说几句话（思考过程）
                    print(f"  Claude 的思考：{block.text[:100]}...")

            # 把所有工具结果作为 user 消息回传给 Claude
            # 这是协议规定的格式：工具结果必须以 "user" 角色发送
            messages.append({
                "role": "user",
                "content": tool_results    # 可能包含多个工具结果（并行调用时）
            })

        # ---- 情况2：Claude 给出最终答案 ----
        elif response.stop_reason == "end_turn":
            # 提取最终文字回复
            final_reply = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_reply += block.text

            print(f"\n{'='*60}")
            print(f"Claude 最终回答：\n{final_reply}")
            print(f"{'='*60}")
            break  # 退出循环

        else:
            # 其他情况（理论上不应该出现）
            print(f"  未预期的 stop_reason: {response.stop_reason}")
            break


# ============================================================
# 第五部分：演示并行工具调用
# ============================================================

def demo_parallel_tool_use():
    """
    演示 Claude 同时调用多个工具（并行调用）
    类比：Claude 同时伸出两只手，左手查天气，右手算数
    """
    print("\n" + "=" * 60)
    print("【演示】并行工具调用")
    print("=" * 60)

    # 这个问题需要同时查天气和做计算
    user_message = "北京现在多少摄氏度？同时告诉我 (28 + 35) / 2 等于多少？"
    run_with_tools(user_message)


# ============================================================
# 主程序
# ============================================================

def main():
    print("\n🔧 Tool Use 工具调用示例\n")

    # 演示1：需要查天气的问题
    run_with_tools("上海今天天气怎么样？")

    # 演示2：需要计算的问题
    run_with_tools("我有 1200 元，买了 3 件商品分别是 299、399、199 元，还剩多少钱？")

    # 演示3：需要搜索笔记的问题
    run_with_tools("帮我查一下我的笔记里有没有关于 Claude API 的内容？")

    # 演示4：并行工具调用
    demo_parallel_tool_use()

    print("\n✅ Tool Use 示例运行完毕！")
    print()
    print("核心要点：")
    print("  1. 工具定义用 JSON Schema 格式，description 写清楚触发条件")
    print("  2. stop_reason == 'tool_use' 表示 Claude 要调用工具")
    print("  3. 执行工具后，用 role='user' + type='tool_result' 把结果回传")
    print("  4. 一次响应可能包含多个 tool_use block（并行调用）")
    print("  5. 循环调用直到 stop_reason == 'end_turn' 才算完成")


if __name__ == "__main__":
    main()
