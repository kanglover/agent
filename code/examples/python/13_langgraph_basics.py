"""
LangGraph 入门指南
==================

LangGraph 是什么？
    LangGraph 是一个用来构建「有状态、多步骤」AI 工作流的框架。
    可以把它想象成一张"流程图"：
      - 每个"节点"（node）是一个处理步骤
      - 每条"边"（edge）决定下一步去哪
      - 整张图共享一个"状态"（state），记录当前进展

安装：
    pip install langgraph

    如果需要可视化功能（mermaid图）：
    pip install langgraph[draw]
    # 或者
    pip install grandalf

官方文档：https://langchain-ai.github.io/langgraph/
"""

# ============================================================
# 第一部分：核心概念 — State（状态）
# ============================================================
#
# State 就像一个"共享白板"，所有节点都可以读取和修改它。
# 用 TypedDict 来定义它的结构（规定白板上有哪些字段）。
#
# TypedDict 是 Python 的类型提示工具，
# 让你像写字典一样定义数据结构，同时有类型检查。

from typing import TypedDict, Annotated, Sequence
import operator


# 最简单的 State：只有一个消息列表
class SimpleState(TypedDict):
    messages: list[str]   # 对话消息列表
    step_count: int        # 当前是第几步


# 稍复杂的 State：带 Annotated + operator.add 自动追加
# operator.add 的意思是：当节点返回新的 messages 时，
# 不是"替换"旧的，而是"追加"到后面——避免覆盖历史记录。
class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]  # 自动追加，不覆盖
    user_input: str
    response: str
    loop_count: int


# ============================================================
# 第二部分：节点（Node）
# ============================================================
#
# 节点就是一个普通的 Python 函数：
#   - 接收当前 state（字典）
#   - 返回一个字典（只包含需要更新的字段）
#
# LangGraph 会把你返回的字典"合并"到全局 state 中。
# 你不需要返回完整的 state，只返回"改变了什么"就行。

def greet_node(state: SimpleState) -> dict:
    """
    打招呼节点：读取 state，做处理，返回更新内容。
    """
    print(f"[greet_node] 当前步骤：{state['step_count']}")
    new_message = "你好！欢迎使用 LangGraph。"
    return {
        "messages": state["messages"] + [new_message],
        "step_count": state["step_count"] + 1,
    }


def process_node(state: SimpleState) -> dict:
    """
    处理节点：模拟一些"业务逻辑"。
    """
    print(f"[process_node] 当前消息数：{len(state['messages'])}")
    new_message = "正在处理您的请求..."
    return {
        "messages": state["messages"] + [new_message],
        "step_count": state["step_count"] + 1,
    }


def farewell_node(state: SimpleState) -> dict:
    """
    告别节点：流程的最后一步。
    """
    print(f"[farewell_node] 流程结束，共 {state['step_count']} 步")
    new_message = "感谢使用，再见！"
    return {
        "messages": state["messages"] + [new_message],
        "step_count": state["step_count"] + 1,
    }


# ============================================================
# 第三部分：构建第一个图（StateGraph）
# ============================================================

from langgraph.graph import StateGraph, START, END

def build_simple_graph():
    """
    构建最简单的线性流程图：
    START → greet → process → farewell → END
    """

    # 1. 创建图，传入 State 类型
    #    这告诉 LangGraph："这张图的状态长什么样"
    graph = StateGraph(SimpleState)

    # 2. 添加节点（add_node）
    #    第一个参数：节点的名字（字符串，随便起）
    #    第二个参数：对应的函数
    graph.add_node("greet", greet_node)
    graph.add_node("process", process_node)
    graph.add_node("farewell", farewell_node)

    # 3. 添加边（add_edge）
    #    表示：从节点 A 执行完后，固定跳到节点 B
    #    START 是特殊的起始节点（不需要定义，直接用）
    #    END   是特殊的终止节点（流程到这里就结束）
    graph.add_edge(START, "greet")       # 开始 → 打招呼
    graph.add_edge("greet", "process")   # 打招呼 → 处理
    graph.add_edge("process", "farewell") # 处理 → 告别
    graph.add_edge("farewell", END)       # 告别 → 结束

    # 4. 编译（compile）
    #    compile() 把图"锁定"成可运行的状态
    #    编译后才能调用 invoke / stream
    app = graph.compile()

    return app


# ============================================================
# 第四部分：条件边（add_conditional_edges）
# ============================================================
#
# 有时候，下一步去哪不是固定的，要根据当前 state 来判断。
# 这就需要"条件边"——类似 if/else 的路由逻辑。

class DecisionState(TypedDict):
    number: int
    result: str


def check_number_node(state: DecisionState) -> dict:
    """读取数字，准备做判断"""
    print(f"[check_number] 收到数字：{state['number']}")
    return {}  # 不改变 state，只是个"检查站"


def handle_even_node(state: DecisionState) -> dict:
    """处理偶数"""
    return {"result": f"{state['number']} 是偶数"}


def handle_odd_node(state: DecisionState) -> dict:
    """处理奇数"""
    return {"result": f"{state['number']} 是奇数"}


def route_by_parity(state: DecisionState) -> str:
    """
    路由函数：根据 state 返回下一个节点的名字（字符串）。
    这个函数会被 add_conditional_edges 调用。
    返回值必须是某个已注册节点的名字，或者 END。
    """
    if state["number"] % 2 == 0:
        return "handle_even"   # 偶数 → 去 handle_even 节点
    else:
        return "handle_odd"    # 奇数 → 去 handle_odd 节点


def build_conditional_graph():
    """
    构建带条件分支的图：
    START → check → (偶数→handle_even / 奇数→handle_odd) → END
    """
    graph = StateGraph(DecisionState)

    graph.add_node("check", check_number_node)
    graph.add_node("handle_even", handle_even_node)
    graph.add_node("handle_odd", handle_odd_node)

    graph.add_edge(START, "check")

    # add_conditional_edges 参数说明：
    #   第1个：从哪个节点出发
    #   第2个：路由函数（返回节点名的函数）
    #   第3个（可选）：路由映射字典 {"返回值": "节点名"}
    #                  如果路由函数直接返回节点名，可以省略第3个参数
    graph.add_conditional_edges(
        "check",           # 从 check 节点出发
        route_by_parity,   # 调用这个函数来决定去哪
        {
            "handle_even": "handle_even",  # 返回 "handle_even" → 去 handle_even
            "handle_odd":  "handle_odd",   # 返回 "handle_odd"  → 去 handle_odd
        }
    )

    graph.add_edge("handle_even", END)
    graph.add_edge("handle_odd", END)

    return graph.compile()


# ============================================================
# 第五部分：invoke 和 stream（两种运行方式）
# ============================================================

def demo_invoke_and_stream():
    """
    invoke：一次性运行，等所有节点跑完，返回最终 state
    stream：流式运行，每跑完一个节点就"吐出"一次当前更新
    """

    app = build_simple_graph()

    # --- invoke 方式 ---
    print("\n" + "="*50)
    print("【invoke 方式】一次性运行，返回最终状态")
    print("="*50)

    initial_state = {
        "messages": [],
        "step_count": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n最终 state：")
    print(f"  step_count: {final_state['step_count']}")
    print(f"  messages:")
    for msg in final_state["messages"]:
        print(f"    - {msg}")

    # --- stream 方式 ---
    print("\n" + "="*50)
    print("【stream 方式】流式运行，每步都能看到")
    print("="*50)

    initial_state = {
        "messages": [],
        "step_count": 0,
    }

    # stream 返回一个迭代器，每次迭代得到 {节点名: 该节点的返回值}
    for step_output in app.stream(initial_state):
        node_name = list(step_output.keys())[0]
        node_result = step_output[node_name]
        print(f"\n节点 [{node_name}] 完成，返回了：{node_result}")


# ============================================================
# 第六部分：状态更新机制详解
# ============================================================

def demo_state_update():
    """
    演示两种 state 更新方式的区别：
    1. 普通字段：直接替换（新值覆盖旧值）
    2. Annotated + operator.add：追加（新值加到旧值后面）
    """

    print("\n" + "="*50)
    print("【状态更新机制】")
    print("="*50)

    # 情况1：普通字段，直接覆盖
    class OverwriteState(TypedDict):
        value: str

    def node_a(state):
        return {"value": "来自节点A"}

    def node_b(state):
        # 这会覆盖节点A的结果
        return {"value": "来自节点B（覆盖了A）"}

    g1 = StateGraph(OverwriteState)
    g1.add_node("a", node_a)
    g1.add_node("b", node_b)
    g1.add_edge(START, "a")
    g1.add_edge("a", "b")
    g1.add_edge("b", END)
    result1 = g1.compile().invoke({"value": "初始值"})
    print(f"\n普通字段（覆盖）：{result1['value']}")
    # 输出：来自节点B（覆盖了A）

    # 情况2：Annotated + operator.add，自动追加
    class AppendState(TypedDict):
        history: Annotated[list[str], operator.add]

    def node_c(state):
        return {"history": ["来自节点C"]}

    def node_d(state):
        return {"history": ["来自节点D"]}

    g2 = StateGraph(AppendState)
    g2.add_node("c", node_c)
    g2.add_node("d", node_d)
    g2.add_edge(START, "c")
    g2.add_edge("c", "d")
    g2.add_edge("d", END)
    result2 = g2.compile().invoke({"history": ["初始值"]})
    print(f"Annotated追加：{result2['history']}")
    # 输出：['初始值', '来自节点C', '来自节点D']


# ============================================================
# 第七部分：可视化（Mermaid 图）
# ============================================================

def demo_visualization():
    """
    LangGraph 可以把图结构输出为 Mermaid 格式的文本，
    然后在 Mermaid 官网（https://mermaid.live/）粘贴查看流程图。
    """
    print("\n" + "="*50)
    print("【可视化：Mermaid 图】")
    print("="*50)

    app = build_simple_graph()

    try:
        # get_graph().draw_mermaid() 返回 Mermaid 格式的字符串
        mermaid_text = app.get_graph().draw_mermaid()
        print("\nMermaid 图（复制到 https://mermaid.live/ 查看）：")
        print(mermaid_text)
    except Exception as e:
        print(f"可视化需要额外依赖（grandalf），当前环境不支持：{e}")
        print("手动描述这张图的结构：")
        print("  START → greet → process → farewell → END")

    # 也可以获取图的节点和边信息
    graph_obj = app.get_graph()
    print(f"\n图中的节点：{list(graph_obj.nodes.keys())}")
    print(f"图中的边：{[(e.source, e.target) for e in graph_obj.edges]}")


# ============================================================
# 第八部分：调试技巧（打印每步 state）
# ============================================================

def debug_wrapper(node_fn, node_name: str):
    """
    调试包装器：在节点执行前后打印 state，方便排查问题。
    这是一种"装饰器"思路——不改变原函数，只在外面套一层。
    """
    def wrapper(state):
        print(f"\n>>> 进入节点 [{node_name}]")
        print(f"    输入 state: {dict(state)}")
        result = node_fn(state)
        print(f"    返回更新: {result}")
        return result
    return wrapper


def demo_debug():
    """
    演示如何在图中加入调试打印，观察每一步的 state 变化。
    """
    print("\n" + "="*50)
    print("【调试技巧：打印每步 state】")
    print("="*50)

    graph = StateGraph(SimpleState)

    # 用 debug_wrapper 包装节点函数
    graph.add_node("greet",    debug_wrapper(greet_node,   "greet"))
    graph.add_node("process",  debug_wrapper(process_node, "process"))
    graph.add_node("farewell", debug_wrapper(farewell_node,"farewell"))

    graph.add_edge(START, "greet")
    graph.add_edge("greet", "process")
    graph.add_edge("process", "farewell")
    graph.add_edge("farewell", END)

    app = graph.compile()
    final = app.invoke({"messages": [], "step_count": 0})

    print(f"\n最终 state: {final}")


# ============================================================
# 第九部分：条件图完整演示
# ============================================================

def demo_conditional():
    print("\n" + "="*50)
    print("【条件边演示】根据数字奇偶走不同分支")
    print("="*50)

    app = build_conditional_graph()

    for num in [4, 7, 0, 13]:
        result = app.invoke({"number": num, "result": ""})
        print(f"  输入 {num:2d} → {result['result']}")


# ============================================================
# 主程序入口：依次运行所有演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LangGraph 入门示例")
    print("=" * 60)

    # 演示1：invoke 和 stream 两种运行方式
    demo_invoke_and_stream()

    # 演示2：状态更新机制（覆盖 vs 追加）
    demo_state_update()

    # 演示3：条件边（奇偶数分支）
    demo_conditional()

    # 演示4：可视化（Mermaid 图）
    demo_visualization()

    # 演示5：调试技巧（打印每步 state）
    demo_debug()

    print("\n" + "="*60)
    print("  全部演示完成！")
    print("="*60)
    print("""
学习路径建议：
  1. 先跑通这个文件，确认环境没问题
  2. 修改 SimpleState，添加你自己的字段
  3. 尝试写一个3节点的图，加入条件分支
  4. 用 stream 模式观察每一步的输出
  5. 看官方文档深入学习：https://langchain-ai.github.io/langgraph/
""")
