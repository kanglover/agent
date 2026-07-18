"""
LangGraph Multi-Agent System
============================
架构：
  Supervisor Agent（路由决策）
    ├── 研究员 Worker（搜索 / 信息收集）
    ├── 程序员 Worker（写代码）
    └── 审查员 Worker（Code Review）

流程：
  用户任务 → Supervisor 路由 → Worker 执行 → 回报 Supervisor
           → 继续路由 或 FINISH

共享 State 贯穿全流程，记录消息历史与性能指标。
"""

from __future__ import annotations

import time
import textwrap
from typing import Annotated, Literal, TypedDict

# ── LangChain / LangGraph ────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────────────────────────────────────
# 1. 共享 State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """全局共享状态 —— 每个节点读写同一份字典。"""

    # 对话消息列表（add_messages 自动追加，而非覆盖）
    messages: Annotated[list[BaseMessage], add_messages]

    # Supervisor 下一步要调用的 Worker，或 "FINISH" 表示终止
    next: str

    # 当前任务的原始描述
    task: str

    # 研究员收集到的背景信息
    research: str

    # 程序员产出的代码
    code: str

    # 审查员的 Review 意见
    review: str

    # 修订轮次计数（防止死循环）
    revision_count: int

    # ── 性能监控 ──────────────────────────────────────────────────────────────
    # 每个 agent 的 token 消耗：{"supervisor": {"input": x, "output": y}, ...}
    token_usage: dict[str, dict[str, int]]

    # 每个 agent 的耗时（秒）
    timing: dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM 工厂（统一创建，方便替换模型）
# ─────────────────────────────────────────────────────────────────────────────

def make_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    返回一个支持 token 计数的 ChatOpenAI 实例。
    实际使用时替换为你的 API Key 和模型名称。
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        # 开启 stream_usage 以便从响应中读取 token 消耗
        stream_usage=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. 性能监控工具函数
# ─────────────────────────────────────────────────────────────────────────────

def record_usage(
    state: AgentState,
    agent_name: str,
    response: AIMessage,
    elapsed: float,
) -> dict:
    """从 AIMessage 提取 token 消耗并写入 state 的监控字段。"""
    token_usage = dict(state.get("token_usage", {}))
    timing = dict(state.get("timing", {}))

    # LangChain 把 usage 放在 response.usage_metadata
    usage_meta = getattr(response, "usage_metadata", None) or {}
    token_usage[agent_name] = {
        "input": usage_meta.get("input_tokens", 0),
        "output": usage_meta.get("output_tokens", 0),
        "total": usage_meta.get("total_tokens", 0),
    }
    timing[agent_name] = round(elapsed, 3)

    return {"token_usage": token_usage, "timing": timing}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Supervisor Agent（路由决策）
# ─────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """\
你是一个多智能体系统的 Supervisor（调度员）。
团队成员：
  - researcher：负责收集背景信息、调研技术方案
  - coder：负责根据需求和研究结果编写 Python 代码
  - reviewer：负责对代码做 Code Review，找出 Bug 和改进点

你的职责：
1. 分析当前任务和历史消息，决定下一步应该调用哪个成员。
2. 如果审查员的 Review 通过（没有重大问题），输出 FINISH。
3. 最多允许 2 轮修改（revision_count >= 2 时强制 FINISH）。

只输出一个词：researcher / coder / reviewer / FINISH
"""

def supervisor_node(state: AgentState) -> dict:
    """Supervisor：读取全局状态，决定路由目标。"""
    llm = make_llm(temperature=0)

    context_parts = [f"任务：{state['task']}"]
    if state.get("research"):
        context_parts.append(f"研究结果：{state['research'][:500]}")
    if state.get("code"):
        context_parts.append(f"当前代码（前300字）：{state['code'][:300]}")
    if state.get("review"):
        context_parts.append(f"Review 意见：{state['review'][:400]}")
    context_parts.append(f"修订轮次：{state.get('revision_count', 0)}")

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    t0 = time.time()
    response: AIMessage = llm.invoke(messages)
    elapsed = time.time() - t0

    decision = response.content.strip().lower()
    # 容错：只取第一个词
    decision = decision.split()[0] if decision.split() else "FINISH"
    valid = {"researcher", "coder", "reviewer", "finish"}
    if decision not in valid:
        decision = "FINISH"

    perf = record_usage(state, "supervisor", response, elapsed)
    print(f"[Supervisor] 路由决策 → {decision.upper()}  ({elapsed:.2f}s)")

    return {
        "next": decision,
        "messages": [AIMessage(content=f"[Supervisor] 路由 → {decision}", name="supervisor")],
        **perf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Worker Agents
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. 研究员 ────────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """\
你是一名资深技术研究员。
根据用户任务，收集相关背景知识、技术方案和最佳实践。
输出结构化的研究报告（500字以内），供程序员参考。
"""

def researcher_node(state: AgentState) -> dict:
    """研究员：收集背景信息。"""
    llm = make_llm(temperature=0.5)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"请为以下任务提供技术调研报告：\n\n{state['task']}"),
    ]

    t0 = time.time()
    response: AIMessage = llm.invoke(messages)
    elapsed = time.time() - t0

    perf = record_usage(state, "researcher", response, elapsed)
    print(f"[研究员] 调研完成 ({elapsed:.2f}s)，输出 {len(response.content)} 字")

    return {
        "research": response.content,
        "messages": [AIMessage(content=response.content, name="researcher")],
        **perf,
    }


# ── 5b. 程序员 ────────────────────────────────────────────────────────────────

CODER_SYSTEM = """\
你是一名高级 Python 工程师。
根据任务描述和研究报告编写高质量 Python 代码。
要求：
- 包含完整的类型注解
- 每个函数有 docstring
- 包含使用示例
- 如果有 Review 意见，必须逐条修复

只输出 Python 代码块，不要额外解释。
"""

def coder_node(state: AgentState) -> dict:
    """程序员：根据任务和研究结果编写代码。"""
    llm = make_llm(temperature=0.2)

    user_content = f"任务：{state['task']}\n\n研究报告：\n{state.get('research', '无')}"
    if state.get("review"):
        user_content += f"\n\nCode Review 意见（请全部修复）：\n{state['review']}"

    messages = [
        SystemMessage(content=CODER_SYSTEM),
        HumanMessage(content=user_content),
    ]

    t0 = time.time()
    response: AIMessage = llm.invoke(messages)
    elapsed = time.time() - t0

    # 更新 revision_count（每次 coder 被调用算一次修订）
    revision_count = state.get("revision_count", 0)
    # 第一次写代码不算修订，有 review 时才算
    if state.get("review"):
        revision_count += 1

    perf = record_usage(state, "coder", response, elapsed)
    print(f"[程序员] 代码完成 ({elapsed:.2f}s)，第 {revision_count} 轮修订")

    return {
        "code": response.content,
        "revision_count": revision_count,
        "messages": [AIMessage(content=response.content, name="coder")],
        **perf,
    }


# ── 5c. 审查员 ────────────────────────────────────────────────────────────────

REVIEWER_SYSTEM = """\
你是一名严格的代码审查员。
请对提供的 Python 代码进行全面 Code Review，检查：
1. 正确性（逻辑错误、边界条件）
2. 安全性（注入、资源泄漏）
3. 可读性（命名、注释）
4. 性能（算法复杂度）
5. Python 最佳实践

输出格式：
- 如果代码质量合格，第一行写"APPROVED"，然后简要说明优点。
- 如果有问题，第一行写"NEEDS_REVISION"，然后逐条列出必须修复的问题。
"""

def reviewer_node(state: AgentState) -> dict:
    """审查员：对代码进行 Code Review。"""
    llm = make_llm(temperature=0.1)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"请 Review 以下代码：\n\n{state.get('code', '（无代码）')}"),
    ]

    t0 = time.time()
    response: AIMessage = llm.invoke(messages)
    elapsed = time.time() - t0

    perf = record_usage(state, "reviewer", response, elapsed)
    approved = response.content.strip().upper().startswith("APPROVED")
    print(f"[审查员] Review 完成 ({elapsed:.2f}s) → {'通过' if approved else '需要修改'}")

    return {
        "review": response.content,
        "messages": [AIMessage(content=response.content, name="reviewer")],
        **perf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. 路由函数（决定下一个节点）
# ─────────────────────────────────────────────────────────────────────────────

def route_from_supervisor(
    state: AgentState,
) -> Literal["researcher", "coder", "reviewer", "__end__"]:
    """
    Supervisor → Worker 的路由边。
    读取 state["next"] 决定跳转目标。
    "finish" / "FINISH" → END
    """
    nxt = state.get("next", "FINISH").lower()
    if nxt == "finish":
        return END
    return nxt  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# 7. 构建 Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    节点：supervisor / researcher / coder / reviewer
    边：
      supervisor --条件路由--> researcher | coder | reviewer | END
      researcher --> supervisor（回报）
      coder      --> supervisor（回报）
      reviewer   --> supervisor（回报）
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder",      coder_node)
    graph.add_node("reviewer",   reviewer_node)

    # 入口
    graph.set_entry_point("supervisor")

    # Supervisor → Worker（条件路由）
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "coder":      "coder",
            "reviewer":   "reviewer",
            END:          END,
        },
    )

    # Worker → Supervisor（回报边）
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("coder",      "supervisor")
    graph.add_edge("reviewer",   "supervisor")

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. 性能报告
# ─────────────────────────────────────────────────────────────────────────────

def print_performance_report(final_state: AgentState) -> None:
    """打印每个 Agent 的 token 消耗与耗时统计。"""
    print("\n" + "=" * 60)
    print("性能监控报告")
    print("=" * 60)

    token_usage: dict = final_state.get("token_usage", {})
    timing: dict = final_state.get("timing", {})

    total_input = total_output = total_time = 0.0
    fmt = "{:<12} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Agent", "输入Token", "输出Token", "总Token", "耗时(s)"))
    print("-" * 60)

    for agent in ["supervisor", "researcher", "coder", "reviewer"]:
        usage = token_usage.get(agent, {})
        inp = usage.get("input", 0)
        out = usage.get("output", 0)
        tot = usage.get("total", inp + out)
        t   = timing.get(agent, 0.0)

        # supervisor 可能被调用多次，这里只记录最后一次（简化）
        total_input  += inp
        total_output += out
        total_time   += t

        print(fmt.format(agent, inp, out, tot, f"{t:.3f}"))

    print("-" * 60)
    print(fmt.format(
        "合计",
        int(total_input),
        int(total_output),
        int(total_input + total_output),
        f"{total_time:.3f}",
    ))
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 9. 主流程演示
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TASK = textwrap.dedent("""\
    请实现一个 Python 线程安全的 LRU 缓存（Least Recently Used Cache）。
    要求：
    1. 支持 get(key) 和 put(key, value) 操作，时间复杂度均为 O(1)
    2. 容量满时自动淘汰最久未使用的条目
    3. 线程安全（使用 threading.Lock）
    4. 支持过期时间（TTL，单位：秒；可选参数，默认不过期）
    5. 提供命中率统计接口 stats()
    6. 完整的单元测试
""")


def run_demo(task: str = DEMO_TASK, *, dry_run: bool = False) -> AgentState:
    """
    运行完整的 Multi-Agent 工作流。

    Args:
        task:     要完成的编程任务描述
        dry_run:  True 时使用 Mock LLM，不调用真实 API（用于单测）

    Returns:
        最终的共享 State
    """
    if dry_run:
        return _run_dry(task)

    print("\n" + "=" * 60)
    print("LangGraph Multi-Agent 系统启动")
    print("=" * 60)
    print(f"任务：\n{task}\n")

    app = build_graph()

    initial_state: AgentState = {
        "messages":       [HumanMessage(content=task)],
        "next":           "",
        "task":           task,
        "research":       "",
        "code":           "",
        "review":         "",
        "revision_count": 0,
        "token_usage":    {},
        "timing":         {},
    }

    # 运行 Graph，stream 每个步骤的输出
    final_state: AgentState = initial_state
    step = 0
    for chunk in app.stream(initial_state, {"recursion_limit": 20}):
        step += 1
        node_name = list(chunk.keys())[0]
        print(f"\n── Step {step}: {node_name} ──")
        # 合并 chunk 到 final_state（简化合并）
        for k, v in chunk[node_name].items():
            if k == "messages":
                final_state["messages"] = final_state.get("messages", []) + (v or [])
            elif k in ("token_usage", "timing"):
                existing = dict(final_state.get(k, {}))
                existing.update(v or {})
                final_state[k] = existing  # type: ignore[literal-required]
            else:
                final_state[k] = v  # type: ignore[literal-required]

    # 最终输出
    print("\n" + "=" * 60)
    print("最终产出代码（前 800 字）：")
    print("=" * 60)
    print(final_state.get("code", "（无代码）")[:800])

    print_performance_report(final_state)
    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# 10. Dry-Run（Mock，不调用真实 API）
# ─────────────────────────────────────────────────────────────────────────────

def _run_dry(task: str) -> AgentState:
    """
    不调用 LLM，用固定数据模拟完整流程，用于演示和单测。
    流程：supervisor→researcher→supervisor→coder→supervisor→reviewer→supervisor→FINISH
    """
    print("\n[Dry-Run 模式] 跳过真实 LLM 调用，使用 Mock 数据\n")

    mock_research = "LRU 常见实现：OrderedDict + Lock。O(1) 需要双向链表 + 哈希表。"
    mock_code = textwrap.dedent("""\
        import threading
        from collections import OrderedDict

        class LRUCache:
            def __init__(self, capacity: int, ttl: float | None = None):
                self._cap = capacity
                self._ttl = ttl
                self._cache: OrderedDict = OrderedDict()
                self._lock = threading.Lock()
                self._hits = self._misses = 0

            def get(self, key):
                with self._lock:
                    if key not in self._cache:
                        self._misses += 1
                        return None
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]

            def put(self, key, value):
                with self._lock:
                    if key in self._cache:
                        self._cache.move_to_end(key)
                    self._cache[key] = value
                    if len(self._cache) > self._cap:
                        self._cache.popitem(last=False)

            def stats(self):
                total = self._hits + self._misses
                return {"hits": self._hits, "misses": self._misses,
                        "hit_rate": self._hits / total if total else 0.0}
    """)
    mock_review = "APPROVED\n代码结构清晰，线程安全，建议补充 TTL 过期检查逻辑。"

    state: AgentState = {
        "messages":       [HumanMessage(content=task)],
        "next":           "FINISH",
        "task":           task,
        "research":       mock_research,
        "code":           mock_code,
        "review":         mock_review,
        "revision_count": 0,
        "token_usage": {
            "supervisor": {"input": 120, "output": 5,   "total": 125},
            "researcher": {"input": 200, "output": 300, "total": 500},
            "coder":      {"input": 400, "output": 600, "total": 1000},
            "reviewer":   {"input": 700, "output": 150, "total": 850},
        },
        "timing": {
            "supervisor": 0.5,
            "researcher": 1.2,
            "coder":      2.8,
            "reviewer":   1.5,
        },
    }

    print("[Dry-Run] 模拟流程：")
    print("  supervisor → researcher → supervisor → coder → supervisor → reviewer → supervisor → FINISH")
    print(f"\n研究报告：{state['research']}")
    print(f"\n代码片段（前300字）：\n{state['code'][:300]}")
    print(f"\nReview：{state['review']}")

    print_performance_report(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 11. 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # 传入 --dry-run 参数时使用 Mock 模式，无需真实 OpenAI Key
    dry = "--dry-run" in sys.argv
    run_demo(dry_run=dry)
