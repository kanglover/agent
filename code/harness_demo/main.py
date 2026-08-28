"""Harness Demo 入口：组装六大子系统并运行完整演示。

演示内容:
  1. 标准任务: 计算圆面积 + 搜索相关知识（多步 think→act→observe）
  2. 上下文压缩: 小窗口下自动摘要历史消息
  3. 约束中止: 步数超限触发安全熔断
  4. 记忆持久化与快照恢复: 断点状态保存/还原
  5. 观测报告: trace 与运行指标摘要

运行: python3 main.py
"""

import os
import sys
import tempfile

from harness import (
    MockLLM,
    ContextManager,
    ToolRegistry,
    CalculatorTool,
    SearchTool,
    FileReadTool,
    Memory,
    Observer,
    ConstraintManager,
    Orchestrator,
    default_script,
)


def build_orchestrator(script=None, max_tokens=4096, max_steps=20, storage_path=None, verbose=True):
    """工厂函数: 组装六大子系统为一个可运行的编排器。"""
    llm = MockLLM(script if script is not None else default_script())
    context = ContextManager(
        system_prompt="你是一个乐于助人的 AI 助手，可以调用工具完成计算和搜索任务。",
        max_tokens=max_tokens,
    )
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(SearchTool())
    tools.register(FileReadTool())
    memory = Memory(storage_path=storage_path)
    observer = Observer()
    constraints = ConstraintManager(max_steps=max_steps, max_tokens=100000)
    return Orchestrator(llm, context, tools, memory, observer, constraints, verbose=verbose)


def section(title: str) -> None:
    print(f"\n{'#' * 64}\n# {title}\n{'#' * 64}")


# ----------------------------------------------------------------------
# 演示 1: 标准多步任务
# ----------------------------------------------------------------------

def demo_standard_run():
    section("演示 1: 标准多步任务（计算 + 搜索 + 总结）")
    orch = build_orchestrator()
    result = orch.run("计算半径为 5 的圆的面积，并搜索圆面积公式的相关知识，最后总结。")

    print("\n>> 状态:", orch.memory.state.status)
    print(">> 步数:", orch.memory.state.step_count)
    print(">> 短期记忆 keys:", orch.memory.keys())

    print("\n" + orch.observer.summary())
    return orch


# ----------------------------------------------------------------------
# 演示 2: 上下文自动压缩
# ----------------------------------------------------------------------

def demo_context_compression():
    section("演示 2: 上下文超限自动摘要压缩")
    # 构造一个长对话脚本，配一个很小的上下文窗口
    long_script = []
    for i in range(1, 13):
        long_script.append({
            "content": f"这是第 {i} 轮思考，我在持续分析问题……（此消息故意写得很长以触发上下文压缩）" * 3,
            "tool_calls": [{
                "id": f"call_{i}",
                "name": "calculator",
                "arguments": {"expression": f"{i} ** 2"},
            }],
            "finish_reason": "tool_calls",
        })
    long_script.append({"content": "分析完成，最终答案已经得出。", "finish_reason": "stop"})

    orch = build_orchestrator(script=long_script, max_tokens=300)
    orch.run("一个会持续多轮计算的长任务")

    cm = orch.context
    print(f"\n>> 摘要压缩的历史消息数: {cm.summarized_count}")
    print(f">> 压缩后上下文 token 估算: {cm.get_token_count()} / 上限 {cm.max_tokens}")
    print(f">> 当前保留消息数: {len(cm.messages)}")
    print(f">> 首条消息: {cm.messages[0].role}: {cm.messages[0].content[:70]}...")


# ----------------------------------------------------------------------
# 演示 3: 约束熔断（步数超限）
# ----------------------------------------------------------------------

def demo_constraint_abort():
    section("演示 3: 约束熔断（max_steps=3 触发安全中止）")
    # 无限循环的脚本: 永远请求工具调用
    infinite_script = [{
        "content": "我还需要继续计算……",
        "tool_calls": [{
            "id": f"call_{i}",
            "name": "calculator",
            "arguments": {"expression": f"{i} + 1"},
        }],
        "finish_reason": "tool_calls",
    } for i in range(1, 100)]

    orch = build_orchestrator(script=infinite_script, max_steps=3)
    result = orch.run("一个停不下来的任务")

    print("\n>> 返回值:", result[:80])
    print(">> 状态:", orch.memory.state.status)
    print(">> 违规记录:", orch.constraints.get_violations())
    print(">> 熔断时步数:", orch.memory.state.step_count)


# ----------------------------------------------------------------------
# 演示 4: 记忆持久化 + 快照恢复
# ----------------------------------------------------------------------

def demo_memory_and_snapshot():
    section("演示 4: 长期记忆持久化 + 快照/恢复")
    storage = os.path.join(tempfile.gettempdir(), "harness_demo_memory.json")
    if os.path.exists(storage):
        os.remove(storage)

    # 第一次运行: 写入长期记忆
    orch = build_orchestrator(storage_path=storage, verbose=False)
    orch.run("计算半径为 5 的圆的面积。")
    orch.memory.remember("favorite_topic", "geometry")
    snap = orch.memory.snapshot()
    print(">> 已写入长期记忆: favorite_topic =", orch.memory.recall("favorite_topic"))
    print(">> 已生成快照: status =", snap["state"]["status"], ", steps =", snap["state"]["step_count"])

    # 新实例从磁盘恢复长期记忆（模拟跨会话）
    memory2 = Memory(storage_path=storage)
    print(">> 跨会话 recall: favorite_topic =", memory2.recall("favorite_topic"))

    # 快照恢复（模拟断点恢复）
    memory2.restore(snap)
    print(">> 快照恢复后状态: task =", memory2.state.task[:30], ", status =", memory2.state.status)


# ----------------------------------------------------------------------
# 演示 5: 工具错误与恢复策略
# ----------------------------------------------------------------------

def demo_error_recovery():
    section("演示 5: 工具错误记录与 Recovery 策略")
    bad_script = [
        {
            "content": "先算一下……",
            "tool_calls": [{
                "id": "call_1",
                "name": "calculator",
                "arguments": {"expression": "import os"},  # 非法表达式 → 工具报错
            }],
            "finish_reason": "tool_calls",
        },
        {
            "content": "表达式非法，改用合法表达式重试。",
            "tool_calls": [{
                "id": "call_2",
                "name": "calculator",
                "arguments": {"expression": "2 ** 10"},
            }],
            "finish_reason": "tool_calls",
        },
        {"content": "2 的 10 次方等于 1024。", "finish_reason": "stop"},
    ]

    orch = build_orchestrator(script=bad_script)
    result = orch.run("先试一个非法表达式，再重试正确表达式。")

    m = orch.observer.get_metrics()
    print("\n>> 工具调用总数:", m["tool_calls"], "其中错误:", m["tool_errors"])
    print(">> 任务仍正常完成，状态:", orch.memory.state.status)
    print(">> 上下文中的错误反馈:",
          orch.context.messages[2].content[:60] if len(orch.context.messages) > 2 else "(无)")


# ----------------------------------------------------------------------

def main():
    print("Agent Harness Demo — 六大子系统演示")
    print("模块: 上下文管理 / 工具系统 / 执行编排 / 记忆与状态 / 评估观测 / 约束与恢复")

    demo_standard_run()
    demo_context_compression()
    demo_constraint_abort()
    demo_memory_and_snapshot()
    demo_error_recovery()

    section("全部演示结束")


if __name__ == "__main__":
    main()
