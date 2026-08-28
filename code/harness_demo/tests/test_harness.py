"""harness 六大子系统的单元与集成测试。

运行: python3 -m unittest tests.test_harness -v
（在 code/harness_demo 目录下执行）
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    MockLLM,
    LLMResponse,
    ToolCall,
    ContextManager,
    Message,
    ToolRegistry,
    ToolResult,
    CalculatorTool,
    SearchTool,
    FileReadTool,
    Memory,
    AgentState,
    Observer,
    TraceEvent,
    Constraint,
    ConstraintManager,
    Recovery,
    Orchestrator,
    AbortRunError,
    default_script,
)


def make_orchestrator(script=None, **kw):
    """测试辅助: 快速组装一个编排器。"""
    llm = MockLLM(script if script is not None else default_script())
    context = ContextManager(system_prompt="test", max_tokens=kw.get("max_tokens", 4096))
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    tools.register(SearchTool())
    memory = Memory(storage_path=kw.get("storage_path"))
    observer = Observer()
    constraints = ConstraintManager(
        max_steps=kw.get("max_steps", 20),
        max_tokens=kw.get("max_tokens", 100000),
    )
    return Orchestrator(llm, context, tools, memory, observer, constraints, verbose=False)


# ======================================================================
# 1. LLM 接口层
# ======================================================================

class TestLLM(unittest.TestCase):
    def test_scripted_flow(self):
        llm = MockLLM(default_script())
        r1 = llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(r1.finish_reason, "tool_calls")
        self.assertEqual(r1.tool_calls[0].name, "calculator")
        self.assertEqual(r1.tool_calls[0].arguments["expression"], "3.14159 * 5 ** 2")

        r2 = llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(r2.tool_calls[0].name, "web_search")

        r3 = llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(r3.finish_reason, "stop")
        self.assertIsNone(r3.tool_calls)
        self.assertIn("78.54", r3.content)

    def test_out_of_script_returns_stop(self):
        llm = MockLLM([{"content": "only one"}])
        llm.chat([])
        r = llm.chat([])
        self.assertEqual(r.finish_reason, "stop")
        self.assertEqual(r.content, "")

    def test_reset_and_step(self):
        llm = MockLLM(default_script())
        llm.chat([])
        llm.chat([])
        self.assertEqual(llm.step, 2)
        llm.reset()
        self.assertEqual(llm.step, 0)

    def test_usage_estimation(self):
        llm = MockLLM(default_script())
        r = llm.chat([{"role": "user", "content": "x" * 400}])
        self.assertGreater(r.usage.prompt_tokens, 50)
        self.assertGreater(r.usage.completion_tokens, 0)


# ======================================================================
# 2. 上下文管理
# ======================================================================

class TestContext(unittest.TestCase):
    def test_message_roles(self):
        cm = ContextManager(system_prompt="sys")
        cm.add_user_message("任务")
        cm.add_assistant_message("思考", tool_calls=[ToolCall("c1", "calculator", {"expression": "1+1"})])
        cm.add_tool_message("c1", "calculator", "Result: 2")

        msgs = cm.get_messages()
        self.assertEqual(len(msgs), 4)  # system + user + assistant + tool
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "calculator")
        self.assertEqual(msgs[3]["role"], "tool")
        self.assertEqual(msgs[3]["tool_call_id"], "c1")

    def test_token_count_grows(self):
        cm = ContextManager(system_prompt="s")
        before = cm.get_token_count()
        cm.add_user_message("hello world " * 100)
        self.assertGreater(cm.get_token_count(), before)

    def test_summarize(self):
        cm = ContextManager(system_prompt="s", max_tokens=10)  # 极小窗口
        for i in range(10):
            cm.add_user_message(f"message number {i} " + "x" * 50)
        self.assertTrue(cm.should_summarize())
        n_before = len(cm.messages)
        cm.summarize()
        self.assertLessEqual(len(cm.messages), 5)
        self.assertGreater(cm.summarized_count, 0)
        self.assertEqual(cm.messages[0].role, "system")
        self.assertIn("summary", cm.messages[0].content)

    def test_round_trip_serialization(self):
        cm = ContextManager(system_prompt="sys", max_tokens=128)
        cm.add_user_message("u")
        cm.add_assistant_message("a", tool_calls=[ToolCall("c1", "web_search", {"query": "q"})])
        cm.add_tool_message("c1", "web_search", "result")

        data = cm.to_dict()
        cm2 = ContextManager.from_dict(data)
        self.assertEqual(cm2.system_prompt, "sys")
        self.assertEqual(cm2.max_tokens, 128)
        self.assertEqual(len(cm2.messages), 3)
        tc = cm2.messages[1].tool_calls
        self.assertEqual(tc[0].name, "web_search")
        self.assertEqual(tc[0].arguments, {"query": "q"})


# ======================================================================
# 3. 工具系统
# ======================================================================

class TestTools(unittest.TestCase):
    def test_calculator(self):
        tool = CalculatorTool()
        r = tool.execute(expression="3.14159 * 5 ** 2")
        self.assertTrue(r.success)
        self.assertIn("78.53975", r.output)

    def test_calculator_rejects_injection(self):
        tool = CalculatorTool()
        r = tool.execute(expression="__import__('os')")
        self.assertFalse(r.success)
        self.assertIsNotNone(r.error)

    def test_calculator_missing_param(self):
        r = CalculatorTool().execute()
        self.assertFalse(r.success)
        self.assertIn("Missing", r.error)

    def test_search(self):
        r = SearchTool().execute(query="circle area formula")
        self.assertTrue(r.success)
        self.assertIn("π", r.output)

    def test_search_no_results(self):
        r = SearchTool().execute(query="zzz nonexistent zzz")
        self.assertTrue(r.success)
        self.assertIn("No results", r.output)

    def test_file_read(self):
        r = FileReadTool().execute(path="/definitely/not/exist/file.txt")
        self.assertFalse(r.success)

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        with self.assertRaises(ValueError):
            reg.register(CalculatorTool())  # 重名

        self.assertIsNone(reg.get("nope"))
        r = reg.execute("nope", {})
        self.assertFalse(r.success)
        self.assertIn("Unknown tool", r.error)

        r = reg.execute("calculator", {"expression": "2+2"})
        self.assertTrue(r.success)

        tools = reg.list_tools()
        self.assertEqual(tools[0]["function"]["name"], "calculator")
        self.assertIn("parameters", tools[0]["function"])


# ======================================================================
# 4. 记忆与状态
# ======================================================================

class TestMemory(unittest.TestCase):
    def test_short_term(self):
        m = Memory()
        m.set("k", "v")
        self.assertEqual(m.get("k"), "v")
        self.assertIsNone(m.get("missing"))
        m.delete("k")
        self.assertIsNone(m.get("k"))

    def test_long_term_persistence(self):
        path = os.path.join(tempfile.gettempdir(), "harness_test_memory.json")
        if os.path.exists(path):
            os.remove(path)
        m1 = Memory(storage_path=path)
        m1.remember("topic", "geometry")
        m1.remember("non-serializable", object())  # 降级为 str

        m2 = Memory(storage_path=path)  # 新实例从磁盘加载
        self.assertEqual(m2.recall("topic"), "geometry")
        self.assertIn("object at", m2.recall("non-serializable"))

    def test_state_lifecycle(self):
        m = Memory()
        self.assertEqual(m.state.status, "pending")
        m.start_task("do something")
        self.assertEqual(m.state.status, "running")
        self.assertEqual(m.state.task, "do something")
        m.finish_task("done")
        self.assertEqual(m.state.status, "completed")
        self.assertEqual(m.state.result, "done")
        self.assertIsNotNone(m.state.finished_at)

    def test_snapshot_restore(self):
        m = Memory()
        m.start_task("t")
        m.set("progress", 42)
        snap = m.snapshot()

        m2 = Memory()
        m2.restore(snap)
        self.assertEqual(m2.get("progress"), 42)
        self.assertEqual(m2.state.task, "t")
        self.assertEqual(m2.state.status, "running")


# ======================================================================
# 5. 评估观测
# ======================================================================

class TestObserver(unittest.TestCase):
    def test_trace_and_metrics(self):
        ob = Observer()
        ob.start()
        ob.log("think", step=1, content="...")
        ob.log("act", step=1, tool="calculator", success=True)
        ob.log("act", step=2, tool="calculator", success=False)
        ob.stop()

        m = ob.get_metrics()
        self.assertEqual(m["llm_calls"], 1)
        self.assertEqual(m["tool_calls"], 2)
        self.assertEqual(m["tool_errors"], 1)
        self.assertEqual(m["total_steps"], 2)

        trace = ob.get_trace()
        self.assertEqual(len(trace), 3)
        self.assertIsInstance(trace[0], TraceEvent)

    def test_hooks_called_and_exceptions_swallowed(self):
        ob = Observer()
        calls = []

        def good_hook(e):
            calls.append(e.event_type)

        def bad_hook(e):
            raise RuntimeError("boom")

        ob.add_hook(good_hook)
        ob.add_hook(bad_hook)
        ob.log("think", step=1)  # bad_hook 异常被吞掉
        self.assertEqual(calls, ["think"])

    def test_summary_and_export(self):
        ob = Observer()
        ob.start()
        ob.log("think", step=1)
        ob.log("act", step=1, tool="t", success=True)
        ob.stop()

        s = ob.summary()
        self.assertIn("运行摘要", s)
        self.assertIn("LLM 调用数: 1", s)

        exported = ob.export()
        self.assertEqual(len(exported["trace"]), 2)
        self.assertIn("metrics", exported)

        t = ob.format_trace()
        self.assertIn("think", t)

    def test_reset(self):
        ob = Observer()
        ob.log("think", step=1)
        ob.reset()
        self.assertEqual(ob.get_trace(), [])
        self.assertEqual(ob.get_metrics()["llm_calls"], 0)


# ======================================================================
# 6. 约束与恢复
# ======================================================================

class TestConstraints(unittest.TestCase):
    def test_step_limit(self):
        cm = ConstraintManager(max_steps=5)
        self.assertTrue(cm.check_step(5))
        self.assertFalse(cm.check_step(6))
        self.assertTrue(cm.should_abort)

    def test_token_limit(self):
        cm = ConstraintManager(max_tokens=100)
        self.assertTrue(cm.check_tokens(100))
        self.assertFalse(cm.check_tokens(101))

    def test_tool_error_budget(self):
        cm = ConstraintManager(max_tool_errors=2)
        self.assertTrue(cm.record_tool_error())
        self.assertTrue(cm.record_tool_error())
        self.assertFalse(cm.record_tool_error())  # 第 3 次超限
        self.assertTrue(cm.should_abort)

    def test_timeout(self):
        cm = ConstraintManager(timeout=10.0)
        self.assertTrue(cm.check_timeout(5.0))
        self.assertFalse(cm.check_timeout(11.0))

    def test_check_all_aggregates(self):
        cm = ConstraintManager(max_steps=1, max_tokens=100000, timeout=999)
        ok, violations = cm.check_all(step=5, tokens=1, elapsed=0.1)
        self.assertFalse(ok)
        self.assertTrue(any("steps" in v for v in violations))

    def test_custom_constraint_warning_not_abort(self):
        cm = ConstraintManager()
        cm.add_constraint(Constraint("soft", "soft limit", lambda: True, severity="warning"))
        ok, _ = cm.check_all(1, 0, 0)
        self.assertFalse(ok)
        self.assertFalse(cm.should_abort)  # warning 不熔断

    def test_recovery_retry(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("transient")
            return "ok"

        self.assertEqual(Recovery.retry(flaky, max_retries=5, delay=0.001), "ok")
        self.assertEqual(len(attempts), 3)

    def test_recovery_retry_exhausted(self):
        def always_fail():
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            Recovery.retry(always_fail, max_retries=2, delay=0.001)

    def test_recovery_fallback(self):
        result = Recovery.with_fallback(
            lambda: 1 / 0,          # 主策略失败
            lambda: "fallback",     # 降级策略
        )
        self.assertEqual(result, "fallback")

    def test_recovery_safe_execute(self):
        self.assertEqual(Recovery.safe_execute(lambda: 1 / 0, default="safe"), "safe")
        self.assertEqual(Recovery.safe_execute(lambda: 42, default="safe"), 42)


# ======================================================================
# 7. 执行编排（集成）
# ======================================================================

class TestOrchestrator(unittest.TestCase):
    def test_standard_run_completes(self):
        orch = make_orchestrator()
        result = orch.run("计算圆面积")

        self.assertIn("78.54", result)
        self.assertEqual(orch.memory.state.status, "completed")
        self.assertEqual(orch.memory.state.step_count, 3)

        # 观测指标
        m = orch.observer.get_metrics()
        self.assertEqual(m["llm_calls"], 3)
        self.assertEqual(m["tool_calls"], 2)
        self.assertEqual(m["tool_errors"], 0)
        self.assertGreater(m["prompt_tokens"], 0)

        # 上下文: user + 3 assistant + 2 tool = 6 条
        self.assertEqual(len(orch.context.messages), 6)

        # 短期记忆存了工具结果
        self.assertIn("step1_calculator", orch.memory.keys())
        self.assertIn("step2_web_search", orch.memory.keys())

    def test_abort_on_max_steps(self):
        infinite = [{
            "content": "continue",
            "tool_calls": [{"id": f"c{i}", "name": "calculator",
                            "arguments": {"expression": f"{i}+1"}}],
            "finish_reason": "tool_calls",
        } for i in range(1, 50)]
        orch = make_orchestrator(script=infinite, max_steps=3)
        result = orch.run("runaway task")

        self.assertIn("aborted", result)
        self.assertEqual(orch.memory.state.status, "aborted")
        self.assertEqual(orch.memory.state.step_count, 4)  # 第 4 步被熔断
        self.assertTrue(orch.constraints.get_violations())

    def test_tool_error_flow(self):
        bad_script = [
            {"content": "try bad",
             "tool_calls": [{"id": "c1", "name": "calculator",
                             "arguments": {"expression": "import os"}}],
             "finish_reason": "tool_calls"},
            {"content": "all done", "finish_reason": "stop"},
        ]
        orch = make_orchestrator(script=bad_script)
        result = orch.run("bad expr")

        self.assertEqual(orch.memory.state.status, "completed")  # 错误后仍完成
        self.assertEqual(orch.observer.get_metrics()["tool_errors"], 1)
        # 工具错误以 tool 消息形式写回上下文
        tool_msgs = [m for m in orch.context.messages if m.role == "tool"]
        self.assertIn("Error", tool_msgs[0].content)

    def test_context_compression_during_run(self):
        long_script = []
        for i in range(1, 15):
            long_script.append({
                "content": f"round {i} " + "thinking very hard about this problem " * 10,
                "tool_calls": [{"id": f"c{i}", "name": "calculator",
                                "arguments": {"expression": f"{i}*2"}}],
                "finish_reason": "tool_calls",
            })
        long_script.append({"content": "final answer here", "finish_reason": "stop"})

        orch = make_orchestrator(script=long_script, max_tokens=400)
        result = orch.run("long task")
        self.assertIn("final answer", result)
        self.assertGreater(orch.context.summarized_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
