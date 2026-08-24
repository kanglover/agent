# agent_harness/main.py
"""
Agent Harness 运行入口

用法：
  cd code
  python -m agent_harness.main              # 默认：只跑模拟 Agent（不需要 API key）
  python -m agent_harness.main --with-llm   # 加上真实 LLM Agent（需要 ANTHROPIC_API_KEY）

这个文件做的事情：
  1. 创建 Harness（考场）
  2. 注册 Agent（考生）
  3. 加载任务（考卷）
  4. 运行测评（考试）
  5. 打印报告（成绩单）
"""
from __future__ import annotations

import sys


def main():
    # ── 导入框架组件 ──
    from agent_harness.harness import Harness
    from agent_harness.agents.simple_agent import SimpleAgent
    from agent_harness.agents.react_agent import MockLLMAgent

    # ── 解析命令行参数 ──
    use_llm = "--with-llm" in sys.argv

    # ── 第一步：创建考场 ──
    harness = Harness()

    # ── 第二步：注册考生 ──
    #   SimpleAgent:   规则匹配，不调 LLM（基线）
    #   MockLLMAgent:  模拟 LLM 行为（对照）
    #   ReactAgent:    真实调用 Claude API（需要 API key）
    harness.add_agent(SimpleAgent())
    harness.add_agent(MockLLMAgent())

    if use_llm:
        try:
            from agent_harness.agents.react_agent import ReactAgent
            harness.add_agent(ReactAgent())
            print("ℹ️  已加载 ReactAgent（真实 LLM），将消耗 API 额度")
        except Exception as e:
            print(f"⚠️  无法加载 ReactAgent: {e}")
            print("   继续使用模拟 Agent 运行...")

    # ── 第三步：加载考卷（使用内置默认任务套件）──
    harness.use_default_tasks()

    # ── 第四步：开考！──
    records = harness.run()

    # ── 第五步：打印成绩单 ──
    harness.print_report(records)

    # ── 返回结果（方便测试断言）──
    return records


if __name__ == "__main__":
    main()
