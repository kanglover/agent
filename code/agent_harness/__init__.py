# agent_harness — Agent 测评框架
#
# 用「考试」类比理解：
#   - protocol.py   = 考生须知（Agent 必须实现的接口）
#   - tasks.py       = 考卷（结构化的测试任务）
#   - evaluator.py   = 评分标准（判定完成度和质量）
#   - harness.py     = 考场总控（调度运行、收集结果、生成报告）
#   - agents/        = 考生们（不同策略的 Agent 实现）
#   - tools.py       = 考场提供的工具（计算器、字典等）
