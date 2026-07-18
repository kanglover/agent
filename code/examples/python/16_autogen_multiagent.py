"""
AutoGen 多 Agent 系统完整示例
=====================================

AutoGen 是微软开源的多 Agent 框架，核心思想：
- 让多个 AI Agent 互相对话、分工协作
- 每个 Agent 有不同的角色和能力
- Agent 之间可以自动传递任务，直到问题解决

安装依赖：
    pip install pyautogen

运行本文件：
    python 16_autogen_multiagent.py

注意：需要配置 ANTHROPIC_API_KEY 环境变量（使用 Claude 作为 LLM 后端）
"""

import os
import sys
import tempfile

# ============================================================
# 第一步：检查依赖是否安装
# ============================================================

def check_and_install():
    """检查 pyautogen 是否已安装，未安装时给出提示"""
    try:
        import autogen
        print(f"[OK] pyautogen 已安装，版本：{autogen.__version__}")
        return True
    except ImportError:
        print("[ERROR] pyautogen 未安装")
        print("请先运行：pip install pyautogen")
        print("如果需要代码执行功能，还需要：pip install pyautogen[local-executor]")
        return False


# ============================================================
# 第二步：配置 Claude 作为 LLM 后端
# ============================================================
# AutoGen 原生支持 OpenAI 格式的 API，Claude 通过 OpenAI 兼容接口接入
# Anthropic 提供了兼容 OpenAI 格式的 API endpoint

def get_llm_config():
    """
    构建 LLM 配置字典

    AutoGen 的 llm_config 结构：
    {
        "config_list": [一个或多个模型配置],
        "temperature": 生成温度,
        "timeout": 超时秒数,
    }

    每个模型配置包含：
    - model: 模型名称
    - api_key: API 密钥
    - base_url: API 地址（使用非 OpenAI 服务时必填）
    - api_type: API 类型（可选）
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")

    llm_config = {
        "config_list": [
            {
                "model": "claude-opus-4-5",          # 使用的模型
                "api_key": api_key,                   # API 密钥
                "base_url": "https://api.anthropic.com/v1",  # Claude API 地址
                "api_type": "anthropic",              # 指定 API 类型
            }
        ],
        "temperature": 0.1,     # 低温度 = 更稳定、确定的输出（适合代码生成）
        "timeout": 120,         # 120 秒超时
        "cache_seed": None,     # 不缓存（每次都调用真实 API）
    }
    return llm_config


# ============================================================
# 第三步：最基础的双 Agent 对话
# ============================================================
# UserProxyAgent  = 代表用户的 Agent（负责发起任务、执行代码）
# AssistantAgent  = AI 助手 Agent（负责思考、生成方案、写代码）

def demo_basic_two_agents():
    """
    演示最基础的双 Agent 对话

    流程：
    1. UserProxy 发出任务
    2. Assistant 思考并给出回答
    3. UserProxy 可以执行代码，将结果反馈给 Assistant
    4. 循环直到任务完成（Assistant 说 TERMINATE）
    """
    print("\n" + "="*60)
    print("演示 1：基础双 Agent 对话")
    print("="*60)

    import autogen

    llm_config = get_llm_config()

    # --- 创建 AssistantAgent（AI 助手）---
    # AssistantAgent 是一个 LLM 驱动的 Agent
    # 它会：分析任务、制定计划、生成代码、给出解释
    assistant = autogen.AssistantAgent(
        name="助手",                    # Agent 的名字（在对话中显示）
        llm_config=llm_config,          # 使用哪个 LLM
        system_message="""你是一个专业的 Python 编程助手。
        当任务完成时，在回复末尾加上 TERMINATE 关键词。
        回答时请简洁清晰，用中文回复。""",
    )

    # --- 创建 UserProxyAgent（用户代理）---
    # UserProxyAgent 代表用户，它可以：
    # 1. 把用户的任务转发给 Assistant
    # 2. 自动执行 Assistant 生成的代码
    # 3. 把执行结果反馈回去
    user_proxy = autogen.UserProxyAgent(
        name="用户",
        human_input_mode="NEVER",       # NEVER = 完全自动，不询问人类
                                        # ALWAYS = 每轮都询问人类
                                        # TERMINATE = 只在结束时询问
        max_consecutive_auto_reply=3,   # 最多自动回复 3 次（防止死循环）
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config={
            "work_dir": tempfile.mkdtemp(),  # 代码执行的工作目录
            "use_docker": False,             # 不用 Docker（本地直接执行）
        },
    )

    # --- 发起对话 ---
    # initiate_chat 启动整个多轮对话
    user_proxy.initiate_chat(
        assistant,
        message="写一个 Python 函数，计算斐波那契数列的前 10 项，并打印结果。",
    )


# ============================================================
# 第四步：GroupChat（多 Agent 群聊）
# ============================================================
# GroupChat 允许 3 个以上的 Agent 参与同一个对话
# GroupChatManager 负责决定每轮由哪个 Agent 发言

def demo_group_chat():
    """
    演示 GroupChat 多 Agent 群聊

    场景：代码生成 → 代码审查 → 代码优化 的三 Agent 团队
    - 程序员 Agent：负责写代码
    - 审查员 Agent：负责找 bug 和问题
    - 优化师 Agent：负责提出优化建议
    - UserProxy：负责协调和执行代码
    """
    print("\n" + "="*60)
    print("演示 2：GroupChat 多 Agent 群聊")
    print("="*60)

    import autogen

    llm_config = get_llm_config()

    # --- 创建三个专职 Agent ---

    # Agent 1：程序员，负责写代码
    programmer = autogen.AssistantAgent(
        name="程序员",
        llm_config=llm_config,
        system_message="""你是一个 Python 程序员。
        专注于编写功能正确的代码。
        收到需求后，直接给出完整的 Python 代码实现。
        代码要包含注释，说明关键步骤。""",
    )

    # Agent 2：代码审查员，负责找问题
    reviewer = autogen.AssistantAgent(
        name="审查员",
        llm_config=llm_config,
        system_message="""你是一个严格的代码审查员。
        负责检查代码中的 bug、边界条件、错误处理等问题。
        如果代码有问题，指出具体问题并建议修复方案。
        如果代码没有问题，说"代码审查通过"。""",
    )

    # Agent 3：优化师，负责性能和风格
    optimizer = autogen.AssistantAgent(
        name="优化师",
        llm_config=llm_config,
        system_message="""你是一个代码优化专家。
        负责提出性能优化、代码风格改进的建议。
        给出优化后的代码版本，并解释改进之处。
        优化完成后，在回复末尾加 TERMINATE。""",
    )

    # --- 创建 UserProxyAgent（执行代码的那个）---
    user_proxy = autogen.UserProxyAgent(
        name="协调员",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config={
            "work_dir": tempfile.mkdtemp(),
            "use_docker": False,
        },
    )

    # --- 创建 GroupChat ---
    # GroupChat 把所有 Agent 聚集在一个"聊天室"里
    groupchat = autogen.GroupChat(
        agents=[user_proxy, programmer, reviewer, optimizer],  # 参与的所有 Agent
        messages=[],                # 消息历史（初始为空）
        max_round=8,                # 最多 8 轮对话（防止无限循环）
        speaker_selection_method="auto",  # 自动决定下一个发言者
                                          # 也可以是 "round_robin"（轮流发言）
    )

    # --- 创建 GroupChatManager（群聊主持人）---
    # GroupChatManager 是一个特殊 Agent，它的职责是：
    # 1. 根据对话内容决定下一个发言的 Agent 是谁
    # 2. 管理对话流程，确保任务推进
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,      # 主持人也需要 LLM 来做决策
    )

    # --- 发起群聊 ---
    user_proxy.initiate_chat(
        manager,
        message="""请团队合作完成以下任务：
        编写一个函数，接受一个字符串列表，返回其中最长的字符串。
        如果列表为空，返回 None。
        请先写代码，然后审查，最后优化。""",
    )


# ============================================================
# 第五步：自定义 Agent（子类化）
# ============================================================
# 通过继承 AssistantAgent 或 ConversableAgent，可以创建专属功能的 Agent

def demo_custom_agent():
    """
    演示自定义 Agent（通过子类化）

    场景：创建一个"调试专家 Agent"
    - 能自动分析错误信息
    - 能提供具体的修复建议
    - 有自定义的响应格式
    """
    print("\n" + "="*60)
    print("演示 3：自定义 Agent（子类化）")
    print("="*60)

    import autogen

    # --- 自定义 Agent 类 ---
    class DebugAgent(autogen.AssistantAgent):
        """
        调试专家 Agent

        在 AssistantAgent 的基础上，添加了：
        1. 专门针对调试的 system_message
        2. 自定义的 receive 方法，在收到消息时打印额外信息
        """

        def __init__(self, name: str, llm_config: dict):
            # 调用父类的 __init__，设置专门的 system_message
            super().__init__(
                name=name,
                llm_config=llm_config,
                system_message="""你是一个专业的调试专家。

                当收到有错误的代码时，你会：
                1. 【定位错误】：指出错误在哪一行，是什么类型的错误
                2. 【分析原因】：解释为什么会出现这个错误
                3. 【修复方案】：给出修复后的完整代码
                4. 【预防建议】：提出如何避免类似错误

                格式要清晰，用中文回答，最后加 TERMINATE。""",
            )
            # 自定义属性：记录调试次数
            self.debug_count = 0

        def receive(self, message, sender, request_reply=None, silent=False):
            """重写 receive 方法，在处理消息前打印统计信息"""
            self.debug_count += 1
            print(f"\n[调试专家收到第 {self.debug_count} 条消息，来自：{sender.name}]")
            # 调用父类的 receive，完成实际的 LLM 调用
            return super().receive(message, sender, request_reply, silent)

    llm_config = get_llm_config()

    # 创建自定义的调试专家 Agent
    debug_expert = DebugAgent(name="调试专家", llm_config=llm_config)

    # 创建 UserProxy
    user_proxy = autogen.UserProxyAgent(
        name="用户",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,   # 0 = 不自动回复，只执行代码
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config=False,    # False = 不执行代码
    )

    # 发送一段有 bug 的代码
    buggy_code = """
下面这段代码有 bug，请帮我调试：

```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # 问题：如果 numbers 是空列表会怎样？

result = calculate_average([])
print(f"平均值是：{result}")
```

运行后报错：ZeroDivisionError: division by zero
"""

    user_proxy.initiate_chat(
        debug_expert,
        message=buggy_code,
    )

    print(f"\n[统计] 调试专家共处理了 {debug_expert.debug_count} 条消息")


# ============================================================
# 第六步：代码生成 → 执行 → 调试 的完整 Agent 团队
# ============================================================

def demo_code_pipeline():
    """
    演示完整的代码生成流水线：代码生成 → 执行 → 调试

    这是一个更贴近实际的使用场景：
    1. 用户提出需求
    2. 代码生成 Agent 写出代码
    3. UserProxy 执行代码
    4. 如果执行失败，调试 Agent 介入修复
    5. 循环直到代码成功运行
    """
    print("\n" + "="*60)
    print("演示 4：代码生成 → 执行 → 调试 完整流水线")
    print("="*60)

    import autogen

    llm_config = get_llm_config()
    work_dir = tempfile.mkdtemp()

    # 代码生成专家
    code_writer = autogen.AssistantAgent(
        name="代码生成",
        llm_config=llm_config,
        system_message="""你是一个 Python 代码生成专家。
        - 根据需求生成完整、可运行的 Python 代码
        - 代码必须包含 if __name__ == '__main__' 入口
        - 遇到执行错误时，根据错误信息修复代码
        - 代码成功运行后，在回复末尾加 TERMINATE""",
    )

    # 执行器（UserProxy）：执行代码并反馈结果
    executor = autogen.UserProxyAgent(
        name="执行器",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,   # 最多尝试 5 次（生成→执行→修复循环）
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config={
            "work_dir": work_dir,
            "use_docker": False,
            "last_n_messages": 3,       # 只看最近 3 条消息中的代码
        },
    )

    # 发起任务
    executor.initiate_chat(
        code_writer,
        message="""请生成一个 Python 脚本，完成以下任务：
        1. 创建一个包含 5 个随机整数（1-100 之间）的列表
        2. 对列表排序
        3. 计算列表的平均值、最大值、最小值
        4. 把结果格式化打印出来

        要求：代码要有注释，输出要清晰易读。""",
    )


# ============================================================
# 主程序：运行所有演示
# ============================================================

def main():
    """主函数：按顺序运行各个演示"""

    print("\n" + "="*60)
    print("AutoGen 多 Agent 系统完整演示")
    print("="*60)

    # 1. 检查依赖
    if not check_and_install():
        sys.exit(1)

    # 2. 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[警告] 未设置 ANTHROPIC_API_KEY 环境变量")
        print("请先设置：export ANTHROPIC_API_KEY='your-api-key'")
        print("\n以下演示将以说明模式运行（不实际调用 API）...")
        run_explanations_only()
        return

    print(f"\n[OK] API Key 已设置（前缀：{api_key[:8]}...）")

    # 选择要运行的演示
    print("\n请选择要运行的演示：")
    print("1. 基础双 Agent 对话（UserProxy + Assistant）")
    print("2. GroupChat 多 Agent 群聊")
    print("3. 自定义 Agent（子类化）")
    print("4. 代码生成→执行→调试 完整流水线")
    print("0. 全部运行")

    choice = input("\n请输入选项（默认 1）：").strip() or "1"

    demos = {
        "1": demo_basic_two_agents,
        "2": demo_group_chat,
        "3": demo_custom_agent,
        "4": demo_code_pipeline,
    }

    if choice == "0":
        for demo_func in demos.values():
            demo_func()
    elif choice in demos:
        demos[choice]()
    else:
        print("无效选项，运行演示 1")
        demo_basic_two_agents()


def run_explanations_only():
    """未配置 API Key 时，仅打印概念说明"""

    explanation = """
AutoGen 核心概念速览
====================

1. ConversableAgent（对话 Agent 基类）
   - 所有 Agent 的父类
   - 能发送和接收消息
   - 能调用 LLM 生成回复

2. AssistantAgent（AI 助手 Agent）
   - 继承自 ConversableAgent
   - 内置了适合编程助手的 system_message
   - 主要职责：思考、规划、生成代码

3. UserProxyAgent（用户代理 Agent）
   - 继承自 ConversableAgent
   - 代表用户参与对话
   - 核心能力：执行代码（code_execution_config）
   - human_input_mode 控制是否需要人工介入

4. GroupChat（群聊）
   - 容纳多个 Agent 的"聊天室"
   - 管理消息历史
   - 决定发言顺序（speaker_selection_method）

5. GroupChatManager（群聊主持人）
   - 特殊的 Agent，负责调度群聊
   - 用 LLM 智能决定下一个发言者
   - 确保对话朝目标方向推进

6. 代码执行流程：
   用户发任务 → Assistant 写代码 → UserProxy 执行代码
   → 执行结果反馈给 Assistant → Assistant 修复或确认完成

7. 自定义 Agent：
   - 继承 AssistantAgent 或 ConversableAgent
   - 重写 __init__ 设置专属 system_message
   - 重写 receive/send/generate_reply 添加自定义逻辑

配置 API Key 后，运行本文件即可看到实际演示。
"""
    print(explanation)


if __name__ == "__main__":
    main()
