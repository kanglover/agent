"""
09 - Middleware（中间件）：在 Agent 运行的每一步插钩子
=====================================================
【这章学什么】
  - LangChain 1.x 的中间件机制（AgentMiddleware）
  - 三种"插钩子"方式的区别：callbacks / listeners / middleware
  - 用中间件做"观察"（记录日志、计时）和"改变行为"（重试）

【为什么学它】
  前面学的链/Agent 都是"一头走到尾"。但实际项目你常常想在中间插手：
  - 想知道 agent 每一步在干嘛 → 加日志
  - 想统计模型调一次花多久 → 加计时
  - 模型偶尔抽风报错，想自动重试 → 加重试
  - 想在调模型前动态改提示词 → 加改写

  中间件就是干这些的——在 agent 循环的每一步"前后"挂钩子，既能"看"也能"改"。

【类比】
  中间件就像快递分拣线上的"检查员"：
  - 有的检查员只负责记录（"这个包裹经过了"）—— 观察型
  - 有的检查员能动手（"这个包裹有问题，退回去重发"）—— 改行为型
  callbacks/listeners 是"只看不摸"的检查员；middleware 是"能看能摸"的。

【关键：middleware 配 create_agent 用】
  middleware 是 langchain.agents 的机制，配套 create_agent（第05章工具调用的"升级版快捷方式"）。
  create_agent 帮你把"模型+工具+循环"打包成一个 agent，middleware 就挂在它身上。

【运行】
  cd code
  uv run python learn_langchain/09_middleware.py
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)

# create_agent：把模型(可带工具)打包成一个可运行的 agent
# 它是第05章"手写工具调用循环"的官方快捷方式
from langchain.agents import create_agent


# ============================================================
# 1. 先看 callbacks：最底层的"只读观察"
# ============================================================
# BaseCallbackHandler 能在 LLM/工具/链的各种事件上挂钩子，但只能"看"不能"改"。
# 适合做日志、监控、token 统计。
print("=" * 60)
print("1. callbacks：只读观察（记录模型开始/结束）")
print("=" * 60)

from langchain_core.callbacks import BaseCallbackHandler


class MyLogger(BaseCallbackHandler):
    """自定义回调：在模型开始和结束时打印。"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("  [callback] 模型开始生成...")

    def on_llm_end(self, response, **kwargs):
        print("  [callback] 模型生成结束")


# 通过 callbacks 参数传入
agent_with_cb = create_agent(model=llm)
agent_with_cb.invoke(
    {"messages": [{"role": "user", "content": "说一个字"}]},
    config={"callbacks": [MyLogger()]},
)
print("→ callbacks 只能观察，不能改输入输出\n")


# ============================================================
# 2. with_listeners：给链挂"开始/结束"监听（同样只读）
# ============================================================
# 比 callbacks 简单：三个函数 on_start/on_end/on_error，针对"整个 Runnable"。
print("=" * 60)
print("2. with_listeners：链级别的开始/结束监听")
print("=" * 60)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "用一句话解释 {x}")])
    | llm
    | StrOutputParser()
).with_listeners(
    on_start=lambda run: print("  [listener] 链开始"),
    on_end=lambda run: print(f"  [listener] 链结束"),
)

print(chain.invoke({"x": "中间件"}))
print("→ listeners 也只能观察\n")


# ============================================================
# 3. AgentMiddleware 装饰器：在 agent 每步插钩子（重点）
# ============================================================
# 真正的"中间件"。用装饰器把普通函数变成中间件：
#   @before_model  → 每次调模型【前】执行
#   @after_model   → 每次调模型【后】执行
#   @before_agent  → 整个 agent 开始前执行一次
#   @after_agent   → 整个 agent 结束后执行一次
# 函数能直接拿到 state（当前状态），比 callbacks 方便。
print("=" * 60)
print("3. AgentMiddleware 装饰器：观察 agent 每一步")
print("=" * 60)

from langchain.agents.middleware import before_model, after_model


@before_model
def log_before_model(state, runtime):
    # state 是当前状态字典，state["messages"] 是对话历史
    print(f"  [before_model] 即将调模型，当前 {len(state['messages'])} 条消息")


@after_model
def log_after_model(state, runtime):
    print(f"  [after_model] 模型回复完成，现在 {len(state['messages'])} 条消息")


agent = create_agent(
    model=llm,
    middleware=[log_before_model, log_after_model],  # ← 中间件列表
)

result = agent.invoke({"messages": [{"role": "user", "content": "一句话介绍你自己"}]})
print("回复：", result["messages"][-1].content[:50])
print()


# ============================================================
# 4. wrap_model_call：能"改行为"的中间件（计时 + 重试）
# ============================================================
# wrap_model_call 包住整个模型调用：你调 handler(request) 才真正执行模型。
# 在 handler 前后可以加逻辑，甚至捕获异常重试。这是 middleware 比 callbacks 强的地方。
print("=" * 60)
print("4. wrap_model_call 类风格：计时 + 重试（能改行为）")
print("=" * 60)

from langchain.agents.middleware import AgentMiddleware


class TimingAndRetryMiddleware(AgentMiddleware):
    """计时 + 失败重试的中间件。"""

    def wrap_model_call(self, request, handler):
        # handler(request) = 真正调模型。你不调它，模型就不会执行
        for attempt in range(3):  # 最多重试 3 次
            t0 = time.time()
            try:
                resp = handler(request)
                print(f"  [wrap] 模型调用成功，耗时 {time.time() - t0:.2f}s（第{attempt+1}次）")
                return resp
            except Exception as e:
                print(f"  [wrap] 第{attempt+1}次失败：{e}")
                if attempt == 2:
                    raise  # 重试 3 次都失败，抛出去


agent2 = create_agent(model=llm, middleware=[TimingAndRetryMiddleware()])
agent2.invoke({"messages": [{"role": "user", "content": "说一个字"}]})
print()


# ============================================================
# 5. 内置中间件：开箱即用（零成本）
# ============================================================
# LangChain 自带一批现成中间件，不用自己写。最常用的是 ModelRetryMiddleware（自动重试）。
print("=" * 60)
print("5. 内置中间件 ModelRetryMiddleware：一行实现自动重试")
print("=" * 60)

from langchain.agents.middleware import ModelRetryMiddleware

# max_retries=3：失败最多重试 3 次。还有指数退避（backoff_factor）等参数
agent3 = create_agent(
    model=llm,
    middleware=[ModelRetryMiddleware(max_retries=3)],
)

agent3.invoke({"messages": [{"role": "user", "content": "说一个字"}]})
print("→ 不用自己写重试逻辑，一行搞定。还有 ModelFallbackMiddleware(降级) 等\n")


# ── 小结 ───────────────────────────────────────────────────
print("=" * 60)
print("小结：三种插钩子方式的区别")
print("=" * 60)
print("""
| 方式        | 能否改行为 | 作用对象        | 典型场景              |
|-------------|-----------|-----------------|----------------------|
| callbacks   | ❌ 只读    | 任意 Runnable   | 日志、监控、tracing   |
| listeners   | ❌ 只读    | 任意 Runnable   | 简单的开始/结束回调    |
| middleware  | ✅ 能改    | create_agent    | 重试、改prompt、拦截工具|

记住：
1. 只想"看"用 callbacks/listeners；想"改"用 middleware
2. middleware 配 create_agent 用，装饰器风格最简单
3. wrap_model_call 能包住模型调用，重试/计时/改写都在这
4. 内置 ModelRetryMiddleware 等开箱即用，优先用现成的
""")
print("🎉 learn_langchain 9 篇全部学完！你已掌握 LangChain 核心用法。")
print("   接下来推荐学 learn_langgraph/ —— 用图编排更复杂的工作流。")
