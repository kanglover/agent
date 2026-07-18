"""
09_context_engineering.py
上下文工程实践 (Context Engineering)

核心思想：
  上下文工程不是"塞更多内容进去"，而是"在正确时机提供正确信息"。
  好的上下文设计直接决定 AI 回复质量、延迟和成本。

涵盖内容：
  1. 动态系统提示构建（按任务类型组装）
  2. Prompt Caching（cache_control 标记，ephemeral vs persistent）
  3. 上下文压缩算法（AI 摘要 vs 关键点提取）
  4. 信息层次化（pinned 信息 vs 动态信息）
  5. Token 预算动态分配
  6. 相同任务不同上下文的对比实验
  7. 工具定义缓存示例
"""

import os
import time
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# 辅助：打印分隔线
# ─────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════
# 1. 动态系统提示构建
#    根据任务类型，像"乐高积木"一样拼装系统提示，
#    避免把所有规则都塞进一条固定提示里。
# ══════════════════════════════════════════════════════════════
section("1. 动态系统提示构建")

# 基础模块：任何任务都带上的"底座"
BASE_SYSTEM = "你是一位专业助手，回答简洁、准确。"

# 可按需拼接的功能模块
MODULES = {
    "code":     "当涉及代码时，必须给出可运行的示例，并附上注释。",
    "safety":   "如果话题涉及安全或法律风险，先给出免责提示。",
    "chinese":  "所有回复必须使用中文，即使用户用英文提问。",
    "concise":  "每条回复不超过 100 字，追求极度精炼。",
}

def build_system_prompt(task_type: str) -> str:
    """
    根据任务类型动态组装系统提示。

    task_type 可选值: "coding", "legal", "translation", "general"
    """
    parts = [BASE_SYSTEM]

    if task_type == "coding":
        parts.append(MODULES["code"])
        parts.append(MODULES["chinese"])
    elif task_type == "legal":
        parts.append(MODULES["safety"])
        parts.append(MODULES["concise"])
    elif task_type == "translation":
        parts.append(MODULES["chinese"])
        parts.append(MODULES["concise"])
    else:
        parts.append(MODULES["concise"])

    prompt = " ".join(parts)
    print(f"[任务类型: {task_type}]")
    print(f"拼装结果 ({len(prompt)} 字符): {prompt[:80]}...")
    return prompt

# 演示：同一问题，不同任务类型产生不同系统提示
for t in ["coding", "legal", "general"]:
    build_system_prompt(t)


# ══════════════════════════════════════════════════════════════
# 2. Prompt Caching
#    原理：Anthropic 服务器可以"记住"你上次发的内容，
#          下次请求相同前缀时直接复用，速度更快、成本更低。
#
#    cache_control 两种类型：
#      - ephemeral：默认缓存 5 分钟（适合会话内复用）
#      - （未来）persistent：更长时间缓存
#
#    关键规则：cache_control 必须加在"足够长"的 block 末尾，
#              且相同前缀在同一会话内自动命中。
# ══════════════════════════════════════════════════════════════
section("2. Prompt Caching 演示")

# 模拟一份"大型知识库"——在真实场景中可能是几千行文档
LARGE_KNOWLEDGE_BASE = """
知识库 v1.0（模拟大文档）：

第1章：上下文工程基础
  上下文（Context）是 AI 在生成回复时"能看到"的全部信息。
  合理管理上下文是控制质量、成本、延迟的核心手段。

第2章：Token 与成本
  输入 token 越多，API 费用越高、延迟越大。
  Prompt Caching 可将重复前缀的输入成本降低约 90%。

第3章：信息新鲜度
  并非所有信息都需要每次传入：
  - 静态规则、角色设定 → 适合缓存
  - 用户最新消息、实时数据 → 不应缓存

【更多章节省略，真实使用时这里可能有上万 token】
""" * 3  # 重复3次模拟较长文档

def demo_prompt_caching():
    """
    演示如何在 system prompt 中打上 cache_control 标记。
    第一次请求会写入缓存，后续请求命中缓存，速度明显提升。
    """
    # 带缓存标记的系统提示：使用 list 格式（多 block）
    system_with_cache = [
        {
            "type": "text",
            "text": LARGE_KNOWLEDGE_BASE,
            # cache_control 打在"想被缓存"的内容末尾
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            # 动态部分不加 cache_control，每次都新鲜传入
            "text": "今天是 2026-07-03，请基于以上知识库回答用户问题。"
        }
    ]

    question = "上下文工程的核心是什么？请用一句话概括。"

    print("第1次请求（写入缓存）...")
    t0 = time.time()
    resp1 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system=system_with_cache,
        messages=[{"role": "user", "content": question}]
    )
    t1 = time.time()
    print(f"  耗时: {t1-t0:.2f}s | 输入tokens: {resp1.usage.input_tokens}")
    # cache_creation_input_tokens 表示本次写入了多少 token 进缓存
    if hasattr(resp1.usage, 'cache_creation_input_tokens'):
        print(f"  缓存写入: {resp1.usage.cache_creation_input_tokens} tokens")
    print(f"  回复: {resp1.content[0].text[:80]}")

    print("\n第2次请求（命中缓存）...")
    t2 = time.time()
    resp2 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system=system_with_cache,
        messages=[{"role": "user", "content": "Token 和成本有什么关系？"}]
    )
    t3 = time.time()
    print(f"  耗时: {t3-t2:.2f}s | 输入tokens: {resp2.usage.input_tokens}")
    # cache_read_input_tokens 表示本次从缓存读了多少 token
    if hasattr(resp2.usage, 'cache_read_input_tokens'):
        print(f"  缓存命中: {resp2.usage.cache_read_input_tokens} tokens（这部分费用约为正常的 10%）")
    print(f"  回复: {resp2.content[0].text[:80]}")

demo_prompt_caching()


# ══════════════════════════════════════════════════════════════
# 3. 上下文压缩算法
#    问题：多轮对话越聊越长，token 消耗越来越多。
#    两种压缩策略对比：
#      A. AI 摘要：让模型把历史对话概括成一段摘要
#      B. 关键点提取：用规则/模型抽取结构化要点
# ══════════════════════════════════════════════════════════════
section("3. 上下文压缩算法对比")

# 模拟一段较长的历史对话
LONG_HISTORY = [
    {"role": "user",      "content": "我想学 Python，从哪开始？"},
    {"role": "assistant", "content": "建议从变量、条件语句、循环开始，用官方文档或《Python Crash Course》。"},
    {"role": "user",      "content": "函数怎么定义？"},
    {"role": "assistant", "content": "用 def 关键字：def greet(name): return f'Hello {name}'"},
    {"role": "user",      "content": "列表和字典有什么区别？"},
    {"role": "assistant", "content": "列表是有序序列（[1,2,3]），字典是键值对（{'a':1}），各有适用场景。"},
    {"role": "user",      "content": "我现在在学面向对象，class 怎么用？"},
    {"role": "assistant", "content": "class Dog: def __init__(self, name): self.name = name"},
]

def compress_by_ai_summary(history: list) -> str:
    """
    策略 A：让 Claude 把对话历史压缩成一段摘要。
    优点：语义保真度高，自然语言易读。
    缺点：需要额外一次 API 调用，有信息损失风险。
    """
    history_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
        for m in history
    )
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"请将以下对话历史压缩为一段不超过80字的摘要，"
                f"保留关键信息点：\n\n{history_text}"
            )
        }]
    )
    return resp.content[0].text

def compress_by_key_points(history: list) -> list[str]:
    """
    策略 B：基于规则的关键点提取（无需额外 API 调用）。
    优点：零延迟、零成本、可控性强。
    缺点：规则写死，对复杂对话效果有限。
    """
    key_points = []
    for msg in history:
        if msg["role"] == "assistant":
            content = msg["content"]
            # 简单规则：取助手回复的前40字作为要点
            point = content[:40].rstrip("，。") + "..."
            key_points.append(point)
    return key_points

original_tokens_estimate = sum(len(m["content"]) for m in LONG_HISTORY)
print(f"原始历史长度（字符数估算）: {original_tokens_estimate}")

summary = compress_by_ai_summary(LONG_HISTORY)
print(f"\n[策略A - AI摘要] ({len(summary)} 字符):")
print(f"  {summary}")

key_points = compress_by_key_points(LONG_HISTORY)
kp_text = " | ".join(key_points)
print(f"\n[策略B - 关键点提取] ({len(kp_text)} 字符):")
for i, kp in enumerate(key_points, 1):
    print(f"  {i}. {kp}")

print(f"\n压缩率对比:")
print(f"  AI摘要: {len(summary)/original_tokens_estimate*100:.1f}% 原始大小")
print(f"  关键点: {len(kp_text)/original_tokens_estimate*100:.1f}% 原始大小")


# ══════════════════════════════════════════════════════════════
# 4. 信息层次化
#    核心思想：不同信息的"生命周期"不同，
#    应分层管理，避免每次都全量传入。
#
#    层次划分：
#      - Pinned（锚定）：角色设定、核心规则，永不变
#      - Session（会话）：当前会话的背景，对话内有效
#      - Dynamic（动态）：当前消息的实时上下文
# ══════════════════════════════════════════════════════════════
section("4. 信息层次化（Pinned vs Dynamic）")

# Pinned 层：每次请求都带，适合缓存
PINNED_INFO = """[系统角色] 你是 Python 编程导师，专注帮助初学者。
[核心规则] 1) 示例必须可运行 2) 解释避免行话 3) 中文回复"""

# Session 层：本次会话背景
SESSION_INFO = """[用户背景] 零基础，刚学 Python 2 周，目标是数据分析。"""

# Dynamic 层：本轮消息的即时上下文
def build_context_layered(user_question: str, recent_topic: str = "") -> list:
    """
    构建层次化上下文，返回 messages 列表。
    Pinned + Session 合并进 system，Dynamic 放在 user 消息前。
    """
    system = f"{PINNED_INFO}\n\n{SESSION_INFO}"

    # 动态注入当前话题（如有）
    user_content = user_question
    if recent_topic:
        user_content = f"[当前话题: {recent_topic}]\n{user_question}"

    return {
        "system": system,
        "messages": [{"role": "user", "content": user_content}]
    }

ctx = build_context_layered("for 循环怎么用？", recent_topic="Python基础语法")
print(f"System ({len(ctx['system'])} 字符):\n  {ctx['system'][:100]}...")
print(f"\nUser 消息:\n  {ctx['messages'][0]['content']}")
print(f"\n层次化优势：Pinned 部分可缓存，Dynamic 部分每次刷新，互不干扰。")


# ══════════════════════════════════════════════════════════════
# 5. Token 预算动态分配
#    思路：根据任务复杂度、历史消耗，动态调整 max_tokens，
#    既不浪费，也不因 token 不足截断回复。
# ══════════════════════════════════════════════════════════════
section("5. Token 预算动态分配")

def estimate_complexity(question: str) -> str:
    """
    简单规则判断问题复杂度。
    实际场景可用分类模型或关键词匹配。
    """
    complex_keywords = ["原理", "架构", "对比", "详细", "完整", "源码", "实现"]
    simple_keywords = ["什么是", "怎么拼", "简单", "一句话"]

    if any(kw in question for kw in complex_keywords):
        return "high"
    if any(kw in question for kw in simple_keywords):
        return "low"
    return "medium"

def get_token_budget(complexity: str, remaining_budget: int = 4096) -> int:
    """
    根据复杂度和剩余预算分配 max_tokens。

    Args:
        complexity: "low" / "medium" / "high"
        remaining_budget: 本次会话剩余可用 token 总量
    """
    base = {"low": 100, "medium": 300, "high": 800}[complexity]
    # 不超过剩余预算的 50%，保留余量给后续对话
    allocated = min(base, remaining_budget // 2)
    return max(allocated, 50)  # 最少保留 50 token

questions = [
    "什么是变量？",
    "for 循环如何使用？",
    "详细解释 Python 的 GIL 原理和对多线程的影响",
]

total_budget = 2000  # 模拟本次会话总预算
for q in questions:
    complexity = estimate_complexity(q)
    budget = get_token_budget(complexity, total_budget)
    total_budget -= budget  # 扣除本次分配
    print(f"问题: {q[:30]:<30} | 复杂度: {complexity:<6} | 分配: {budget:>4} tokens | 剩余: {total_budget}")


# ══════════════════════════════════════════════════════════════
# 6. 相同任务不同上下文的对比实验
#    实证：上下文质量直接影响回复质量。
#    对比"裸问"vs"有背景"的差异。
# ══════════════════════════════════════════════════════════════
section("6. 上下文质量对比实验")

QUESTION = "我应该用 list 还是 dict？"

def call_with_context(system: str, label: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=120,
        system=system,
        messages=[{"role": "user", "content": QUESTION}]
    )
    text = resp.content[0].text
    print(f"\n[{label}]")
    print(f"  问题: {QUESTION}")
    print(f"  回复: {text[:150]}")
    return text

# 对比 A：极简上下文（裸问）
call_with_context(
    system="你是助手。",
    label="A - 极简上下文"
)

# 对比 B：丰富上下文（有用户背景 + 具体场景）
call_with_context(
    system=(
        "你是 Python 导师，学生是零基础初学者，刚学完变量和循环。"
        "学生正在做一个「记录每天步数」的小项目，需要存储 7 天的数据。"
        "用中文回答，给出具体代码示例。"
    ),
    label="B - 丰富上下文（含用户背景+场景）"
)

print("\n结论：上下文越贴近真实场景，AI 回复越具体、越有用。")


# ══════════════════════════════════════════════════════════════
# 7. 工具定义缓存示例
#    工具定义（Tool definitions）往往很长，每次请求都重复传入。
#    通过 cache_control 缓存工具定义，可大幅降低成本。
# ══════════════════════════════════════════════════════════════
section("7. 工具定义缓存")

# 模拟一套复杂的工具集（实际场景可能有十几个工具）
TOOLS_WITH_CACHE = [
    {
        "name": "search_docs",
        "description": "在文档库中搜索相关内容，支持关键词和语义搜索。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "搜索关键词"},
                "top_k":   {"type": "integer", "description": "返回结果数量，默认5"},
                "filters": {"type": "object",  "description": "过滤条件，如 {'lang': 'zh'}"}
            },
            "required": ["query"]
        },
        # 在最后一个工具上加缓存标记，Claude API 会缓存截止到此处的所有工具定义
        "cache_control": {"type": "ephemeral"}
    },
    {
        "name": "run_code",
        "description": "在沙箱中执行 Python 代码片段，返回输出。",
        "input_schema": {
            "type": "object",
            "properties": {
                "code":    {"type": "string", "description": "要执行的 Python 代码"},
                "timeout": {"type": "integer", "description": "超时秒数，默认10"}
            },
            "required": ["code"]
        }
    }
]

print("工具定义结构（含缓存标记）:")
for tool in TOOLS_WITH_CACHE:
    cached = "✓ cached" if "cache_control" in tool else "  (no cache)"
    print(f"  [{cached}] {tool['name']}: {tool['description'][:40]}...")

print("""
缓存策略说明：
  - cache_control 加在工具列表中"最后一个想缓存的工具"上
  - API 会缓存从头到该标记为止的所有工具定义
  - 后续请求若工具定义不变，直接命中缓存，节省 ~90% 该部分的输入费用
  - 适用场景：工具集稳定、同一会话多次调用时收益最大
""")

# 演示：使用带缓存工具定义发起请求
resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=80,
    tools=TOOLS_WITH_CACHE,
    messages=[{"role": "user", "content": "帮我搜索'Python列表用法'相关文档"}]
)
print(f"模型决策: stop_reason={resp.stop_reason}")
if resp.stop_reason == "tool_use":
    tool_call = next(b for b in resp.content if b.type == "tool_use")
    print(f"  调用工具: {tool_call.name}")
    print(f"  参数: {tool_call.input}")
else:
    print(f"  回复: {resp.content[0].text[:100]}")


# ══════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════
section("总结：上下文工程核心原则")

print("""
1. 动态组装 > 固定提示
   像乐高一样按需拼装，避免"万能大提示"带来的噪音。

2. 缓存静态 = 省钱省时
   角色设定、知识库、工具定义 → 加 cache_control，
   用户消息、实时数据 → 每次新鲜传入。

3. 压缩长历史 ≠ 丢信息
   选 AI 摘要（高保真）或关键点提取（零成本），
   按场景取舍，不要让历史消耗掉所有 token 预算。

4. 信息分层 = 可维护
   Pinned / Session / Dynamic 三层分治，
   各司其职，互不污染。

5. 预算动态分配 = 不浪费不截断
   按复杂度给 max_tokens，为后续轮次留余量。

6. 上下文质量决定回复质量
   同一问题，背景越丰富、越精准，AI 越有用。
   垃圾进 → 垃圾出，黄金进 → 黄金出。
""")
