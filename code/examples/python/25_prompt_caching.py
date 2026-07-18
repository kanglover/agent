"""
25_prompt_caching.py
Prompt Caching 完整指南

核心思想：
  把"不变的内容"标记为可缓存，让 Claude 在后续请求中直接读取缓存副本，
  跳过重新处理的步骤。读取缓存比重新计算便宜约 90%，延迟也大幅降低。

涵盖内容：
  1. Prompt Caching 工作原理（5分钟 TTL、最小 1024 tokens）
  2. cache_control 标记位置（system prompt 最前面）
  3. 工具定义缓存（大量工具时缓存工具列表）
  4. 长文档缓存（把参考文档放进 cache）
  5. 缓存命中率测试（cache_creation vs cache_read tokens 对比）
  6. 成本节省计算（cache_read 比 normal 便宜 90%）
  7. 缓存失效场景（模型变更、5分钟超时）
  8. 生产中的缓存策略（什么内容值得缓存）
  9. 多轮对话中的缓存优化

依赖安装：
  pip install anthropic

运行：
  python 25_prompt_caching.py
"""

import os
import time
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def show_usage(usage, label: str = ""):
    """格式化打印 token 用量，清晰展示缓存命中情况。"""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Token 用量明细：")
    print(f"  input_tokens              = {usage.input_tokens:,}   ← 普通输入（按正常价计费）")
    print(f"  cache_creation_input_tokens = {getattr(usage, 'cache_creation_input_tokens', 0):,}   ← 写入缓存（收 1.25× 费用）")
    print(f"  cache_read_input_tokens     = {getattr(usage, 'cache_read_input_tokens', 0):,}   ← 读取缓存（只收 0.1× 费用）")
    print(f"  output_tokens             = {usage.output_tokens:,}   ← 输出 token")


def calc_savings(usage, input_price_per_mtok: float = 3.0) -> float:
    """
    估算本次请求因缓存节省的费用（USD）。

    定价参考（claude-3-5-sonnet 为例，单位：USD / 百万 token）：
      - 普通 input：$3.00
      - cache write：$3.75（1.25×）
      - cache read ：$0.30（0.1×）
    """
    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
    if cache_read == 0:
        return 0.0
    # 若没有缓存，这些 token 按正常 input 价格计费
    normal_cost  = cache_read * input_price_per_mtok / 1_000_000
    # 实际只收 cache_read 价格（0.1×）
    actual_cost  = cache_read * (input_price_per_mtok * 0.1) / 1_000_000
    saved = normal_cost - actual_cost
    return saved


# ─────────────────────────────────────────────
# 1. Prompt Caching 工作原理
# ─────────────────────────────────────────────

def demo_caching_basics():
    """
    演示 Prompt Caching 的基本机制。

    关键规则：
    - 必须在消息/系统提示的 content 块上加 "cache_control": {"type": "ephemeral"}
    - 缓存的最小粒度：1024 tokens（不足则不会缓存）
    - TTL（存活时间）：5 分钟，超时自动失效
    - 每次请求最多可以有 4 个独立的缓存断点（breakpoint）
    - 缓存按"前缀"匹配：只要请求的前缀与已缓存内容完全一致，就会命中
    """
    section("1. Prompt Caching 工作原理")

    print("""
工作原理图解：

  第 1 次请求
  ┌──────────────────────────────────────────────────┐
  │  [system: 长篇背景文档 🏷️ cache_control]          │  ← 写入缓存
  │  [user: 问题 A]                                  │
  └──────────────────────────────────────────────────┘
   → cache_creation_input_tokens: N（付 1.25× 费写缓存）

  第 2 次请求（5 分钟内，前缀相同）
  ┌──────────────────────────────────────────────────┐
  │  [system: 长篇背景文档 🏷️ cache_control]          │  ← 命中缓存！
  │  [user: 问题 B]                                  │
  └──────────────────────────────────────────────────┘
   → cache_read_input_tokens: N（只付 0.1× 费读缓存）

关键数字：
  最小缓存 token 数 = 1024
  缓存 TTL         = 5 分钟（每次命中会刷新计时器）
  最多缓存断点     = 4 个（同一请求内）
  写入倍率         = 1.25×（比普通 input 贵 25%）
  读取倍率         = 0.10×（比普通 input 便宜 90%）
""")


# ─────────────────────────────────────────────
# 2. cache_control 标记位置示例
# ─────────────────────────────────────────────

def demo_cache_control_placement():
    """
    演示 cache_control 的正确标记位置。

    原则：把"几乎不变的长内容"放在最前面并打上标记；
    把"每次都不同的内容"放在标记之后，不打标记。
    """
    section("2. cache_control 标记位置")

    # ── 构造一段"足够长"的系统提示来触发缓存 ──────────────────────────────
    # 真实场景下这里会是产品说明书、代码库、法规文档等，轻松超过 1024 tokens
    long_static_context = (
        "你是一位资深 Python 工程师，专注于后端开发和系统架构设计。\n\n"
        + "以下是本项目的完整技术规范（模拟长文档）：\n\n"
        + "项目使用 FastAPI + PostgreSQL + Redis 的技术栈。\n"
        + "数据库设计遵循第三范式，所有外键必须建立索引。\n"
        + "API 响应时间 P99 必须低于 200ms。\n"
        + "所有接口必须支持分页，默认 page_size=20，最大 100。\n"
        + "错误码遵循 RFC 7807 Problem Details 标准。\n"
        + "认证使用 JWT，access_token 有效期 15 分钟，refresh_token 7 天。\n"
        + "日志格式为 JSON，必须包含 trace_id、user_id、duration_ms 字段。\n"
        + "部署在 Kubernetes，每个服务至少 2 个副本，支持水平扩展。\n"
        + "CI/CD 使用 GitHub Actions，main 分支合并自动触发部署。\n"
        + "代码覆盖率要求不低于 80%，核心业务逻辑不低于 95%。\n"
        + "所有配置通过环境变量注入，禁止硬编码敏感信息。\n"
        + "数据库迁移使用 Alembic，禁止直接修改生产数据库 schema。\n"
        + "缓存策略：用户信息缓存 5 分钟，商品信息缓存 30 分钟，配置缓存 1 小时。\n"
        + "异步任务使用 Celery + Redis，重试策略为指数退避，最大重试 3 次。\n"
        # … 真实场景中此处会有更多内容，轻松超过 1024 tokens
    ) * 3  # 重复 3 次以凑足演示长度

    print("正确写法：cache_control 放在 system prompt 内容块上\n")

    # ── 正确写法：在 system 的 content 列表中，给最后一个 text 块打标记 ──
    request_params = {
        "model": "claude-opus-4-5",
        "max_tokens": 256,
        "system": [
            {
                "type": "text",
                "text": long_static_context,
                "cache_control": {"type": "ephemeral"},   # ← 关键：标记在这里
            }
            # 如果还有其他 system 块，可以继续追加（最多 4 个断点）
        ],
        "messages": [
            {
                "role": "user",
                "content": "请用一句话总结项目的认证策略。"   # ← 每次可以不同
            }
        ]
    }

    print("请求结构：")
    print("""
  system: [
    {
      "type": "text",
      "text": "（超过 1024 tokens 的长文档...）",
      "cache_control": {"type": "ephemeral"}   ← 打标记
    }
  ],
  messages: [
    {"role": "user", "content": "具体问题（每次可以不同）"}
  ]
""")

    print("发送第 1 次请求（写入缓存）...")
    try:
        resp1 = client.messages.create(**request_params)
        print(f"回答：{resp1.content[0].text}")
        show_usage(resp1.usage, "第1次")

        print("\n等待 1 秒后发送第 2 次请求（应命中缓存）...")
        time.sleep(1)

        # 修改 user 问题，但 system 前缀不变 → 应命中缓存
        request_params["messages"][0]["content"] = "请用一句话总结项目的日志要求。"
        resp2 = client.messages.create(**request_params)
        print(f"回答：{resp2.content[0].text}")
        show_usage(resp2.usage, "第2次")

        cr = getattr(resp2.usage, 'cache_read_input_tokens', 0)
        if cr > 0:
            print(f"\n缓存命中！节省了约 ${calc_savings(resp2.usage):.6f} USD")
        else:
            print("\n缓存未命中（可能内容不足 1024 tokens 或模型不支持）")
    except Exception as e:
        print(f"API 调用出错（可能无 API Key）：{e}")
        print("以上代码结构是正确的，实际运行需要有效的 ANTHROPIC_API_KEY")


# ─────────────────────────────────────────────
# 3. 工具定义缓存
# ─────────────────────────────────────────────

def demo_tool_caching():
    """
    演示工具定义缓存。

    场景：Agent 挂载了几十个工具（搜索、数据库、文件系统、API 等）。
    工具的 JSON Schema 描述可能有几千 tokens，但每次对话都是一样的。
    → 把工具列表标记为可缓存，后续请求读取缓存即可。

    标记位置：tools 列表中最后一个工具的定义上加 cache_control。
    """
    section("3. 工具定义缓存")

    # 模拟一个拥有大量工具的 Agent
    many_tools = [
        {
            "name": f"tool_{i:02d}",
            "description": (
                f"这是工具 {i:02d} 的详细描述。"
                f"它可以执行第 {i} 类操作，"
                f"支持参数 A、B、C，"
                f"返回结构化 JSON 数据。"
                f"适用场景：当用户需要进行 {i} 号任务时调用。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": f"工具 {i:02d} 的查询参数"},
                    "limit": {"type": "integer", "description": "返回结果数量上限", "default": 10},
                },
                "required": ["query"],
            },
        }
        for i in range(1, 21)  # 20 个工具，模拟真实 Agent
    ]

    # ── 关键：在最后一个工具上打 cache_control ───────────────────────────
    many_tools[-1]["cache_control"] = {"type": "ephemeral"}

    print("工具缓存结构（最后一个工具打标记）：")
    print("""
  tools = [
    {"name": "tool_01", "description": "...", "input_schema": {...}},
    {"name": "tool_02", "description": "...", "input_schema": {...}},
    ...
    {"name": "tool_20", "description": "...", "input_schema": {...},
     "cache_control": {"type": "ephemeral"}},   ← 只在最后一个工具打标记
  ]
""")
    print("原理：缓存是按"前缀"工作的，只需在最后一个工具后设置断点，")
    print("整个工具列表（tool_01 ~ tool_20）就会被包含在缓存前缀中。\n")

    print("适合缓存工具定义的场景：")
    print("  ✅ 工具列表固定不变（大多数 Agent 都是这种情况）")
    print("  ✅ 工具数量多（> 5 个），描述详细，总 token 数 > 1024")
    print("  ❌ 工具列表会随用户权限动态变化（缓存会频繁失效，得不偿失）")


# ─────────────────────────────────────────────
# 4. 长文档缓存
# ─────────────────────────────────────────────

def demo_long_document_caching():
    """
    演示如何把参考文档放入缓存。

    典型场景：
    - RAG：把检索到的文档块放进缓存，多个问题共用同一份上下文
    - 代码审查：把整个代码库放进缓存，然后反复提问
    - 合同分析：把合同全文放进缓存，分多次提取不同条款
    """
    section("4. 长文档缓存")

    print("长文档缓存的两种模式：\n")

    print("── 模式 A：文档放 system，问题放 user ──")
    print("""
  system: [
    {"type": "text", "text": "你是文档分析助手"},
    {"type": "text", "text": "【参考文档全文，数万字...】",
     "cache_control": {"type": "ephemeral"}}   ← 文档打标记
  ]
  messages: [{"role": "user", "content": "问题 1"}]
  → 后续问题 2、3、4 都能命中文档缓存
""")

    print("── 模式 B：文档放 user 的第一条消息（多轮对话场景）──")
    print("""
  messages: [
    {
      "role": "user",
      "content": [
        {"type": "text",
         "text": "【参考文档全文，数万字...】",
         "cache_control": {"type": "ephemeral"}},   ← 文档打标记
        {"type": "text", "text": "基于以上文档，回答：问题 1"}
      ]
    },
    {"role": "assistant", "content": "回答 1..."},
    {"role": "user", "content": "继续追问：问题 2"}   ← 文档已在缓存中
  ]
""")

    print("最佳实践：")
    print("  1. 把稳定的大文档放在消息列表的最前面")
    print("  2. 把变化的问题放在后面")
    print("  3. 缓存断点（cache_control）尽量靠近"稳定内容"的末尾")
    print("  4. 一次请求可以设置最多 4 个断点，用于多层级缓存")

    print("\n多层级缓存断点示例：")
    print("""
  system: [
    {"text": "角色定义（100 tokens）",
     "cache_control": {"type": "ephemeral"}},   ← 断点 1

    {"text": "领域知识库（5000 tokens）",
     "cache_control": {"type": "ephemeral"}},   ← 断点 2
  ]
  tools: [
    ...
    {"name": "last_tool", ...,
     "cache_control": {"type": "ephemeral"}},   ← 断点 3
  ]
  messages: [
    {用户上传的文档（10000 tokens）+ cache_control},   ← 断点 4
    {用户问题（无标记）}
  ]
""")


# ─────────────────────────────────────────────
# 5. 缓存命中率测试
# ─────────────────────────────────────────────

def demo_cache_hit_test():
    """
    演示如何测量缓存命中率，对比两次请求的 token 用量。
    """
    section("5. 缓存命中率测试")

    print("缓存命中率 = cache_read_input_tokens / (cache_creation + cache_read)\n")

    # 模拟一份"足够长"的文档（实际使用时替换为真实文档）
    big_doc = (
        "这是一份模拟的技术文档，包含大量背景知识。\n"
        "章节一：系统架构概述\n"
        "本系统采用微服务架构，共包含 12 个独立服务。\n"
        "每个服务通过 gRPC 进行内部通信，通过 REST API 对外暴露接口。\n"
        "章节二：数据模型\n"
        "核心数据实体包括：用户（User）、订单（Order）、商品（Product）。\n"
        "用户表包含：id, email, created_at, last_login, status 字段。\n"
        "订单表包含：id, user_id, total_amount, status, created_at 字段。\n"
        "章节三：安全规范\n"
        "所有接口必须验证 JWT token。\n"
        "敏感数据（手机号、邮箱）在数据库中加密存储。\n"
        "操作日志保留 90 天，安全日志保留 1 年。\n"
    ) * 10  # 扩展内容以超过 1024 tokens

    def make_request(question: str, label: str):
        start = time.time()
        try:
            resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=128,
                system=[
                    {
                        "type": "text",
                        "text": big_doc,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": question}],
            )
            elapsed = time.time() - start
            print(f"\n{label} 用时：{elapsed:.2f}s")
            show_usage(resp.usage, label)

            cr  = getattr(resp.usage, 'cache_read_input_tokens', 0)
            cc  = getattr(resp.usage, 'cache_creation_input_tokens', 0)
            total_cache = cr + cc
            hit_rate = (cr / total_cache * 100) if total_cache > 0 else 0
            print(f"  缓存命中率：{hit_rate:.1f}%")
            print(f"  节省费用：约 ${calc_savings(resp.usage):.6f} USD")
            return resp
        except Exception as e:
            print(f"API 调用出错：{e}")
            return None

    print("发送请求 1（首次，预计写入缓存）：")
    make_request("用户表有哪些字段？", "请求1")

    print("\n等待 2 秒后发送请求 2（预计命中缓存）：")
    time.sleep(2)
    make_request("订单表有哪些字段？", "请求2")

    print("\n等待 2 秒后发送请求 3（预计命中缓存）：")
    time.sleep(2)
    make_request("安全日志保留多久？", "请求3")


# ─────────────────────────────────────────────
# 6. 成本节省计算
# ─────────────────────────────────────────────

def demo_cost_savings():
    """
    用数字说明 Prompt Caching 的成本节省。
    """
    section("6. 成本节省计算")

    # Claude Sonnet 3.5 定价（2024 年，USD / 百万 tokens）
    PRICE = {
        "input":         3.00,   # 普通 input
        "cache_write":   3.75,   # cache 写入（1.25×）
        "cache_read":    0.30,   # cache 读取（0.10×）
        "output":       15.00,   # output
    }

    print("定价表（以 claude-sonnet-4-5 为例）：")
    print(f"  普通 input  = ${PRICE['input']:.2f} / 百万 tokens")
    print(f"  cache write = ${PRICE['cache_write']:.2f} / 百万 tokens（1.25×）")
    print(f"  cache read  = ${PRICE['cache_read']:.2f} / 百万 tokens（0.10×）")
    print(f"  output      = ${PRICE['output']:.2f} / 百万 tokens\n")

    # 场景：每天 1000 次请求，每次携带 10,000 tokens 的系统提示
    daily_requests    = 1000
    system_prompt_toks = 10_000   # tokens
    output_toks        = 500      # 每次回复约 500 tokens

    print(f"场景：每天 {daily_requests:,} 次请求，系统提示 {system_prompt_toks:,} tokens\n")

    # 不使用缓存：每次都按普通 input 计费
    cost_no_cache = (
        daily_requests * system_prompt_toks * PRICE["input"] / 1_000_000
        + daily_requests * output_toks * PRICE["output"] / 1_000_000
    )

    # 使用缓存：
    #   第 1 次：写入缓存（1.25× input）
    #   后续 999 次：读取缓存（0.10× input）
    cost_cache_write = 1 * system_prompt_toks * PRICE["cache_write"] / 1_000_000
    cost_cache_read  = (daily_requests - 1) * system_prompt_toks * PRICE["cache_read"] / 1_000_000
    cost_output      = daily_requests * output_toks * PRICE["output"] / 1_000_000
    cost_with_cache  = cost_cache_write + cost_cache_read + cost_output

    savings      = cost_no_cache - cost_with_cache
    savings_pct  = savings / cost_no_cache * 100

    print(f"不使用缓存：${cost_no_cache:.4f} / 天")
    print(f"使用缓存  ：${cost_with_cache:.4f} / 天")
    print(f"每天节省  ：${savings:.4f}（节省 {savings_pct:.1f}%）")
    print(f"每月节省  ：${savings * 30:.2f}")
    print(f"每年节省  ：${savings * 365:.2f}\n")

    print("快速估算公式：")
    print("  节省率 ≈ (1 - 0.1) × 系统提示占比")
    print("  例如系统提示占总 input 的 80%，节省率 ≈ 0.9 × 80% = 72%")


# ─────────────────────────────────────────────
# 7. 缓存失效场景
# ─────────────────────────────────────────────

def demo_cache_invalidation():
    """
    列举会导致缓存失效（即无法命中）的场景。
    """
    section("7. 缓存失效场景")

    print("以下任何一种变化都会导致缓存失效（cache miss）：\n")

    scenarios = [
        (
            "5 分钟 TTL 超时",
            "缓存的默认生存时间是 5 分钟。每次命中会刷新计时。\n"
            "  → 解决方案：对于低频用例，接受首次请求的写入成本；\n"
            "               对于高频用例，保持请求节奏在 5 分钟以内。",
        ),
        (
            "切换模型版本",
            "从 claude-sonnet-4-5 换成 claude-opus-4-5 会使缓存失效。\n"
            "  → 缓存与模型绑定，不同模型有不同的缓存命名空间。",
        ),
        (
            "修改缓存断点之前的内容",
            "缓存按前缀匹配。若在 cache_control 标记之前插入/修改任何内容，\n"
            "  整个缓存失效，需要重新写入。\n"
            "  → 解决方案：把稳定内容放前面，不稳定内容放后面。",
        ),
        (
            "temperature / top_p 等参数不同",
            "不同的推理参数会影响缓存命中（某些参数组合会使缓存失效）。\n"
            "  → 对于同一缓存前缀，尽量保持推理参数一致。",
        ),
        (
            "API 账户 / 组织切换",
            "缓存与 API key（账户/组织）绑定，不同账户之间不共享缓存。",
        ),
        (
            "工具列表顺序变化",
            "tools 列表的顺序也是前缀的一部分。调整工具顺序会导致缓存失效。\n"
            "  → 解决方案：工具列表按固定顺序排列，不要动态调整顺序。",
        ),
    ]

    for i, (title, desc) in enumerate(scenarios, 1):
        print(f"  {i}. {title}")
        print(f"     {desc}\n")


# ─────────────────────────────────────────────
# 8. 生产中的缓存策略
# ─────────────────────────────────────────────

def demo_production_strategy():
    """
    讲解生产环境中如何设计缓存策略。
    """
    section("8. 生产中的缓存策略")

    print("判断一段内容是否值得缓存的标准：\n")

    criteria = [
        ("✅ 值得缓存", [
            "系统提示（角色定义、行为规范）— 每次请求都相同",
            "工具定义列表 — 20+ 个工具的 JSON Schema 描述",
            "领域知识库 — 行业规范、产品手册、法规文本",
            "代码库快照 — 代码审查、代码问答时的参考代码",
            "会话前缀 — 多轮对话中不变的开场白和说明",
        ]),
        ("❌ 不值得缓存", [
            "短内容（< 1024 tokens）— 低于最小缓存阈值",
            "高频变化的内容 — 每次请求都不同，缓存命中率为零",
            "用户个性化数据 — 每位用户不同，无法共用缓存",
            "时效性强的数据 — 实时价格、实时天气等",
            "一次性内容 — 只请求一次，没有"后续命中"的机会",
        ]),
    ]

    for label, items in criteria:
        print(f"  {label}：")
        for item in items:
            print(f"    - {item}")
        print()

    print("生产系统缓存层次设计建议：\n")
    print("""
  Layer 1（最稳定，最长 TTL）
  ┌─────────────────────────────────────────┐
  │  系统角色定义 + 领域知识库               │  ← 几乎不变，缓存价值最高
  │  cache_control: ephemeral（断点 1）      │
  └─────────────────────────────────────────┘

  Layer 2（按会话缓存）
  ┌─────────────────────────────────────────┐
  │  工具定义列表                            │  ← 按用户角色不同
  │  cache_control: ephemeral（断点 2）      │
  └─────────────────────────────────────────┘

  Layer 3（按请求缓存）
  ┌─────────────────────────────────────────┐
  │  本次任务的参考文档                      │  ← 同一任务的多个子问题共用
  │  cache_control: ephemeral（断点 3）      │
  └─────────────────────────────────────────┘

  动态部分（不缓存）
  ┌─────────────────────────────────────────┐
  │  用户的具体问题（每次不同）               │
  └─────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────
# 9. 多轮对话中的缓存优化
# ─────────────────────────────────────────────

def demo_multi_turn_caching():
    """
    演示多轮对话中的缓存优化技巧。

    关键洞察：
    每一轮对话都会在消息列表末尾追加新消息，这会改变"前缀"。
    → 把"不变的系统提示"放在 system 字段，而不是 messages 里，
      这样无论对话进行多少轮，system 的缓存始终有效。
    """
    section("9. 多轮对话中的缓存优化")

    print("多轮对话的挑战：\n")
    print("  每追加一条消息，messages 列表就变长了一点。")
    print("  如果把"大文档"放在 messages[0] 里：")
    print("    第 1 轮：messages = [文档, 问题1]          → 前缀包含文档 ✅ 命中")
    print("    第 2 轮：messages = [文档, 问题1, 答1, 问题2] → 前缀仍包含文档 ✅ 命中")
    print("    第 N 轮：messages = [文档, ..., 问题N]     → 前缀仍包含文档 ✅ 命中")
    print()
    print("  ✅ 结论：只要文档放在消息列表的最前面并打上标记，多轮对话中缓存一直有效。\n")

    # 演示多轮对话中缓存持续命中的代码结构
    conversation_history = []
    static_document = (
        "这是一份关于 Python 最佳实践的指导文档。\n"
        "1. 使用类型注解提升代码可读性\n"
        "2. 使用 dataclass 代替普通 class 处理数据容器\n"
        "3. 使用 pathlib 代替 os.path 处理文件路径\n"
        "4. 使用 contextmanager 管理资源，避免资源泄漏\n"
        "5. 异步 I/O 使用 asyncio，避免阻塞主线程\n"
        "6. 日志使用 structlog 或 logging，不要用 print\n"
        "7. 配置使用 pydantic-settings，支持环境变量注入\n"
        "8. 测试使用 pytest + pytest-asyncio，覆盖率 > 80%\n"
        "9. 依赖管理使用 poetry 或 uv，锁定版本\n"
        "10. 代码格式化使用 ruff，静态检查使用 mypy\n"
    ) * 8  # 扩展内容

    def multi_turn_request(user_question: str, turn: int) -> str:
        """
        构建多轮对话请求，每一轮都在文档缓存的基础上追加新消息。
        """
        # 本轮的 user 消息（不含缓存标记）
        conversation_history.append({
            "role": "user",
            "content": user_question,
        })

        # ── 把"文档"作为 user 消息列表的第一条，并打缓存标记 ──────────────
        # 注意：第一条消息是带文档的特殊格式，后续消息是纯文本
        messages_with_doc = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_document,
                        "cache_control": {"type": "ephemeral"},   # ← 文档缓存
                    },
                    {
                        "type": "text",
                        "text": conversation_history[0]["content"],  # 第一轮问题
                    },
                ],
            }
        ]

        # 追加后续对话（第 2 轮之后）
        for msg in conversation_history[1:]:
            messages_with_doc.append(msg)

        try:
            resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=128,
                messages=messages_with_doc,
            )
            answer = resp.content[0].text
            # 把 AI 回答加入历史
            conversation_history.append({
                "role": "assistant",
                "content": answer,
            })

            cr = getattr(resp.usage, 'cache_read_input_tokens', 0)
            cc = getattr(resp.usage, 'cache_creation_input_tokens', 0)
            status = "命中缓存 ✅" if cr > 0 else ("写入缓存 📝" if cc > 0 else "无缓存")
            print(f"  第 {turn} 轮：{status}  (read={cr}, create={cc})")
            return answer
        except Exception as e:
            print(f"  第 {turn} 轮：API 调用出错 → {e}")
            return ""

    print("多轮对话缓存演示（使用实际 API）：")
    questions = [
        "类型注解有什么好处？",
        "什么时候用 dataclass？",
        "pathlib 比 os.path 好在哪里？",
    ]

    for i, q in enumerate(questions, 1):
        multi_turn_request(q, i)

    print()
    print("最佳实践总结：")
    print("  1. 把稳定的长文档放在 messages[0] 的 content 列表首位并打标记")
    print("  2. system 字段放角色定义（同样可以打标记）")
    print("  3. 多轮对话中每次都携带完整历史 + 缓存标记，缓存会持续命中")
    print("  4. 不要在缓存内容之前插入任何"每次不同"的文本")


# ─────────────────────────────────────────────
# 主函数：按章节顺序演示
# ─────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  25_prompt_caching.py  —  Prompt Caching 完整指南")
    print("=" * 62)

    demo_caching_basics()        # 1. 工作原理
    demo_cache_control_placement()  # 2. 标记位置（真实 API 调用）
    demo_tool_caching()          # 3. 工具定义缓存
    demo_long_document_caching() # 4. 长文档缓存
    demo_cache_hit_test()        # 5. 命中率测试（真实 API 调用）
    demo_cost_savings()          # 6. 成本节省计算
    demo_cache_invalidation()    # 7. 缓存失效场景
    demo_production_strategy()   # 8. 生产缓存策略
    demo_multi_turn_caching()    # 9. 多轮对话缓存（真实 API 调用）

    print("\n" + "=" * 62)
    print("  演示完毕！核心要点：")
    print("  - cache_control 打在"不变内容"的最后一个文本块上")
    print("  - 缓存 TTL = 5 分钟，最小 1024 tokens")
    print("  - cache_read 只需 0.1× 普通 input 价格，节省 90%")
    print("  - 稳定内容放前面，变化内容放后面，缓存命中率才高")
    print("=" * 62)


if __name__ == "__main__":
    main()
