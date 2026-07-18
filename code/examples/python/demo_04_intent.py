"""
demo_04_intent.py — 意图识别：分类、槽位填充、路由

意图识别 = 搞清楚用户真正想做什么

两个层次：
  1. 意图分类：用户想干什么？（查信息 / 执行操作 / 闲聊 / 需要澄清）
  2. 槽位填充：需要哪些参数？哪些缺失需要追问？

实际工程中常用方案：
  a. 关键词规则（简单快速，可解释）
  b. LLM 分类（灵活准确，稍慢）
  c. 专门的分类模型（最快，需要训练数据）

本 demo 展示：规则 + LLM 两种方式的结合

运行：
  cd code && python examples/python/demo_04_intent.py
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Mock LLM（替换成真实 API 时改这里）
# ─────────────────────────────────────────────────────────────

def call_llm1(prompt: str) -> str:
    """
    Mock LLM 意图分类 — 真实版本：
        resp = client.messages.create(
            model="claude-haiku-...",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    """
    # 模拟 LLM 分类结果
    text = prompt.lower()
    if "订机票" in text or "预订" in text or "booking" in text:
        return "book_ticket"
    if "退款" in text or "取消" in text:
        return "cancel_order"
    if "查" in text and ("天气" in text or "weather" in text):
        return "query_weather"
    if "查" in text and ("快递" in text or "物流" in text or "订单" in text):
        return "query_order"
    if "你好" in text or "hi" in text or "hello" in text or "几岁" in text:
        return "chitchat"
    if "帮我" in text or "能不能" in text:
        return "task_request"
    return "unknown"



import anthropic
client = anthropic.Anthropic()
def call_llm(prompt: str) -> str:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()


# ─────────────────────────────────────────────────────────────
# 意图定义
# ─────────────────────────────────────────────────────────────

@dataclass
class Intent:
    """意图定义：名称 + 描述 + 必需槽位"""
    name: str
    description: str
    required_slots: list[str] = field(default_factory=list)
    optional_slots: list[str] = field(default_factory=list)


@dataclass
class Slot:
    """槽位：一个参数"""
    name: str
    value: Optional[str] = None
    filled: bool = False
    question: str = ""  # 缺失时怎么问用户


# 系统支持的所有意图
INTENTS = {
    "book_ticket": Intent(
        name="book_ticket",
        description="预订交通票务（机票、火车票等）",
        required_slots=["destination", "date"],
        optional_slots=["seat_class", "departure"],
    ),
    "query_weather": Intent(
        name="query_weather",
        description="查询天气情况",
        required_slots=["city"],
        optional_slots=["date"],
    ),
    "cancel_order": Intent(
        name="cancel_order",
        description="取消订单或申请退款",
        required_slots=["order_id"],
        optional_slots=["reason"],
    ),
    "query_order": Intent(
        name="query_order",
        description="查询订单状态或物流信息",
        required_slots=["order_id"],
    ),
    "chitchat": Intent(
        name="chitchat",
        description="闲聊，不需要调用工具",
    ),
    "unknown": Intent(
        name="unknown",
        description="无法识别，需要澄清",
    ),
}


# ─────────────────────────────────────────────────────────────
# 方式一：规则匹配（关键词 + 正则）
# ─────────────────────────────────────────────────────────────

INTENT_RULES = [
    # (正则 pattern, intent_name)
    (r"(订|预订|买|购买).*(机票|火车票|高铁|车票)", "book_ticket"),
    (r"(查|看|询问).*(天气|气温|下雨)", "query_weather"),
    (r"(退款|退货|取消).*(订单|购买|预订)", "cancel_order"),
    (r"(查|看|追踪).*(快递|物流|包裹|订单)", "query_order"),
    (r"(你好|hello|hi|在吗|聊聊)", "chitchat"),
]

def rule_based_classify(text: str) -> str:
    """方式一：规则匹配（快速，可解释，但覆盖不全）"""
    for pattern, intent in INTENT_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return intent
    return "unknown"


# ─────────────────────────────────────────────────────────────
# 方式二：LLM 分类（灵活，覆盖边缘情况）
# ─────────────────────────────────────────────────────────────

def llm_classify(text: str) -> str:
    """方式二：让 LLM 分类意图"""
    intent_descriptions = "\n".join(
        f"  - {k}: {v.description}" for k, v in INTENTS.items()
    )
    prompt = f"""将用户输入分类到以下意图之一：
{intent_descriptions}

用户输入："{text}"

只返回意图的英文名称（如 book_ticket），不要解释。"""

    return call_llm(prompt)


# ─────────────────────────────────────────────────────────────
# 槽位提取
# ─────────────────────────────────────────────────────────────

SLOT_PATTERNS = {
    "destination": r"去\s*([^\s，,。？！的到]+?)(?:\s|$|的|去|到|机票|票)",
    "date": r"(明天|后天|下周[一二三四五六日]|\d+月\d+[日号]|今天)",
    "city": r"([^\s，,。？！]+?)(?:的)?天气",
    "order_id": r"([A-Z]{2}\d{8,12}|\d{10,15})",
    "seat_class": r"(头等舱|商务舱|经济舱|软卧|硬卧|一等座|二等座)",
    "departure": r"从\s*([^\s，,。？！到]+?)\s*(?:出发|飞|到|去)",
}

def extract_slots(text: str, intent: Intent) -> dict[str, Slot]:
    """从用户文本中提取槽位值"""
    slots = {}
    all_slot_names = intent.required_slots + intent.optional_slots

    for slot_name in all_slot_names:
        slot = Slot(
            name=slot_name,
            question=f"请问{slot_name}是什么？",  # 默认追问语
        )

        # 用正则从文本提取
        pattern = SLOT_PATTERNS.get(slot_name)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                slot.value = match.group(1).strip()
                slot.filled = True

        slots[slot_name] = slot

    return slots


# ─────────────────────────────────────────────────────────────
# 意图识别 + 槽位填充 + 路由
# ─────────────────────────────────────────────────────────────

SLOT_QUESTIONS = {
    "destination": "您要去哪个城市？",
    "date": "请问是哪一天出发？",
    "city": "您想查哪个城市的天气？",
    "order_id": "请提供您的订单编号（格式如：ORD1234567890）。",
    "seat_class": "您需要什么舱位/座位等级？",
    "departure": "您从哪个城市出发？",
    "reason": "请简述退款原因（可选）。",
}

def process_user_input(text: str, verbose: bool = True) -> dict:
    """
    完整意图识别流程：
      1. 规则分类（先快速试一下）
      2. 规则失败 → LLM 分类（兜底）
      3. 提取槽位
      4. 检查必需槽位是否齐全
      5. 返回：意图 + 已填槽位 + 缺失槽位（需要追问）
    """
    if verbose:
        print(f"\n  输入: {text!r}")

    # ① 规则分类
    rule_intent = rule_based_classify(text)
    if verbose:
        print(f"  规则分类: {rule_intent}")

    # ② 规则不确定时，用 LLM 分类
    if rule_intent == "unknown":
        llm_intent = llm_classify(text)
        final_intent = llm_intent if llm_intent in INTENTS else "unknown"
        if verbose:
            print(f"  LLM分类:  {llm_intent} → 最终: {final_intent}")
    else:
        final_intent = rule_intent

    intent_def = INTENTS.get(final_intent, INTENTS["unknown"])

    # ③ 提取槽位
    slots = extract_slots(text, intent_def)

    # ④ 检查缺失的必需槽位
    missing = [
        s for s in intent_def.required_slots
        if not slots.get(s, Slot(name=s)).filled
    ]
    filled = {k: v.value for k, v in slots.items() if v.filled}

    # ⑤ 生成追问（如果有缺失槽位）
    follow_up = None
    if missing:
        slot_name = missing[0]  # 一次追问一个
        follow_up = SLOT_QUESTIONS.get(slot_name, f"请提供 {slot_name}。")

    result = {
        "intent": final_intent,
        "description": intent_def.description,
        "filled_slots": filled,
        "missing_required": missing,
        "follow_up_question": follow_up,
        "can_execute": len(missing) == 0 and final_intent != "unknown",
    }

    if verbose:
        print(f"  意图: {final_intent} ({intent_def.description})")
        print(f"  已填槽位: {filled}")
        if missing:
            print(f"  缺失必填: {missing}")
            print(f"  追问: 「{follow_up}」")
        else:
            print(f"  ✅ 槽位完整，可以执行")

    return result


# ─────────────────────────────────────────────────────────────
# 路由：根据意图调用对应工具/处理器
# ─────────────────────────────────────────────────────────────

def route_and_execute(result: dict) -> str:
    """意图路由：意图 → 对应的处理函数"""
    if not result["can_execute"]:
        return f"[无法执行] 缺少必填信息: {result['missing_required']}"

    intent = result["intent"]
    slots = result["filled_slots"]

    if intent == "book_ticket":
        return f"[执行] 正在为您预订 {slots.get('date', '待定')} 前往 {slots.get('destination', '待定')} 的票..."
    elif intent == "query_weather":
        return f"[执行] 正在查询 {slots.get('city', '待定')} 的天气..."
    elif intent == "cancel_order":
        return f"[执行] 正在处理订单 {slots.get('order_id', '待定')} 的退款申请..."
    elif intent == "query_order":
        return f"[执行] 正在查询订单 {slots.get('order_id', '待定')} 的物流信息..."
    elif intent == "chitchat":
        return "[执行] 直接回复，不调用工具"
    return "[执行] 未知意图，转人工客服"


# ─────────────────────────────────────────────────────────────
# Demo 运行
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Demo 04 — 意图识别：分类 + 槽位填充 + 路由")
    print("=" * 60)

    test_cases = [
        # ─── 信息完整，直接执行 ────────────────────────────
        "帮我订明天从北京去上海的机票",
        "查一下广州今天的天气",
        # ─── 信息缺失，触发追问 ────────────────────────────
        "我要订机票",           # 缺 destination 和 date
        "查一下天气",           # 缺 city
        "我要退款",             # 缺 order_id
        # ─── 闲聊（不需要工具）────────────────────────────
        "你好，你会什么？",
        # ─── 边缘情况（规则无法匹配，触发LLM分类）──────────
        "麻烦帮我查一下我的快递现在到哪了，单号是ORD12345678901",
    ]

    print("\n【规则 + LLM 意图分类对比】")
    print("─" * 50)

    for text in test_cases:
        print(f"\n{'─'*50}")
        result = process_user_input(text)
        execution = route_and_execute(result)
        print(f"  → 路由结果: {execution}")

    # ── 展示槽位填充的追问流程 ──────────────────────────────
    print("\n\n【多轮追问补全槽位】")
    print("─" * 50)

    print("\n  用户第1轮: 我要订机票")
    r1 = process_user_input("我要订机票")
    print(f"  系统追问: 「{r1['follow_up_question']}」")

    print("\n  用户第2轮: 去上海")
    r2 = process_user_input("去上海", verbose=False)
    # 合并已知信息
    merged_slots = {**r1["filled_slots"], **r2["filled_slots"]}
    still_missing = [s for s in ["destination", "date"] if s not in merged_slots]
    print(f"  已收集: {merged_slots}")
    if still_missing:
        print(f"  系统追问: 「{SLOT_QUESTIONS.get(still_missing[0])}」")

    print("\n  用户第3轮: 明天出发")
    r3 = process_user_input("明天出发", verbose=False)
    final_slots = {**merged_slots, **r3["filled_slots"]}
    print(f"  已收集: {final_slots}")
    print(f"  ✅ 槽位全部填满！")
    print(f"  → 执行: 正在预订明天前往上海的机票...")

    print("\n" + "=" * 60)
    print("  关键结论：")
    print("  1. 意图分类：规则兜底 → LLM 精判（两者结合效果最好）")
    print("  2. 槽位填充：从用户输入正则提取，缺什么问什么")
    print("  3. 多轮追问：每轮补充一个缺失槽位，直到完整")
    print("  4. 路由：意图 → 对应工具/处理器（解耦设计）")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
============================================================
  Demo 04 — 意图识别：分类 + 槽位填充 + 路由
============================================================

【规则 + LLM 意图分类对比】
──────────────────────────────────────────────────

──────────────────────────────────────────────────

  输入: '帮我订明天从北京去上海的机票'
  规则分类: book_ticket
  意图: book_ticket (预订交通票务（机票、火车票等）)
  已填槽位: {'destination': '上海', 'date': '明天', 'departure': '北京'}
  ✅ 槽位完整，可以执行
  → 路由结果: [执行] 正在为您预订 明天 前往 上海 的票...

──────────────────────────────────────────────────

  输入: '查一下广州今天的天气'
  规则分类: query_weather
  意图: query_weather (查询天气情况)
  已填槽位: {'city': '查一下广州今天', 'date': '今天'}
  ✅ 槽位完整，可以执行
  → 路由结果: [执行] 正在查询 查一下广州今天 的天气...

──────────────────────────────────────────────────

  输入: '我要订机票'
  规则分类: book_ticket
  意图: book_ticket (预订交通票务（机票、火车票等）)
  已填槽位: {}
  缺失必填: ['destination', 'date']
  追问: 「您要去哪个城市？」
  → 路由结果: [无法执行] 缺少必填信息: ['destination', 'date']

──────────────────────────────────────────────────

  输入: '查一下天气'
  规则分类: query_weather
  意图: query_weather (查询天气情况)
  已填槽位: {'city': '查一下'}
  ✅ 槽位完整，可以执行
  → 路由结果: [执行] 正在查询 查一下 的天气...

──────────────────────────────────────────────────

  输入: '我要退款'
  规则分类: unknown
  LLM分类:  cancel_order → 最终: cancel_order
  意图: cancel_order (取消订单或申请退款)
  已填槽位: {}
  缺失必填: ['order_id']
  追问: 「请提供您的订单编号（格式如：ORD1234567890）。」
  → 路由结果: [无法执行] 缺少必填信息: ['order_id']

──────────────────────────────────────────────────

  输入: '你好，你会什么？'
  规则分类: chitchat
  意图: chitchat (闲聊，不需要调用工具)
  已填槽位: {}
  ✅ 槽位完整，可以执行
  → 路由结果: [执行] 直接回复，不调用工具

──────────────────────────────────────────────────

  输入: '麻烦帮我查一下我的快递现在到哪了，单号是ORD12345678901'
  规则分类: query_order
  意图: query_order (查询订单状态或物流信息)
  已填槽位: {'order_id': 'RD12345678901'}
  ✅ 槽位完整，可以执行
  → 路由结果: [执行] 正在查询订单 RD12345678901 的物流信息...


【多轮追问补全槽位】
──────────────────────────────────────────────────

  用户第1轮: 我要订机票

  输入: '我要订机票'
  规则分类: book_ticket
  意图: book_ticket (预订交通票务（机票、火车票等）)
  已填槽位: {}
  缺失必填: ['destination', 'date']
  追问: 「您要去哪个城市？」
  系统追问: 「您要去哪个城市？」

  用户第2轮: 去上海
  已收集: {'destination': '上海'}
  系统追问: 「请问是哪一天出发？」

  用户第3轮: 明天出发
  已收集: {'destination': '上海', 'date': '明天'}
  ✅ 槽位全部填满！
  → 执行: 正在预订明天前往上海的机票...

============================================================
  关键结论：
  1. 意图分类：规则兜底 → LLM 精判（两者结合效果最好）
  2. 槽位填充：从用户输入正则提取，缺什么问什么
  3. 多轮追问：每轮补充一个缺失槽位，直到完整
  4. 路由：意图 → 对应工具/处理器（解耦设计）
============================================================
"""