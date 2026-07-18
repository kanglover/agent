# 意图识别与槽位填充

> 沉淀时间：2026-07-06

## 一句话总结

意图识别 = 搞清楚用户真正想干什么；槽位填充 = 收集执行任务所需的所有参数。

## 解决什么问题？

用户说话往往模糊或不完整：

```
"帮我看看这个"  → 看什么？看完要干嘛？
"帮我订机票"    → 去哪？哪天？几等座？
"查一下天气"    → 哪个城市？
```

意图识别负责"知道用户想干什么"，槽位填充负责"收集执行所需的参数"。

---

## 意图识别

### 两种实现方式

**方式一：关键词规则（快速、可解释，覆盖不全）**

```python
INTENT_RULES = [
    (r"(订|预订).*(机票|火车票)", "book_ticket"),
    (r"(查|看).*(天气|气温)",     "query_weather"),
    (r"(退款|退货|取消).*(订单)", "cancel_order"),
    (r"(你好|hello|hi)",          "chitchat"),
]

def rule_based_classify(text: str) -> str:
    for pattern, intent in INTENT_RULES:
        if re.search(pattern, text):
            return intent
    return "unknown"  # 规则无法覆盖时
```

**方式二：LLM 分类（灵活、准确，适合边缘情况）**

```python
def llm_classify(text: str) -> str:
    prompt = f"""将用户输入分类到以下意图之一：
- book_ticket: 预订票务
- query_weather: 查天气
- cancel_order: 退款/取消
- chitchat: 闲聊

用户输入："{text}"
只返回意图名称。"""
    return call_llm(prompt)
```

**推荐组合：规则先跑，失败时 LLM 兜底**

```python
intent = rule_based_classify(text)
if intent == "unknown":
    intent = llm_classify(text)  # 只在规则不确定时才调 LLM（省钱）
```

---

## 槽位填充（Slot Filling）

### 是什么

意图确定后，用正则或 NER 从用户文本中提取执行所需的参数：

```
用户："帮我订明天去上海的机票"

意图: book_ticket
  ├── destination: 上海  ✓ 已提取
  ├── date:        明天  ✓ 已提取
  ├── departure:   ???  ✗ 缺失！→ 追问
  └── seat_class:  ???  ✗ 缺失，但非必填，用默认值
```

### 实现：正则提取

```python
SLOT_PATTERNS = {
    "destination": r"去\s*([^\s，到]+?)(?:\s|的|到|机票)",
    "date":        r"(明天|后天|下周[一-日]|\d+月\d+[日号])",
    "city":        r"([^\s，]+?)(?:的)?天气",
    "order_id":    r"([A-Z]{2}\d{8,12}|\d{10,15})",
}
```

---

## 多轮追问流程

当必填槽位缺失时，**一次追问一个**，逐步补全：

```
用户: 我要订机票
  → 意图=book_ticket，缺 destination 和 date
  → 追问：「您要去哪个城市？」

用户: 去上海
  → destination=上海，还缺 date
  → 追问：「请问是哪一天出发？」

用户: 明天
  → 所有必填槽位齐全 → 执行！
```

实际效果（真实跑出来的）：
```
已收集: {'destination': '上海', 'date': '明天'}
✅ 槽位全部填满！
→ 执行: 正在预订明天前往上海的机票...
```

---

## 意图路由

意图识别完成后，根据意图分发到不同的处理函数：

```python
def route(intent: str, slots: dict) -> str:
    if intent == "book_ticket":
        return book_ticket_api(**slots)
    elif intent == "query_weather":
        return weather_api(city=slots["city"])
    elif intent == "chitchat":
        return llm_chat(user_input)   # 不调工具，直接让 LLM 聊
    elif intent == "unknown":
        return transfer_to_human()    # 转人工
```

**好处：解耦设计**，新增意图只需加一个 handler，不改主流程。

---

## 规则 vs LLM 的真实局限

从代码运行结果观察到：

- **规则的局限**：城市提取出了"查一下广州今天"而不是"广州"（正则不够精准）
- **LLM 的局限**：mock 分类"我要退款"错误分到了 book_ticket（实际 LLM 不会犯这种错）
- **结论**：规则适合高频标准场景，LLM 适合长尾模糊表达，两者结合最佳

---

## 代码文件

`code/examples/python/demo_04_intent.py` — 完整演示

---
*相关概念：NLU、槽位、实体识别、对话管理、意图路由、Agent工具调用*
