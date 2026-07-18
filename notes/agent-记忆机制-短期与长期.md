# Agent 记忆机制：短期记忆与长期记忆

> 沉淀时间：2026-07-06

## 一句话总结

短期记忆 = 对话中的 messages 列表；长期记忆 = 写进磁盘/向量库，重启也不丢。

## 四种记忆类型

| 类型 | 类比 | 技术实现 |
|------|------|---------|
| 短期记忆 | 当前这轮对话 | Context Window（messages 列表） |
| 长期记忆 | 记住老朋友信息 | 写入数据库/向量库 |
| 情节记忆 | 记得"上周做了啥" | 对话历史摘要 |
| 程序记忆 | 会骑车（不用想） | 模型权重（训练进去的） |

工程中最常用：**短期记忆** + **长期记忆**。

---

## 短期记忆

### 是什么

就是 `messages` 列表，存在内存里，程序重启就消失。

```python
messages = [
    {"role": "user",      "content": "我叫小明"},
    {"role": "assistant", "content": "好的，小明！"},
    {"role": "user",      "content": "我叫什么名字？"},  # ← 靠历史"记住"
]
```

### 核心痛点

Context window 有限，对话越来越长 → token 越来越多 → 成本上升，速度变慢，甚至超限。

### 解法：摘要压缩

超过阈值 → 把早期对话总结为摘要 → 注入 system prompt 开头：

```python
def _compress(self):
    old_msgs = self.messages[:cutoff]
    summary = llm(f"总结以下对话的关键信息：{old_msgs}")

    # 用摘要替换原始消息（节省 token）
    self.messages = [
        {"role": "system", "content": f"[历史摘要] {summary}"}
    ] + self.messages[cutoff:]
```

**效果验证：** 第4轮触发压缩后，第6轮问"我叫什么名字"，模型从摘要里找到了"小明"——历史没丢，token 省了一半。

### 生命周期

```
会话开始 → 创建 messages 列表
每轮对话 → append 进去
会话结束 → 内存清空（消失！）
          → 如需保留：主动存入长期记忆
```

---

## 长期记忆

### 是什么

存在磁盘，程序重启后记忆依然存在。分两种：

| 存储类型 | 适合存什么 | 查找方式 |
|---------|-----------|---------|
| JSON / 关系数据库 | 姓名、偏好、结构化事实 | 精确键值查找 O(1) |
| 向量数据库（ChromaDB） | 对话片段、知识内容 | 语义相似度检索 |

### 存储

```python
# 精确事实 → JSON
memory.remember_fact("name", "小明")  # → 写入磁盘 JSON

# 语义内容 → 向量库
memory.remember_episode(
    "用户问：什么是RAG？回答：RAG是检索增强生成...",
    tags=["RAG", "技术"]
)  # → 向量化后存入 ChromaDB
```

### 调用（新会话开始时）

```python
def build_context(self, current_query: str) -> str:
    # ① 加载结构化事实
    facts = self.profile["facts"]  # {"name": "小明", "goal": "学AI"}

    # ② 语义检索相关历史
    memories = self.recall_semantic(current_query, top_k=3)
    # 返回与当前问题最相关的历史对话片段

    # ③ 拼成 system prompt 注入
    return f"用户信息：{facts}\n相关历史：{memories}"
```

### 效果验证

检索"向量数据库怎么用"：
```
相似度=0.519 → "用户对ChromaDB很感兴趣..." ← 精准命中！
相似度=0.015 → "Python适合做AI..."           ← 低分，不相关
```

---

## 短期 + 长期协同工作完整流程

```
新会话开始
    ↓
① 从长期记忆加载用户画像（JSON）
   "用户叫小明，目标是学AI"
    ↓
② 语义检索相关历史（向量库）
   "发现3条相关对话片段"
    ↓
③ 全部注入 System Prompt → 成为短期记忆的一部分
    ↓
④ 对话进行中：短期记忆实时 append
    ↓
⑤ 超过阈值：短期记忆触发压缩
    ↓
⑥ 会话结束：提炼重要信息 → 写入长期记忆
```

---

## 代码文件

`code/examples/python/demo_01_short_memory.py` — 短期记忆完整演示
`code/examples/python/demo_02_long_memory.py` — 长期记忆（JSON + TF-IDF向量）演示

---
*相关概念：Context Window、摘要压缩、向量数据库、ChromaDB、TF-IDF*
