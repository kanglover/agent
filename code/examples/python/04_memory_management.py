# 运行: python 04_memory_management.py
# 依赖: pip install anthropic
# 说明: 演示如何管理 Claude 对话的"记忆"（上下文窗口管理）
#
# 核心问题：Claude 没有持久记忆，每次调用都要把历史带上。
# 但历史越来越长 → token 越来越多 → 成本越来越高，速度越来越慢
# 甚至可能超出 context window 限制（Claude 的上下文窗口有上限）
#
# 类比：就像开会做会议记录
#   - 最初：记录每句话（完整历史）
#   - 会议太长：只保留最近20条记录（滑动窗口）
#   - 重要内容：单独贴便利贴，永远不删（pin）
#   - 旧内容太多：请秘书做一份摘要，替换掉原来的冗长记录（压缩摘要）

import anthropic    # Claude SDK
import json         # JSON 序列化/反序列化（用于文件存储）
import os           # 文件操作
import time         # 时间戳
from typing import Optional  # 类型提示（可选类型）
from dataclasses import dataclass, field, asdict  # 数据类工具


# ============================================================
# 第一部分：消息数据结构
# ============================================================

@dataclass
class Message:
    """
    单条消息的数据结构

    @dataclass 是 Python 3.7+ 的语法糖，
    自动生成 __init__、__repr__ 等方法
    类比：就像给消息贴了一张标签，记录它的所有属性
    """
    role: str           # "user" 或 "assistant"
    content: str        # 消息的文字内容
    timestamp: float    # 时间戳（Unix 时间，方便排序和显示）
    pinned: bool = False   # 是否被钉住（钉住的消息不会被压缩或删除）
    index: int = 0         # 在历史列表中的原始位置（用于调试）

    def to_api_format(self) -> dict:
        """
        转换为 Claude API 需要的格式
        API 只需要 role 和 content，其他字段不要
        """
        return {"role": self.role, "content": self.content}


# ============================================================
# 第二部分：ConversationMemory 类（核心）
# ============================================================

class ConversationMemory:
    """
    对话记忆管理器

    功能一览：
    ┌─────────────────────────────────────────────────┐
    │  add_message()      添加新消息                  │
    │  estimate_tokens()  估算当前 token 用量          │
    │  sliding_window()   只保留最近 N 条消息          │
    │  summarize_old()    用 AI 压缩旧消息为摘要       │
    │  pin_message()      标记重要消息（永不删除）     │
    │  save_to_file()     保存到 JSON 文件             │
    │  load_from_file()   从 JSON 文件恢复             │
    │  get_messages()     获取 API 格式的消息列表      │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        max_tokens_threshold: int = 4000,   # token 超过这个数就触发压缩
        verbose: bool = True                # 是否打印操作日志
    ):
        """
        初始化记忆管理器

        参数：
            model                 = 使用的 Claude 模型（摘要时会调用）
            max_tokens_threshold  = token 阈值，超过就压缩
            verbose               = 是否打印详细日志
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens_threshold = max_tokens_threshold
        self.verbose = verbose

        self.messages: list[Message] = []   # 消息列表（核心存储）
        self.summary: Optional[str] = None  # 历史摘要（压缩后的旧内容）
        self._msg_counter = 0               # 全局消息计数器（用于 index 字段）

    def _log(self, msg: str):
        """打印日志（仅在 verbose=True 时输出）"""
        if self.verbose:
            print(f"  [Memory] {msg}")

    # -------- 添加消息 --------

    def add_message(self, role: str, content: str) -> Message:
        """
        添加一条新消息到记忆

        参数：
            role    = "user" 或 "assistant"
            content = 消息内容
        返回：创建的 Message 对象

        示例：
            memory.add_message("user", "你好！")
            memory.add_message("assistant", "你好，有什么可以帮你？")
        """
        # 参数验证
        if role not in ("user", "assistant"):
            raise ValueError(f"role 必须是 'user' 或 'assistant'，收到: {role!r}")
        if not content or not content.strip():
            raise ValueError("content 不能为空")

        # 创建消息对象
        msg = Message(
            role=role,
            content=content.strip(),
            timestamp=time.time(),
            pinned=False,
            index=self._msg_counter
        )
        self._msg_counter += 1
        self.messages.append(msg)

        self._log(f"添加消息 [{role}] index={msg.index}，当前共 {len(self.messages)} 条")

        # 自动检查是否需要压缩（token 超出阈值时自动触发）
        estimated = self.estimate_tokens()
        if estimated > self.max_tokens_threshold:
            self._log(f"⚠️  估算 token（{estimated}）超过阈值（{self.max_tokens_threshold}），自动压缩...")
            self.summarize_old_messages()

        return msg

    # -------- Token 估算 --------

    def estimate_tokens(self) -> int:
        """
        粗略估算当前消息列表的 token 数量

        为什么是"粗略"估算？
        因为精确的 tokenize 需要调用 API（慢且有成本）
        粗略规则：1 个英文单词 ≈ 1.3 个 token，1 个汉字 ≈ 1.5 个 token
        实践中用"字符数 / 4"作为英文的快速估算

        类比：就像估算一篇文章的字数，数字不精确但够用
        """
        total_chars = 0

        # 如果有摘要，也要算进去
        if self.summary:
            total_chars += len(self.summary)

        # 累加所有消息的字符数
        for msg in self.messages:
            total_chars += len(msg.content)
            # 角色标签本身也占一点 token（约 4 个）
            total_chars += 10

        # 估算公式：中文占比高时用 / 2，英文占比高时用 / 4
        # 简单起见，统一用 / 3 作为中英混合的估算
        estimated_tokens = total_chars // 3

        self._log(f"估算 token: {estimated_tokens}（共 {total_chars} 字符）")
        return estimated_tokens

    # -------- 滑动窗口 --------

    def sliding_window(self, max_messages: int = 20):
        """
        滑动窗口：只保留最近 max_messages 条消息
        被删除的消息（不包括 pinned）会消失

        类比：就像手机相册设置了容量上限，
              超过了就自动删掉最老的照片（除非你收藏了）

        参数：max_messages = 最多保留的消息数量
        """
        self._log(f"滑动窗口：保留最近 {max_messages} 条（当前 {len(self.messages)} 条）")

        if len(self.messages) <= max_messages:
            self._log("消息数未超限，无需裁剪")
            return

        # 分离钉住的和未钉住的消息
        pinned = [m for m in self.messages if m.pinned]
        unpinned = [m for m in self.messages if not m.pinned]

        # 从未钉住的消息中，只保留最近 max_messages 条
        # max(0, ...) 防止负数导致切片行为异常
        keep_count = max(0, max_messages - len(pinned))
        kept_unpinned = unpinned[-keep_count:] if keep_count > 0 else []

        removed_count = len(unpinned) - len(kept_unpinned)

        # 重新拼合：钉住的 + 保留的未钉住的，按原始 index 排序
        self.messages = sorted(pinned + kept_unpinned, key=lambda m: m.index)

        self._log(f"删除了 {removed_count} 条旧消息，保留 {len(self.messages)} 条（其中 {len(pinned)} 条已钉住）")

    # -------- 摘要压缩 --------

    def summarize_old_messages(self, keep_recent: int = 6):
        """
        用 Claude 把旧消息压缩成摘要，然后替换掉原来的旧消息
        这样可以保留历史语义，同时大幅减少 token 占用

        类比：就像把一本厚厚的日记压缩成一页精华摘要，
              扔掉日记原文，只留摘要

        参数：keep_recent = 保留最近几条消息不压缩（保持对话连贯性）
        """
        if len(self.messages) <= keep_recent:
            self._log("消息太少，无需摘要")
            return

        # 分离：要压缩的"旧消息"和要保留的"近期消息"
        pinned = [m for m in self.messages if m.pinned]
        unpinned = [m for m in self.messages if not m.pinned]

        # 近期的不压缩，旧的才压缩
        # 类比：最近的会议记录留着，上个月的才归档
        old_to_compress = unpinned[:-keep_recent] if len(unpinned) > keep_recent else []
        recent_to_keep = unpinned[-keep_recent:]

        if not old_to_compress:
            self._log("没有足够的旧消息可以压缩")
            return

        self._log(f"准备压缩 {len(old_to_compress)} 条旧消息（保留最近 {keep_recent} 条 + {len(pinned)} 条钉住的）")

        # 把要压缩的消息格式化成文本
        history_text = ""
        for msg in old_to_compress:
            history_text += f"[{msg.role.upper()}]: {msg.content}\n\n"

        # 调用 Claude 生成摘要
        # 这里用一个独立的 messages.create 调用，不影响当前对话
        try:
            summary_response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "请对以下对话历史做简洁的摘要。"
                            "重点保留：用户的主要需求、达成的结论、重要事实。"
                            "摘要控制在 200 字以内，用第三人称描述。\n\n"
                            f"对话历史：\n{history_text}"
                        )
                    }
                ]
            )
            new_summary = summary_response.content[0].text

        except Exception as e:
            self._log(f"摘要生成失败（{e}），跳过压缩")
            return

        # 如果已经有旧摘要，把新摘要和旧摘要合并
        if self.summary:
            combined_summary = (
                f"【更早期的历史摘要】\n{self.summary}\n\n"
                f"【近期历史摘要】\n{new_summary}"
            )
            self.summary = combined_summary
        else:
            self.summary = new_summary

        # 保留：钉住的消息 + 近期消息
        self.messages = sorted(pinned + recent_to_keep, key=lambda m: m.index)

        self._log(f"摘要完成！压缩了 {len(old_to_compress)} 条 → {len(self.summary)} 字摘要")
        self._log(f"当前剩余消息：{len(self.messages)} 条")

    # -------- 钉住消息 --------

    def pin_message(self, index: int):
        """
        钉住指定位置的消息，使其不被滑动窗口或压缩删除

        类比：就像在便利贴上按了一颗图钉，
              清桌子时贴纸能扔，图钉贴的不能扔

        参数：index = 消息在 self.messages 列表中的位置（0 开头）
        """
        if index < 0 or index >= len(self.messages):
            raise IndexError(f"index {index} 超出范围（0 ~ {len(self.messages)-1}）")

        msg = self.messages[index]
        if msg.pinned:
            self._log(f"消息 index={index} 已经是钉住状态")
            return

        msg.pinned = True
        preview = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        self._log(f"📌 钉住消息 index={index}: [{msg.role}] {preview!r}")

    def unpin_message(self, index: int):
        """
        取消钉住指定消息

        参数：index = 消息在 self.messages 列表中的位置
        """
        if index < 0 or index >= len(self.messages):
            raise IndexError(f"index {index} 超出范围")

        self.messages[index].pinned = False
        self._log(f"取消钉住消息 index={index}")

    # -------- 获取 API 格式消息 --------

    def get_messages(self) -> list[dict]:
        """
        获取用于传给 Claude API 的消息列表
        如果有摘要，会把摘要作为最开头的 user+assistant 消息对插入

        类比：就像把会议记录整理成一份完整的报告，
              先放摘要，再放近期详细记录

        返回：[{"role": "user", "content": "..."}, ...]
        """
        result = []

        # 如果有历史摘要，插入到消息最前面
        # 用一个虚构的"用户请求摘要/AI回应"消息对来注入
        if self.summary:
            result.append({
                "role": "user",
                "content": "[系统提示：以下是之前对话的摘要，请参考]\n" + self.summary
            })
            result.append({
                "role": "assistant",
                "content": "好的，我已了解之前的对话背景。"
            })

        # 加入当前保留的消息
        for msg in self.messages:
            result.append(msg.to_api_format())

        return result

    # -------- 持久化（存/取文件）--------

    def save_to_file(self, path: str):
        """
        把当前记忆状态保存到 JSON 文件

        类比：就像把会议记录拍照存档，下次会议前可以调出来看

        参数：path = 文件保存路径，例如 "/tmp/memory.json"
        """
        data = {
            "summary": self.summary,
            "messages": [asdict(m) for m in self.messages],
            "msg_counter": self._msg_counter,
            "saved_at": time.time()
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._log(f"💾 已保存到 {path}（{len(self.messages)} 条消息）")

    def load_from_file(self, path: str):
        """
        从 JSON 文件恢复记忆状态
        会覆盖当前内存中的数据

        参数：path = JSON 文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在：{path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.summary = data.get("summary")
        self._msg_counter = data.get("msg_counter", 0)

        # 把字典列表重新转换为 Message 对象
        raw_messages = data.get("messages", [])
        self.messages = [Message(**m) for m in raw_messages]

        self._log(f"📂 已从 {path} 恢复（{len(self.messages)} 条消息）")

    # -------- 状态打印 --------

    def print_status(self):
        """
        打印当前记忆状态的概览，方便调试
        """
        print(f"\n{'─'*50}")
        print(f"📊 记忆状态概览")
        print(f"{'─'*50}")
        print(f"  消息总数    : {len(self.messages)}")
        print(f"  钉住数量    : {sum(1 for m in self.messages if m.pinned)}")
        print(f"  估算 token  : {self.estimate_tokens()}")
        print(f"  是否有摘要  : {'是（' + str(len(self.summary)) + '字）' if self.summary else '否'}")

        if self.messages:
            print(f"\n  最近3条消息：")
            for msg in self.messages[-3:]:
                pin_mark = "📌" if msg.pinned else "  "
                preview = msg.content[:60].replace("\n", " ")
                print(f"  {pin_mark} [{msg.role:9}] {preview}")
        print(f"{'─'*50}\n")


# ============================================================
# 第三部分：演示主程序
# ============================================================

def simulate_long_conversation(memory: ConversationMemory, turns: int = 15):
    """
    模拟一段长对话，观察记忆管理的效果

    参数：
        memory = ConversationMemory 实例
        turns  = 模拟的对话轮数
    """
    # 模拟的对话内容（用户消息和 AI 回复交替）
    mock_exchanges = [
        ("我想学 Python，从哪里开始？", "建议从官方教程开始，先学基础语法，再学数据结构。"),
        ("函数怎么写？", "用 def 关键字定义函数，例如：def greet(name): return f'Hello {name}'"),
        ("什么是列表？", "列表是有序的数据集合，用方括号 [] 表示，例如：fruits = ['苹果', '香蕉']"),
        ("字典有什么用？", "字典存储键值对，用花括号 {}，例如：person = {'name': '张三', 'age': 25}"),
        ("循环怎么用？", "for 循环遍历序列，while 循环按条件执行，最常用的是 for item in list。"),
        ("怎么读取文件？", "用 open() 函数，推荐 with open('file.txt', 'r') as f: content = f.read()"),
        ("什么是异常处理？", "用 try/except 捕获错误，避免程序崩溃：try: ... except Exception as e: print(e)"),
        ("面向对象是什么？", "OOP 用 class 定义对象，对象包含属性（数据）和方法（函数）。"),
        ("模块怎么导入？", "用 import 语句，例如：import os 或 from datetime import datetime"),
        ("pip 是什么？", "pip 是 Python 的包管理器，用来安装第三方库，例如：pip install requests"),
        ("虚拟环境有什么用？", "venv 隔离不同项目的依赖，避免版本冲突，用 python -m venv env 创建。"),
        ("什么是装饰器？", "装饰器用 @ 符号，在函数前添加额外功能，是 Python 的高级特性。"),
        ("async/await 怎么理解？", "异步编程让程序等待 IO 时去做其他事，asyncio 是标准库实现。"),
        ("怎么调试代码？", "用 print() 是最简单的调试方法，pdb 是内置调试器，VSCode 有图形调试界面。"),
        ("如何部署 Python 应用？", "常见方式：Docker 容器化、serverless 函数、或直接部署到云服务器。"),
    ]

    print(f"\n模拟 {turns} 轮对话...\n")

    for i in range(min(turns, len(mock_exchanges))):
        user_msg, ai_msg = mock_exchanges[i]

        print(f"轮次 {i+1}:")
        print(f"  用户: {user_msg[:50]}")

        # 添加用户消息
        memory.add_message("user", user_msg)

        # 如果是第3条，钉住它（演示 pin 功能）
        if i == 2:
            print("  → 钉住这条消息（重要概念）")
            memory.pin_message(len(memory.messages) - 1)

        # 添加 AI 回复
        memory.add_message("assistant", ai_msg)

        # 每5轮打印一次状态
        if (i + 1) % 5 == 0:
            memory.print_status()


def demo_save_load():
    """
    演示保存和加载记忆
    """
    print("\n" + "="*60)
    print("【演示】保存与加载记忆")
    print("="*60)

    # 创建新记忆并添加几条消息
    memory = ConversationMemory(verbose=True)
    memory.add_message("user", "我叫小明，我在学 Python。")
    memory.add_message("assistant", "你好小明！Python 是很好的编程语言。")
    memory.add_message("user", "我已经学了一个月了。")
    memory.add_message("assistant", "一个月进步很大！你学到什么了？")

    # 钉住第一条（重要信息）
    memory.pin_message(0)

    # 保存到文件
    save_path = "/tmp/demo_memory.json"
    memory.save_to_file(save_path)

    # 创建新的 memory 对象，从文件恢复
    new_memory = ConversationMemory(verbose=True)
    new_memory.load_from_file(save_path)

    print(f"\n恢复后的消息数：{len(new_memory.messages)}")
    print(f"第一条消息是否钉住：{new_memory.messages[0].pinned}")
    print("✅ 存取功能正常！")


def main():
    """
    主演示程序
    """
    print("🧠 对话记忆管理演示\n")

    # ---- 演示1：基础记忆管理 + 长对话压缩 ----
    print("="*60)
    print("【演示1】长对话记忆管理")
    print("="*60)

    # max_tokens_threshold=300 是故意调小，让压缩更快触发（便于演示）
    # 实际使用建议设 50000 以上
    memory = ConversationMemory(
        verbose=True,
        max_tokens_threshold=300
    )

    simulate_long_conversation(memory, turns=10)

    print("\n最终状态：")
    memory.print_status()

    # 展示摘要内容
    if memory.summary:
        print(f"📝 历史摘要内容：\n{memory.summary[:300]}...")

    # ---- 演示2：手动触发滑动窗口 ----
    print("\n" + "="*60)
    print("【演示2】手动滑动窗口")
    print("="*60)

    mem2 = ConversationMemory(verbose=True, max_tokens_threshold=99999)

    # 快速添加10条消息
    for i in range(10):
        mem2.add_message("user", f"这是第 {i+1} 条用户消息，内容是测试数据 {i*100}")
        mem2.add_message("assistant", f"这是第 {i+1} 条AI回复，已收到你的第 {i+1} 条消息。")

    print(f"添加后：{len(mem2.messages)} 条消息")

    # 钉住第1条
    mem2.pin_message(0)
    print("钉住第0条消息")

    # 触发滑动窗口，只保留最近6条
    mem2.sliding_window(max_messages=6)
    print(f"滑动窗口后：{len(mem2.messages)} 条消息（第0条被钉住，会保留）")

    # ---- 演示3：保存和加载 ----
    demo_save_load()

    print("\n✅ 所有演示完成！")
    print("\n核心要点：")
    print("  1. Claude 无自动记忆，每次要手动带上 messages 列表")
    print("  2. sliding_window = 直接截断旧消息（快速，但丢失信息）")
    print("  3. summarize_old = 用AI压缩旧消息（慢但保留语义）")
    print("  4. pin_message = 钉住重要消息，不会被任何操作删除")
    print("  5. save/load = JSON持久化，程序重启后恢复上下文")


if __name__ == "__main__":
    main()
