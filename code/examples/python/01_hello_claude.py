# 运行: python 01_hello_claude.py
# 依赖: pip install anthropic
# 说明: 这是学习 Claude API 的第一个文件，从最基础的调用开始
#       类比：就像你第一次学打电话——先学怎么拨号，再学怎么说话

import anthropic  # 导入 Anthropic 官方 SDK，类似 JS 里 import fetch
import asyncio    # Python 异步编程库，用来写 async/await 代码

# ============================================================
# 第一部分：最基础的同步调用（最简单的打招呼方式）
# ============================================================

def basic_hello():
    """
    最简单的 Claude 调用示例
    类比：就像发一条短信，等对方回复，再看回复内容
    """
    print("=" * 50)
    print("【示例1】基础文本调用")
    print("=" * 50)

    # 创建客户端，SDK 会自动读取环境变量 ANTHROPIC_API_KEY
    # 类比：就像打开微信，自动登录你的账号
    client = anthropic.Anthropic()

    # 发送一条消息给 Claude，等待完整回复
    # client.messages.create 是核心方法，类似 JS 的 fetch("/api/chat", {method:"POST"})
    message = client.messages.create(
        model="claude-opus-4-8",          # 指定使用哪个 Claude 模型（型号）
        max_tokens=1024,                   # 最多生成多少个 token（约等于字数上限）
        messages=[                         # 对话历史列表，每条消息是一个字典
            {
                "role": "user",            # role 是发言角色，"user" 表示用户在说话
                "content": "你好，请用一句话介绍你自己。"  # content 是具体说的话
            }
        ]
    )

    # 解析返回的 response 对象
    # message.content 是一个列表，每个元素是一个内容块（通常只有一个文本块）
    reply_text = message.content[0].text  # 取第一个内容块的 .text 属性

    print(f"Claude 回复：{reply_text}")
    print()

    # 解析更多响应字段——response 对象包含很多有用的信息
    print("【响应对象详解】")
    print(f"  stop_reason    = {message.stop_reason}")
    # stop_reason 表示 Claude 为什么停止生成
    # "end_turn" = 正常结束（把话说完了）
    # "max_tokens" = 达到 token 上限被截断了
    # "tool_use" = 要调用工具（见文件2）
    # "stop_sequence" = 遇到了预设的停止符

    print(f"  model          = {message.model}")
    # 实际使用的模型名称（有时和你传入的略有差异）

    print(f"  input_tokens   = {message.usage.input_tokens}")
    # 你发送的内容消耗了多少 token（token 类比：汉字约 1.5-2 个 token）

    print(f"  output_tokens  = {message.usage.output_tokens}")
    # Claude 回复消耗了多少 token

    print(f"  total_tokens   = {message.usage.input_tokens + message.usage.output_tokens}")
    # 本次对话总消耗（计费依据）

    print(f"  message.id     = {message.id}")
    # 每条消息都有唯一 ID，方便追踪和调试

    print()
    return reply_text


# ============================================================
# 第二部分：流式输出（streaming）
# ============================================================

def streaming_hello():
    """
    流式输出示例：Claude 边生成边返回，不用等全部生成完
    类比：就像看弹幕直播，字一个个出来，而不是等主播说完再一次性显示
    适合场景：回复很长时，用户不用干等，体验更好
    """
    print("=" * 50)
    print("【示例2】流式输出 Streaming")
    print("=" * 50)

    client = anthropic.Anthropic()

    print("Claude 正在回复（流式）：", end="", flush=True)

    # 使用 client.messages.stream() 上下文管理器开启流式模式
    # 类比：就像订阅一个直播频道，数据包来一个处理一个
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": "请用3句话描述今天的天气可能是什么样的，要有想象力。"
            }
        ]
    ) as stream:
        # stream.text_stream 是一个生成器，每次 yield 一小段文字
        # 类比：水龙头打开，水一滴滴流出来，我们接一滴处理一滴
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
            # end="" 表示不换行，flush=True 表示立即刷新到屏幕
            # 这样就能看到文字一个个出现的效果

    print("\n")  # 流结束后换行

    # 流结束后，可以获取完整的最终消息对象
    final_message = stream.get_final_message()
    print(f"  流式完成，总 tokens: {final_message.usage.input_tokens + final_message.usage.output_tokens}")
    print()


# ============================================================
# 第三部分：System Prompt（给 Claude 设定角色和规则）
# ============================================================

def system_prompt_demo():
    """
    System Prompt 示例：在对话开始前给 Claude 下"规矩"
    类比：就像你雇了一个新员工，先给他写一份岗位说明书
          "你是客服人员，只能回答产品相关问题，用礼貌语气"
    """
    print("=" * 50)
    print("【示例3】System Prompt 使用")
    print("=" * 50)

    client = anthropic.Anthropic()

    # system 参数放在 messages 列表之外，单独传入
    # 它会作为整个对话的"背景设定"，用户看不到，但 Claude 始终遵守
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=200,
        system=(
            "你是一位古代诗人，名叫李白。"
            "你只能用古诗词的风格回答问题。"
            "每句话必须押韵，不超过四行。"
        ),
        messages=[
            {
                "role": "user",
                "content": "今天天气真好，你心情怎么样？"
            }
        ]
    )

    print(f"（角色：李白）Claude 回答：\n{message.content[0].text}")
    print()


# ============================================================
# 第四部分：多轮对话（维护对话历史）
# ============================================================

def multi_turn_conversation():
    """
    多轮对话示例：Claude 没有自动记忆，每次都要把历史带上
    类比：就像给客服发邮件，每次都要 CC 之前的邮件链，
          对方才知道你们之前聊了什么
    """
    print("=" * 50)
    print("【示例4】多轮对话")
    print("=" * 50)

    client = anthropic.Anthropic()

    # messages 列表就是整个对话历史
    # 每次调用后，把 Claude 的回复也加进去，下次一起发送
    messages = []  # 从空对话开始

    # --- 第一轮 ---
    user_input_1 = "我叫小明，我喜欢吃饺子。"
    print(f"用户：{user_input_1}")

    messages.append({"role": "user", "content": user_input_1})  # 加入用户消息

    response_1 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        messages=messages  # 发送完整历史（目前只有1条）
    )

    assistant_reply_1 = response_1.content[0].text
    print(f"Claude：{assistant_reply_1}")

    # 把 Claude 的回复也加入历史，下一轮就能"记住"了
    messages.append({"role": "assistant", "content": assistant_reply_1})

    # --- 第二轮 ---
    user_input_2 = "你还记得我叫什么名字吗？我喜欢吃什么？"
    print(f"\n用户：{user_input_2}")

    messages.append({"role": "user", "content": user_input_2})

    response_2 = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        messages=messages  # 发送包含第一轮在内的完整历史（共3条）
    )

    assistant_reply_2 = response_2.content[0].text
    print(f"Claude：{assistant_reply_2}")

    # 打印当前对话历史长度
    messages.append({"role": "assistant", "content": assistant_reply_2})
    print(f"\n  当前对话历史共 {len(messages)} 条消息")
    print()


# ============================================================
# 第五部分：async 异步版本（与同步版本对比）
# ============================================================

async def async_hello():
    """
    async 异步版本：适合在 Web 服务器或同时处理多个请求的场景
    类比：同步版 = 去银行排队，一个人办完才轮到你
          异步版 = 拿号等叫号，等待时可以去别处办其他事
    什么时候用 async：
      - 你的程序同时要处理很多用户请求（Web API）
      - 你想同时调用多个 AI 接口不互相等待
    什么时候用同步：
      - 简单脚本、一次性任务、初学练习
    """
    print("=" * 50)
    print("【示例5】async 异步版本")
    print("=" * 50)

    # AsyncAnthropic 是异步版客户端，其他参数完全一样
    # 类比：AsyncAnthropic 是"异步模式银行"，Anthropic 是"普通排队银行"
    client = anthropic.AsyncAnthropic()

    # await 表示"在这里等一下，但不阻塞其他任务"
    # 类比：await = "等我外卖到了通知我，我先去做别的事"
    message = await client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "用一句话说明同步和异步的区别。"
            }
        ]
    )

    print(f"（异步）Claude 回复：{message.content[0].text}")
    print()


async def async_streaming_hello():
    """
    async 流式输出：异步 + 流式的组合，适合生产环境 Web 服务
    """
    print("=" * 50)
    print("【示例6】async 流式输出")
    print("=" * 50)

    client = anthropic.AsyncAnthropic()

    print("（异步流式）Claude 回复：", end="", flush=True)

    # async with 是异步版的 with，async for 是异步版的 for
    async with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": "用两句话介绍 Python。"
            }
        ]
    ) as stream:
        # async for 会异步等待每个 chunk 到来
        async for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

    print("\n")


# ============================================================
# 主程序入口
# ============================================================

def main():
    """
    主函数：依次运行所有示例
    """
    print("\n🚀 Claude API 基础示例开始运行\n")

    # 示例1：基础调用
    basic_hello()

    # 示例2：流式输出
    streaming_hello()

    # 示例3：System Prompt
    system_prompt_demo()

    # 示例4：多轮对话
    multi_turn_conversation()

    # 示例5 & 6：async 版本（需要用 asyncio.run 来运行 async 函数）
    # asyncio.run() 是启动 async 程序的"开关"，只能在最外层调用一次
    asyncio.run(async_hello())
    asyncio.run(async_streaming_hello())

    print("✅ 所有示例运行完毕！")
    print()
    print("学到了什么？")
    print("  1. client.messages.create() = 基础调用，等待完整回复")
    print("  2. client.messages.stream() = 流式调用，边生成边输出")
    print("  3. system= 参数 = 给 Claude 设定角色和规则")
    print("  4. messages 列表 = 手动维护对话历史（Claude 没有自动记忆）")
    print("  5. AsyncAnthropic = 异步版，适合并发场景")


if __name__ == "__main__":
    # 这是 Python 标准入口写法
    # 意思是：只有直接运行这个文件时才执行 main()
    # 如果被其他文件 import，则不自动执行
    main()
