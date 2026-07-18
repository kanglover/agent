"""
07_streaming.py — Claude Streaming 完整指南

什么是 Streaming（流式输出）？
────────────────────────────
普通模式：等 Claude 把完整回答生成完，一次性返回给你。
Streaming 模式：Claude 生成一个词就立刻推送给你，像打字机一样实时显示。

为什么需要 Streaming？
1. 用户体验：不用盯着空白屏幕等 10 秒，可以边看边读。
2. 长文本场景：生成长文章时，等待时间可能很长，streaming 让用户感觉更流畅。
3. Web 应用：配合 SSE（Server-Sent Events）可以实现网页实时打字效果。
"""

import json
import time
import signal
import sys
import anthropic

# ─────────────────────────────────────────────────────────────
# 第一部分：基础 Streaming
# ─────────────────────────────────────────────────────────────

def demo_basic_streaming() -> None:
    """
    最基础的 streaming 用法：with client.messages.stream() as stream:
    text_stream 迭代器每次 yield 一小段文字（token 级别）。
    """
    print("\n" + "=" * 60)
    print("【演示 1】基础 Streaming — 实时打印文字")
    print("=" * 60)

    client = anthropic.Anthropic()

    print("\nClaude 正在生成（实时）：\n")

    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "请用3句话介绍 Python 语言，每句话之间换行。",
            }
        ],
    ) as stream:
        # text_stream 是一个迭代器，每次 yield 一小段新文字
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)   # end="" 不换行，flush=True 立即输出
        print()   # 最后换行

    # 流结束后，可以获取完整的最终消息
    final_message = stream.get_final_message()
    print(f"\n[完成] 总共生成 {final_message.usage.output_tokens} tokens")


# ─────────────────────────────────────────────────────────────
# 第二部分：打印所有事件类型
#
# Streaming 底层是一个事件流，包含多种事件类型：
#   message_start        : 消息开始，包含 input_tokens 信息
#   content_block_start  : 一个内容块开始（可能是文字块或工具调用块）
#   content_block_delta  : 内容块的增量（新增的一小段文字或 JSON）
#   content_block_stop   : 一个内容块结束
#   message_delta        : 消息级别的更新（stop_reason、output_tokens）
#   message_stop         : 整个消息结束
# ─────────────────────────────────────────────────────────────

def demo_all_events() -> None:
    """
    展示底层事件流，打印每个事件的类型和内容。
    适合调试和深入理解 streaming 协议。
    """
    print("\n" + "=" * 60)
    print("【演示 2】打印所有 Streaming 事件类型")
    print("=" * 60)

    client = anthropic.Anthropic()

    with client.messages.stream(
        model="claude-haiku-4-5",   # 用轻量模型，节省演示成本
        max_tokens=64,
        messages=[{"role": "user", "content": "一句话回答：1+1等于几？"}],
    ) as stream:
        for event in stream:
            event_type = event.type

            if event_type == "message_start":
                # 消息开始：包含初始的 usage 信息（此时 output_tokens 还是 0）
                print(f"[message_start] input_tokens={event.message.usage.input_tokens}")

            elif event_type == "content_block_start":
                # 内容块开始：index 是块编号（从 0 起），type 是 text 或 tool_use
                block = event.content_block
                print(f"[content_block_start] index={event.index}, type={block.type}")

            elif event_type == "content_block_delta":
                # 增量更新：最频繁的事件，每次包含一小段新内容
                delta = event.delta
                if delta.type == "text_delta":
                    print(f"[content_block_delta] text='{delta.text}'")
                elif delta.type == "input_json_delta":
                    # 工具调用参数的增量（JSON 片段）
                    print(f"[content_block_delta] json_partial='{delta.partial_json}'")

            elif event_type == "content_block_stop":
                # 内容块结束
                print(f"[content_block_stop] index={event.index}")

            elif event_type == "message_delta":
                # 消息级别更新：stop_reason（end_turn/tool_use/max_tokens）和最终 token 数
                print(f"[message_delta] stop_reason={event.delta.stop_reason}, "
                      f"output_tokens={event.usage.output_tokens}")

            elif event_type == "message_stop":
                # 消息彻底结束
                print(f"[message_stop] 流式传输完毕")


# ─────────────────────────────────────────────────────────────
# 第三部分：带工具调用的 Streaming
#
# 当 Claude 决定调用工具时，content_block_delta 里的类型是
# input_json_delta，包含工具参数 JSON 的片段。
# 需要自己把这些片段拼起来，等 content_block_stop 后再解析 JSON。
# ─────────────────────────────────────────────────────────────

def demo_streaming_with_tools() -> None:
    """
    带工具调用的 Streaming 演示：
    累积 tool_use block 的 JSON 片段，在 block 结束时解析。
    """
    print("\n" + "=" * 60)
    print("【演示 3】带工具调用的 Streaming（累积 JSON 片段）")
    print("=" * 60)

    client = anthropic.Anthropic()

    # 定义一个简单的计算器工具
    tools = [
        {
            "name": "calculate",
            "description": "执行数学计算",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        }
    ]

    # 用于累积工具调用 JSON 的字典
    # key: block_index, value: {"name": ..., "json_str": ...}
    tool_calls_accumulator: dict[int, dict] = {}
    current_text = ""

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=256,
        tools=tools,
        messages=[{"role": "user", "content": "请计算 (15 + 7) * 3 的结果"}],
    ) as stream:
        for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    # 工具调用块开始：记录工具名称
                    tool_calls_accumulator[event.index] = {
                        "name": block.name,
                        "json_str": "",   # 用于累积 JSON 片段
                    }
                    print(f"\n[工具调用开始] 工具名: {block.name}")

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    current_text += delta.text
                    print(delta.text, end="", flush=True)
                elif delta.type == "input_json_delta":
                    # 累积 JSON 片段（此时 JSON 是不完整的，不能直接解析）
                    idx = event.index
                    if idx in tool_calls_accumulator:
                        tool_calls_accumulator[idx]["json_str"] += delta.partial_json
                        print(f".", end="", flush=True)   # 用点表示接收到 JSON 片段

            elif event_type == "content_block_stop":
                idx = event.index
                if idx in tool_calls_accumulator:
                    # 块结束后，JSON 已完整，可以解析了
                    tool_info = tool_calls_accumulator[idx]
                    try:
                        tool_input = json.loads(tool_info["json_str"])
                        print(f"\n[工具参数解析完成] {tool_info['name']}({tool_input})")

                        # 模拟执行工具
                        if tool_info["name"] == "calculate":
                            expr = tool_input.get("expression", "")
                            try:
                                # 安全起见只允许数字和基本运算符
                                safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expr) if __import__("re").sub else expr
                                result = eval(expr)   # 演示用，生产环境不要用 eval
                                print(f"[工具执行结果] {expr} = {result}")
                            except Exception:
                                print("[工具执行失败]")
                    except json.JSONDecodeError as e:
                        print(f"\n[JSON 解析失败] {e}")

    print(f"\n[流式工具调用演示完成]")


# ─────────────────────────────────────────────────────────────
# 第四部分：实时显示 Token 计数
# ─────────────────────────────────────────────────────────────

def demo_realtime_token_count() -> None:
    """
    在 streaming 过程中实时追踪 token 消耗。
    message_start 提供 input_tokens，message_delta 提供 output_tokens。
    """
    print("\n" + "=" * 60)
    print("【演示 4】实时显示 Token 计数")
    print("=" * 60)

    client = anthropic.Anthropic()

    input_tokens = 0
    output_tokens = 0
    char_count = 0
    start_time = time.time()

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{"role": "user", "content": "请用100字介绍 token 是什么意思"}],
    ) as stream:
        for event in stream:
            if event.type == "message_start":
                input_tokens = event.message.usage.input_tokens
                print(f"[开始] 输入 tokens: {input_tokens}")
                print("生成中：", end="", flush=True)

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    text = event.delta.text
                    print(text, end="", flush=True)
                    char_count += len(text)

            elif event.type == "message_delta":
                output_tokens = event.usage.output_tokens

    elapsed = time.time() - start_time
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0
    print(f"\n\n[统计]")
    print(f"  输入 tokens : {input_tokens}")
    print(f"  输出 tokens : {output_tokens}")
    print(f"  输出字符数  : {char_count}")
    print(f"  耗时        : {elapsed:.2f}s")
    print(f"  速度        : {tokens_per_sec:.1f} tokens/s")


# ─────────────────────────────────────────────────────────────
# 第五部分：KeyboardInterrupt 优雅中断
#
# 用户按下 Ctrl+C 时，stream 应该干净地关闭，
# 不抛出难看的 traceback，并保留已生成的部分内容。
# ─────────────────────────────────────────────────────────────

def demo_graceful_interrupt() -> None:
    """
    演示如何优雅地处理 Ctrl+C 中断。
    按 Ctrl+C 后，程序会打印已收到的内容并退出，不报错。
    """
    print("\n" + "=" * 60)
    print("【演示 5】KeyboardInterrupt 优雅中断")
    print("（生成过程中按 Ctrl+C 试试，或等它自动完成）")
    print("=" * 60)

    client = anthropic.Anthropic()
    received_text = []

    try:
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": "请写一首关于 AI 的七言律诗，要有完整的起承转合。",
                }
            ],
        ) as stream:
            print("\n生成中（按 Ctrl+C 可中断）：\n")
            for text_chunk in stream.text_stream:
                print(text_chunk, end="", flush=True)
                received_text.append(text_chunk)

    except KeyboardInterrupt:
        # 捕获中断信号，打印已收到的内容而不崩溃
        print("\n\n[用户中断]")
    except anthropic.APIError as e:
        print(f"\n[API 错误] {e}")
    finally:
        # finally 块无论是否中断都会执行
        if received_text:
            partial = "".join(received_text)
            print(f"[已接收 {len(partial)} 个字符，{len(received_text)} 个片段]")


# ─────────────────────────────────────────────────────────────
# 第六部分：FastAPI SSE 接口示例（注释形式，仅参考）
#
# SSE = Server-Sent Events，服务器向浏览器推送实时数据的标准协议。
# 配合 FastAPI 可以构建网页版的流式聊天界面。
# ─────────────────────────────────────────────────────────────

FASTAPI_EXAMPLE = '''
# ── 以下代码需要 pip install fastapi uvicorn anthropic ──
#
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import anthropic, json
#
# app = FastAPI()
# client = anthropic.Anthropic()
#
# @app.post("/chat/stream")
# async def chat_stream(request: dict):
#     """SSE 流式接口：每生成一个 token 就推送一条 SSE 事件"""
#     question = request.get("question", "")
#
#     def generate():
#         # SSE 格式：每行以 "data: " 开头，以 "\\n\\n" 结尾
#         with client.messages.stream(
#             model="claude-opus-4-5",
#             max_tokens=1024,
#             messages=[{"role": "user", "content": question}],
#         ) as stream:
#             for text in stream.text_stream:
#                 # 把每个 token 包装成 SSE 事件发送给浏览器
#                 data = json.dumps({"type": "delta", "text": text})
#                 yield f"data: {data}\\n\\n"
#
#             # 发送结束信号
#             usage = stream.get_final_message().usage
#             end_data = json.dumps({
#                 "type": "done",
#                 "input_tokens": usage.input_tokens,
#                 "output_tokens": usage.output_tokens,
#             })
#             yield f"data: {end_data}\\n\\n"
#
#     return StreamingResponse(generate(), media_type="text/event-stream")
#
# # 启动命令：uvicorn 07_streaming:app --reload
'''

# ─────────────────────────────────────────────────────────────
# 第七部分：前端 EventSource 消费代码（JS 注释）
#
# 浏览器端用 EventSource API 消费 SSE 流。
# ─────────────────────────────────────────────────────────────

FRONTEND_EVENTSOURCE_EXAMPLE = '''
/*
 * 前端 JavaScript：用 EventSource 消费后端 SSE 流
 *
 * // 向后端发 POST 请求，并用 fetch + ReadableStream 读取 SSE
 * async function chatStream(question) {
 *   const response = await fetch('/chat/stream', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ question }),
 *   });
 *
 *   const reader = response.body.getReader();
 *   const decoder = new TextDecoder();
 *   let buffer = '';
 *
 *   while (true) {
 *     const { done, value } = await reader.read();
 *     if (done) break;
 *
 *     buffer += decoder.decode(value, { stream: true });
 *     const lines = buffer.split('\n\n');
 *     buffer = lines.pop();  // 保留未完整的最后一行
 *
 *     for (const line of lines) {
 *       if (!line.startsWith('data: ')) continue;
 *       const data = JSON.parse(line.slice(6));
 *
 *       if (data.type === 'delta') {
 *         // 实时追加文字到页面
 *         document.getElementById('output').textContent += data.text;
 *       } else if (data.type === 'done') {
 *         console.log('完成！消耗 tokens:', data.input_tokens, '+', data.output_tokens);
 *       }
 *     }
 *   }
 * }
 *
 * chatStream('请介绍一下 Claude');
 */
'''


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("07_streaming.py — Claude Streaming 完整演示")
    print("=" * 60)

    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("\n未设置 ANTHROPIC_API_KEY，无法运行演示。")
        print("请先执行：export ANTHROPIC_API_KEY='your-key'")
        print("\n以下是本文件包含的演示（设置 key 后可运行）：")
        print("  1. 基础 Streaming — 实时打印文字")
        print("  2. 打印所有事件类型（message_start / content_block_delta / ...）")
        print("  3. 带工具调用的 Streaming（累积 JSON 片段）")
        print("  4. 实时显示 Token 计数与速度")
        print("  5. KeyboardInterrupt 优雅中断（Ctrl+C）")
        print("\n注释中还包含：")
        print("  - FastAPI SSE 接口示例")
        print("  - 前端 EventSource 消费代码（JavaScript）")
        return

    # 依次运行各演示
    demo_basic_streaming()

    input("\n按 Enter 继续下一个演示...")
    demo_all_events()

    input("\n按 Enter 继续下一个演示...")
    demo_streaming_with_tools()

    input("\n按 Enter 继续下一个演示...")
    demo_realtime_token_count()

    input("\n按 Enter 继续下一个演示（按 Ctrl+C 可中断流）...")
    demo_graceful_interrupt()

    print("\n" + "=" * 60)
    print("所有 Streaming 演示完成！")
    print("=" * 60)
    print("\n扩展阅读（见文件注释）：")
    print("  FASTAPI_EXAMPLE        — FastAPI SSE 接口示例")
    print("  FRONTEND_EVENTSOURCE_EXAMPLE — 前端 JS 代码")


import re   # 在文件顶层导入（给 demo_streaming_with_tools 用）

if __name__ == "__main__":
    main()
