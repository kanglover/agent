"""
MCP Client 示例 —— 连接 Server，自动发现并调用工具

这个脚本演示 MCP 的完整流程：
  1. 启动 Server 子进程，通过 stdio 跟它建立连接
  2. 问 Server「你有哪些工具？」（动态发现）
  3. 逐个调用工具，打印每一步的结果

运行方式：
  cd code/mcp_demo
  python client.py

你会看到 Client 和 Server 之间完整的「一问一答」过程。
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ===== 分隔线小工具，让输出更好看 =====
def section(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


async def main():
    # ---- 第 1 步：告诉 Client 怎么启动 Server ----
    # stdio 模式下，Client 会把 Server 当成一个子进程拉起来
    # 通过「标准输入 / 标准输出」管道来通信（不走网络）
    # 用 sys.executable 拿到当前 Python 的完整路径
    # 这样不管在什么环境（venv、conda），都能正确启动 Server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
    )

    section("第 1 步：连接 Server（启动子进程）")
    print(f"命令：{sys.executable} server.py")

    # stdio_client 会启动 Server 子进程，返回读写两个管道
    async with stdio_client(server_params) as (read_stream, write_stream):
        # ClientSession 负责按 MCP 协议收发消息
        async with ClientSession(read_stream, write_stream) as session:
            # ---- 第 2 步：初始化握手 ----
            # Client 和 Server 互相打招呼，确认协议版本和能力
            await session.initialize()
            print("握手成功，连接已建立。")

            # ---- 第 3 步：动态发现工具 ----
            # 这一步是 MCP 的核心能力：
            # Client 不需要提前知道 Server 有哪些工具，直接问就行
            section("第 2 步：列出 Server 拥有的工具")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            print(f"\n共发现 {len(tools_result.tools)} 个工具。")

            # ---- 第 4 步：调用 add_note 添加几条笔记 ----
            section("第 3 步：调用 add_note 添加笔记")
            notes_to_add = [
                ("MCP 入门", "MCP 是 AI 世界的 USB 接口，统一了工具连接方式。"),
                ("今天买菜", "西红柿、鸡蛋、面条，预算 30 元。"),
                ("Python 技巧", "用装饰器可以给函数加功能，不用改原代码。"),
            ]
            for title, content in notes_to_add:
                result = await session.call_tool(
                    "add_note",
                    {"title": title, "content": content},
                )
                print(f"  添加「{title}」→ {result.content[0].text}")

            # ---- 第 5 步：调用 list_notes 列出所有笔记 ----
            section("第 4 步：调用 list_notes 列出全部笔记")
            result = await session.call_tool("list_notes", {})
            print(result.content[0].text)

            # ---- 第 6 步：调用 search_notes 搜索 ----
            section("第 5 步：调用 search_notes 搜索「Python」")
            result = await session.call_tool(
                "search_notes",
                {"keyword": "Python"},
            )
            print(result.content[0].text)

            # ---- 再搜一个不存在的关键词，看看错误处理 ----
            section("第 6 步：搜索一个不存在的关键词「火箭」")
            result = await session.call_tool(
                "search_notes",
                {"keyword": "火箭"},
            )
            print(result.content[0].text)

    section("完成！Server 子进程已自动关闭。")
    print("\n回顾一下刚才发生的事：")
    print("  1. Client 启动了 Server 子进程")
    print("  2. 双方握手建立连接")
    print("  3. Client 自动发现了 3 个工具（不需要硬编码）")
    print("  4. Client 依次调用工具，Server 执行并返回结果")
    print("  5. 连接关闭，Server 进程退出")
    print("\n这就是 MCP 的核心：Client 和 Server 之间标准的「发现 → 调用」流程。")


if __name__ == "__main__":
    asyncio.run(main())
