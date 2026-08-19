"""
MCP Client 远程版 —— 连接 SSE 远程 Server

和 client.py 的区别：
  - client.py  用 stdio 传输（启动 Server 子进程，通过管道通信）
  - 本文件     用 SSE 传输（连接一个已运行的 HTTP Server）

使用前提：
  先在另一个终端启动 Server：
    python server_remote.py
  然后运行本文件：
    python client_remote.py
"""

import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


def section(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


async def main():
    # ---- 第 1 步：连接远程 Server ----
    # 和 stdio 模式的区别：不再启动子进程，而是直接连接一个 HTTP URL
    # sse_client 会先 GET /sse 建立长连接，之后通过 POST /messages/ 发请求
    server_url = "http://127.0.0.1:8000/sse"

    section("第 1 步：连接远程 Server")
    print(f"地址：{server_url}")

    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("连接成功，握手完成。")

            # ---- 第 2 步：动态发现工具 ----
            section("第 2 步：列出 Server 拥有的工具")
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            print(f"\n共发现 {len(tools_result.tools)} 个工具。")

            # ---- 第 3 步：调用 add_note 添加笔记 ----
            section("第 3 步：调用 add_note 添加笔记")
            notes_to_add = [
                ("远程笔记 1", "这条笔记是通过 SSE 远程连接添加的。"),
                ("MCP 传输对比", "stdio 适合本地，SSE 适合远程共享。"),
                ("今天天气", "晴，适合出门。"),
            ]
            for title, content in notes_to_add:
                result = await session.call_tool(
                    "add_note",
                    {"title": title, "content": content},
                )
                print(f"  添加「{title}」→ {result.content[0].text}")

            # ---- 第 4 步：列出全部笔记 ----
            section("第 4 步：调用 list_notes 列出全部笔记")
            result = await session.call_tool("list_notes", {})
            print(result.content[0].text)

            # ---- 第 5 步：搜索笔记 ----
            section("第 5 步：调用 search_notes 搜索「远程」")
            result = await session.call_tool(
                "search_notes",
                {"keyword": "远程"},
            )
            print(result.content[0].text)

    section("完成！已断开与远程 Server 的连接。")
    print("\n和 stdio 版的区别回顾：")
    print("  stdio：Client 启动 Server 子进程，管道通信，进程退出即结束")
    print("  SSE：  Server 独立运行，Client 通过 HTTP 连接，可多人同时连")
    print("\n这就是远程模式的价值：Server 部署一次，多个 Client 都能连。")


if __name__ == "__main__":
    asyncio.run(main())
