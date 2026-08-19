"""
MCP Server 远程版 —— 同样的「笔记管家」，但用 SSE 传输

和 server.py 的区别：
  - server.py  用 stdio 传输（本地子进程，通过管道通信）
  - 本文件     用 SSE 传输（HTTP 服务，通过网络通信）

核心变化只有一行：
  mcp.run()                      → stdio 模式（默认）
  mcp.run(transport="sse", ...)  → SSE 模式（远程服务）

工具定义完全复用 server.py，一行都不用改。
这正体现了 MCP 的设计理念：工具逻辑和传输方式解耦。

运行方式：
  python server_remote.py
  （然后 Server 会监听 http://127.0.0.1:8000，等待 Client 连接）

  另开一个终端运行 client_remote.py 即可连接。
"""

# 直接从 server.py 导入同一个 MCPServer 实例
# 工具（add_note / list_notes / search_notes）全部复用，不用重写
from server import mcp


if __name__ == "__main__":
    print("笔记管家（SSE 远程版）")
    print(f"监听地址：http://127.0.0.1:8000")
    print(f"SSE 端点：/sse")
    print(f"消息端点：/messages/")
    print()
    print("等待 Client 连接...（按 Ctrl+C 退出）")
    print()

    # transport="sse" 让 Server 通过 HTTP + SSE 传输工作
    # host / port 指定监听地址，和生产用的 uvicorn 一样
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8000,
    )
