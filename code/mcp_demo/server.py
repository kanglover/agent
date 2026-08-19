"""
MCP Server 示例 ——「笔记管家」

这是一个最简单的 MCP Server，用大白话理解：
  - Server 就像一个「小助手」，它掌握了一些技能（工具）
  - 任何人（Client）只要按 MCP 协议跟它对话，就能使用这些技能
  - 在这里，我们的小助手会三件事：添加笔记、列出笔记、搜索笔记

运行方式：
  它自己不会主动做任何事，需要被 Client 连接后才会响应。
  你可以直接运行 client.py，它会自动启动这个 Server。

  单独测试时也可以运行：python server.py
  （然后它会等待 stdin 上的 MCP 消息，不会退出）
"""

from mcp.server.mcpserver import MCPServer

# ===== 1. 创建一个 MCP Server，起个名字 =====
# 名字会告诉 Client「我是谁」，方便调试时辨认
# MCPServer 是 MCP Python SDK 的高级 API（旧版叫 FastMCP，2.0 起改名）
mcp = MCPServer("笔记管家")

# 用一个字典来存笔记，key 是标题，value 是内容
# 注意：数据存在内存里，Server 一关就没了（学习示例，够用）
_notes: dict[str, str] = {}


# ===== 2. 定义工具（给 AI 用的「技能」）=====
# 用 @mcp.tool() 装饰器注册，MCPServer 会自动从函数签名生成工具描述
# docstring 会变成工具的说明，AI 靠它判断「什么时候该用这个工具」


@mcp.tool()
def add_note(title: str, content: str) -> str:
    """添加一条笔记。如果标题已存在，会覆盖旧内容。

    参数：
        title:   笔记标题（简短的一句话）
        content: 笔记正文内容
    """
    _notes[title] = content
    return f"已保存笔记「{title}」，目前共有 {len(_notes)} 条笔记。"


@mcp.tool()
def list_notes() -> str:
    """列出所有已保存的笔记标题。

    没有参数。返回每条笔记的标题列表。
    """
    if not _notes:
        return "当前没有任何笔记。"

    lines = [f"共 {len(_notes)} 条笔记："]
    for i, title in enumerate(_notes, 1):
        lines.append(f"  {i}. {title}")
    return "\n".join(lines)


@mcp.tool()
def search_notes(keyword: str) -> str:
    """按关键词搜索笔记（同时匹配标题和正文）。

    参数：
        keyword: 要搜索的关键词
    """
    results = []
    for title, content in _notes.items():
        if keyword in title or keyword in content:
            results.append(f"「{title}」\n  {content}")

    if not results:
        return f"没有找到包含「{keyword}」的笔记。"

    return f"找到 {len(results)} 条匹配「{keyword}」的笔记：\n" + "\n".join(results)


# ===== 3. 启动 Server =====
# run() 会监听 stdio（标准输入输出），等待 Client 的消息
# Client 发来「列出你的工具」→ Server 回答三个工具
# Client 发来「调用 add_note，参数是 ...」→ Server 执行并返回结果
if __name__ == "__main__":
    mcp.run()
