"""
19_mcp_client.py - MCP Client 完整实现示例

演示内容：
  1. MCP Client 基本用法
  2. 连接 stdio 和 SSE 两种 transport
  3. list_tools（动态发现工具）
  4. call_tool（调用工具）
  5. list_resources 和 read_resource
  6. 将 MCP 工具集成进 Agent Loop（作为 Claude 的 tools）
  7. 同时连接多个 MCP Server（工具合并）
  8. 断线重连机制
  9. 演示：用 MCP 工具的完整 Agent 调用

依赖安装：
  pip install mcp anthropic

运行方式：
  python 19_mcp_client.py
  ANTHROPIC_API_KEY=sk-... python 19_mcp_client.py
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


# ─────────────────────────────────────────────────────────────────
# 第一部分：基本用法 —— stdio transport
# ─────────────────────────────────────────────────────────────────

async def demo_stdio_connection():
    """
    通过 stdio 连接本地 MCP Server 进程。

    stdio transport 原理：
      Claude Code 启动一个子进程（如 `python -m my_mcp_server`），
      通过标准输入/输出与其通信，协议完全在本机内完成。
    """
    print("\n=== 1. stdio 连接演示 ===")

    # StdioServerParameters 描述启动 Server 的命令
    server_params = StdioServerParameters(
        command="python",              # 可执行文件
        args=["-m", "my_mcp_server"],  # 命令参数（替换为真实 server 模块）
        env=None,                      # None = 继承父进程环境变量
    )

    async with AsyncExitStack() as stack:
        try:
            # stdio_client 启动子进程，返回 (read_stream, write_stream)
            read, write = await stack.enter_async_context(
                stdio_client(server_params)
            )
            # ClientSession 封装 MCP 协议的全部通信细节
            session = await stack.enter_async_context(
                ClientSession(read, write)
            )
            # 必须先 initialize()，完成 MCP 握手
            await session.initialize()

            print("  stdio 连接成功")
            await _show_session_info(session)

        except Exception as e:
            print(f"  stdio 连接失败（无真实 server，属正常）: {type(e).__name__}")


# ─────────────────────────────────────────────────────────────────
# 第二部分：SSE transport
# ─────────────────────────────────────────────────────────────────

async def demo_sse_connection(url: str = "http://localhost:8000/sse"):
    """
    通过 SSE（Server-Sent Events）连接远程 MCP Server。

    SSE transport 适合：
      - MCP Server 部署在远端（云服务器 / 局域网）
      - 需要跨进程或跨机器共享同一个 MCP Server
    """
    print("\n=== 2. SSE 连接演示 ===")

    async with AsyncExitStack() as stack:
        try:
            read, write = await stack.enter_async_context(sse_client(url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            print(f"  SSE 连接成功: {url}")
            await _show_session_info(session)

        except Exception as e:
            print(f"  SSE 连接失败（无真实 server，属正常）: {type(e).__name__}")


# ─────────────────────────────────────────────────────────────────
# 第三部分：list_tools / call_tool / list_resources / read_resource
# ─────────────────────────────────────────────────────────────────

async def _show_session_info(session: ClientSession):
    """展示一个已连接 session 的全部可用信息。"""

    # ── list_tools ──────────────────────────────────────────────
    print("\n  [list_tools] 可用工具：")
    tools_result = await session.list_tools()
    for tool in tools_result.tools:
        schema_keys = list((tool.inputSchema or {}).get("properties", {}).keys())
        print(f"    • {tool.name}: {tool.description}  参数={schema_keys}")

    # ── call_tool ───────────────────────────────────────────────
    if tools_result.tools:
        tool = tools_result.tools[0]
        print(f"\n  [call_tool] 尝试调用: {tool.name}")
        try:
            result = await session.call_tool(tool.name, arguments={})
            print(f"    结果: {result.content[:120]}")
        except Exception as e:
            print(f"    调用失败（参数不匹配，正常）: {e}")

    # ── list_resources ──────────────────────────────────────────
    print("\n  [list_resources] 可用资源：")
    try:
        res_result = await session.list_resources()
        for r in res_result.resources:
            print(f"    • {r.uri}  ({r.name})")

        # ── read_resource ────────────────────────────────────────
        if res_result.resources:
            uri = res_result.resources[0].uri
            print(f"\n  [read_resource] 读取: {uri}")
            content = await session.read_resource(uri)
            preview = str(content)[:120]
            print(f"    内容预览: {preview}")
    except Exception as e:
        print(f"    资源功能不可用: {e}")


# ─────────────────────────────────────────────────────────────────
# 第四部分：断线重连机制
# ─────────────────────────────────────────────────────────────────

class ReconnectingMCPClient:
    """
    带自动断线重连的 MCP Client。

    使用场景：长时间运行的 Agent，Server 可能因超时或重启而断开。
    内部逻辑：connect() 失败后等待 retry_delay 秒再重试，
             最多重试 max_retries 次；call_tool_safe() 发现异常
             时也会触发一次重连再重试。
    """

    def __init__(
        self,
        server_params: StdioServerParameters,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.server_params = server_params
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> bool:
        """建立连接；失败则按策略重试。"""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"  [重连] 尝试 {attempt}/{self.max_retries} ...")
                self._stack = AsyncExitStack()
                read, write = await self._stack.enter_async_context(
                    stdio_client(self.server_params)
                )
                self.session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self.session.initialize()
                print("  [重连] 连接成功")
                return True
            except Exception as e:
                print(f"  [重连] 失败: {type(e).__name__}")
                if self._stack:
                    await self._stack.aclose()
                    self._stack = None
                if attempt < self.max_retries:
                    print(f"  [重连] 等待 {self.retry_delay}s 后重试...")
                    await asyncio.sleep(self.retry_delay)

        print("  [重连] 达到最大重试次数，放弃")
        return False

    async def call_tool_safe(self, name: str, arguments: dict) -> Any:
        """调用工具；若断线则先重连再重试一次。"""
        if self.session is None:
            await self.connect()

        try:
            return await self.session.call_tool(name, arguments=arguments)
        except Exception as e:
            print(f"  [重连] 工具调用异常，触发重连: {e}")
            ok = await self.connect()
            if ok:
                return await self.session.call_tool(name, arguments=arguments)
            raise

    async def close(self):
        if self._stack:
            await self._stack.aclose()


# ─────────────────────────────────────────────────────────────────
# 第五部分：同时连接多个 MCP Server，工具合并
# ─────────────────────────────────────────────────────────────────

class MultiServerMCPClient:
    """
    连接多个 MCP Server，将全部工具合并成一个统一视图。

    工具名冲突时，自动改为 server_name::tool_name 格式，
    对外调用者无需关心路由细节，直接用统一名称即可。
    """

    def __init__(self):
        # server_name -> (session, stack)
        self.servers: dict[str, tuple[ClientSession, AsyncExitStack]] = {}
        # unified_name -> (server_name, original_tool_name)
        self.tool_map: dict[str, tuple[str, str]] = {}

    async def add_stdio_server(self, name: str, params: StdioServerParameters):
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.servers[name] = (session, stack)
            await self._register_tools(name, session)
            print(f"  [MultiServer] 已接入 stdio Server: {name}")
        except Exception as e:
            await stack.aclose()
            print(f"  [MultiServer] stdio Server {name} 连接失败: {type(e).__name__}")

    async def add_sse_server(self, name: str, url: str):
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(sse_client(url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.servers[name] = (session, stack)
            await self._register_tools(name, session)
            print(f"  [MultiServer] 已接入 SSE Server: {name} ({url})")
        except Exception as e:
            await stack.aclose()
            print(f"  [MultiServer] SSE Server {name} 连接失败: {type(e).__name__}")

    async def _register_tools(self, server_name: str, session: ClientSession):
        """把 server 的工具注册到统一 tool_map，冲突时加前缀。"""
        result = await session.list_tools()
        for tool in result.tools:
            key = tool.name
            if key in self.tool_map:
                key = f"{server_name}::{tool.name}"  # 冲突时加命名空间
            self.tool_map[key] = (server_name, tool.name)
            print(f"    注册工具: {key}")

    async def list_all_tools(self) -> list[dict]:
        """
        返回所有工具的 Claude tool 格式列表（直接传给 messages.create）。
        """
        all_tools: list[dict] = []
        for server_name, (session, _) in self.servers.items():
            result = await session.list_tools()
            for tool in result.tools:
                # 找出该工具注册时用的统一名称
                unified = next(
                    (k for k, v in self.tool_map.items() if v == (server_name, tool.name)),
                    tool.name,
                )
                all_tools.append({
                    "name": unified,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                })
        return all_tools

    async def call_tool(self, unified_name: str, arguments: dict) -> Any:
        """通过统一名称调用工具，自动路由到对应 Server。"""
        if unified_name not in self.tool_map:
            raise ValueError(f"未知工具: {unified_name}")
        server_name, orig_name = self.tool_map[unified_name]
        session, _ = self.servers[server_name]
        return await session.call_tool(orig_name, arguments=arguments)

    async def close_all(self):
        for _, (_, stack) in self.servers.items():
            await stack.aclose()


# ─────────────────────────────────────────────────────────────────
# 第六部分：Agent Loop（MCP 工具 + Claude）
# ─────────────────────────────────────────────────────────────────

async def run_agent_with_mcp(
    client: MultiServerMCPClient,
    user_message: str,
    model: str = "claude-opus-4-5",
    max_iterations: int = 10,
):
    """
    完整 Agent Loop：

    流程：
      用户消息
        → Claude（携带 MCP 工具列表）
        → 若 stop_reason == "tool_use"：调用 MCP 工具 → 结果送回 Claude
        → 循环，直到 stop_reason == "end_turn"
    """
    print(f"\n用户: {user_message}\n")

    api = anthropic.Anthropic()

    # 从已连接的 MCP Server 动态获取工具
    mcp_tools = await client.list_all_tools()

    # 演示模式：若无真实 Server，使用模拟工具
    if not mcp_tools:
        mcp_tools = [
            {
                "name": "calculator",
                "description": "执行数学表达式计算，返回数值结果",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "合法的数学表达式，如 (123+456)*2"}
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "get_weather",
                "description": "查询指定城市的当前天气",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称，如 北京"}
                    },
                    "required": ["city"],
                },
            },
        ]
        print(f"[演示模式] 使用模拟工具: {[t['name'] for t in mcp_tools]}")

    messages: list[dict] = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        print(f"--- 第 {iteration + 1} 轮 ---")

        response = api.messages.create(
            model=model,
            max_tokens=4096,
            tools=mcp_tools,
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")

        # 打印 Claude 的文本输出
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"Claude: {block.text}")

        if response.stop_reason == "end_turn":
            print("\nAgent 任务完成。")
            break

        # 处理工具调用
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            print(f"\n  → 调用工具: {block.name}  参数: {json.dumps(block.input, ensure_ascii=False)}")

            try:
                if client.tool_map:
                    # 真实 MCP 调用
                    mcp_result = await client.call_tool(block.name, block.input)
                    content = str(mcp_result.content)
                else:
                    # 演示模式模拟结果
                    content = _mock_tool(block.name, block.input)

                print(f"  ← 结果: {content}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
            except Exception as e:
                err = f"工具调用失败: {e}"
                print(f"  ← 错误: {err}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": err,
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    return messages


def _mock_tool(name: str, args: dict) -> str:
    """演示模式下模拟工具执行结果（不依赖真实 MCP Server）。"""
    if name == "calculator":
        try:
            # 只允许安全的数学运算
            result = eval(args.get("expression", "0"), {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception:
            return "表达式错误"
    if name == "get_weather":
        city = args.get("city", "未知城市")
        return f"{city}：晴，25°C，湿度 60%，微风"
    return f"[模拟] 工具 {name} 执行成功"


# ─────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────

async def main():
    print("=" * 56)
    print("MCP Client 完整示例")
    print("=" * 56)

    # ── 演示 1：stdio（无真实 server 时会报错，展示用法即可）──────
    # await demo_stdio_connection()

    # ── 演示 2：SSE（无真实 server 时会报错，展示用法即可）───────
    # await demo_sse_connection()

    # ── 演示 3：断线重连 ──────────────────────────────────────────
    print("\n=== 3. 断线重连机制演示 ===")
    dummy_params = StdioServerParameters(
        command="python", args=["-c", "raise SystemExit(1)"]
    )
    rc = ReconnectingMCPClient(dummy_params, max_retries=2, retry_delay=0.3)
    await rc.connect()  # 会失败并重试，演示重连逻辑
    await rc.close()

    # ── 演示 4：多 Server 工具合并 ───────────────────────────────
    print("\n=== 4. 多 Server 工具合并演示 ===")
    multi = MultiServerMCPClient()

    # 真实使用时取消注释，填入实际 Server 信息：
    # await multi.add_stdio_server("fs", StdioServerParameters(
    #     command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    # ))
    # await multi.add_sse_server("search", "http://localhost:9000/sse")

    print(f"  当前已注册工具数: {len(multi.tool_map)}")

    # ── 演示 5：完整 Agent Loop ──────────────────────────────────
    print("\n=== 5. 完整 Agent Loop（MCP 工具 + Claude）===")
    if os.environ.get("ANTHROPIC_API_KEY"):
        await run_agent_with_mcp(
            client=multi,
            user_message="请帮我计算 (123 + 456) * 2，并告诉我北京今天的天气。",
        )
    else:
        print("  未设置 ANTHROPIC_API_KEY，跳过 Agent Loop 演示。")
        print("  设置方法:  export ANTHROPIC_API_KEY=sk-ant-...")

    await multi.close_all()

    print("\n" + "=" * 56)
    print("全部演示完成")
    print("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
