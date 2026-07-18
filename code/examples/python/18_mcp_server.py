"""
18_mcp_server.py  —  构建完整的 MCP Server

MCP（Model Context Protocol）是 Anthropic 推出的开放协议，
让 AI 模型能够安全、标准化地访问外部工具、数据和提示词模板。

安装依赖：
    pip install mcp httpx

启动 Server（stdio 模式）：
    python 18_mcp_server.py

测试方法（任选其一）：
    1. 用 MCP Inspector:
       npx @modelcontextprotocol/inspector python 18_mcp_server.py

    2. 配置到 Claude Desktop（~/Library/Application Support/Claude/claude_desktop_config.json）:
       {
         "mcpServers": {
           "my-server": {
             "command": "python",
             "args": ["/path/to/18_mcp_server.py"],
             "env": { "MCP_API_KEY": "secret-key-123" }
           }
         }
       }

    3. 用 mcp CLI:
       mcp dev 18_mcp_server.py
"""

# ── 1. 安装提示 ──────────────────────────────────────────────────────────────
# pip install mcp httpx
# mcp 是官方 Python SDK，httpx 是异步 HTTP 客户端

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

# ── 2. MCP Server 基本结构 ───────────────────────────────────────────────────

# Server 实例是整个 MCP Server 的核心，名字会显示在客户端
app = Server("demo-mcp-server")

# ── 7. 认证：API Key 校验 ────────────────────────────────────────────────────
# 真实场景中应从环境变量读取，不要硬编码在代码里
VALID_API_KEYS = {"secret-key-123", "dev-key-456"}


def verify_api_key() -> bool:
    """从环境变量读取并校验 API Key。"""
    key = os.environ.get("MCP_API_KEY", "")
    if key not in VALID_API_KEYS:
        # 开发模式：没有配置 Key 时也允许访问（仅用于本地调试）
        if os.environ.get("MCP_DEV_MODE", "false").lower() == "true":
            return True
        return False
    return True


# ── 临时工作目录（文件工具读写这里）───────────────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "mcp_demo"
WORK_DIR.mkdir(exist_ok=True)

# ── 内存中的 mock 数据库 ────────────────────────────────────────────────────
_DB_PATH = WORK_DIR / "mock.db"


def _init_mock_db() -> None:
    """初始化 mock SQLite 数据库，预置几条演示数据。"""
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            stock INTEGER
        )"""
    )
    conn.execute("DELETE FROM products")  # 每次启动重置数据
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?)",
        [
            (1, "Claude Pro 订阅", 20.0, 9999),
            (2, "MCP 教程书", 49.9, 50),
            (3, "AI 工具包", 199.0, 10),
        ],
    )
    conn.commit()
    conn.close()


_init_mock_db()

# ── 3. 实现 Tools ────────────────────────────────────────────────────────────
# Tools = AI 可以调用的函数，类似于 function calling


@app.list_tools()
async def list_tools() -> ListToolsResult:
    """告诉客户端本 Server 提供哪些工具。"""
    return ListToolsResult(
        tools=[
            # 工具 1：文件读写
            Tool(
                name="file_rw",
                description="读取或写入工作目录中的文本文件。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["read", "write"],
                            "description": "read=读文件，write=写文件",
                        },
                        "filename": {
                            "type": "string",
                            "description": "文件名（不含路径），例如 notes.txt",
                        },
                        "content": {
                            "type": "string",
                            "description": "写入时的文件内容（action=write 时必填）",
                        },
                    },
                    "required": ["action", "filename"],
                },
            ),
            # 工具 2：HTTP 请求
            Tool(
                name="http_get",
                description="发起 HTTP GET 请求并返回响应体（支持 JSON 和纯文本）。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "目标 URL，例如 https://api.github.com/zen",
                        },
                        "headers": {
                            "type": "object",
                            "description": "可选的请求头，键值对形式",
                        },
                    },
                    "required": ["url"],
                },
            ),
            # 工具 3：数据库查询（mock）
            Tool(
                name="db_query",
                description="查询 mock 商品数据库，支持按名称模糊搜索或获取全部数据。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "搜索关键词，为空则返回全部商品",
                        }
                    },
                    "required": [],
                },
            ),
        ]
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """统一的工具调用入口，按名称路由到具体实现。"""

    # ── 认证检查 ──────────────────────────────────────────────────────────────
    if not verify_api_key():
        return CallToolResult(
            content=[TextContent(type="text", text="❌ 认证失败：请设置环境变量 MCP_API_KEY")],
            isError=True,
        )

    # ── 6. 错误处理：用 try/except 包裹每个工具 ─────────────────────────────
    try:
        if name == "file_rw":
            return await _tool_file_rw(arguments)
        elif name == "http_get":
            return await _tool_http_get(arguments)
        elif name == "db_query":
            return await _tool_db_query(arguments)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ 未知工具：{name}")],
                isError=True,
            )
    except Exception as exc:
        # 捕获所有未预期的异常，返回友好错误信息
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ 工具执行失败 [{name}]：{exc}")],
            isError=True,
        )


async def _tool_file_rw(args: dict) -> CallToolResult:
    """工具实现：文件读写。"""
    action = args["action"]
    filename = Path(args["filename"]).name  # 只取文件名，防止路径穿越攻击
    filepath = WORK_DIR / filename

    if action == "read":
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在：{filename}")
        content = filepath.read_text(encoding="utf-8")
        return CallToolResult(
            content=[TextContent(type="text", text=f"📄 {filename} 内容：\n\n{content}")]
        )
    elif action == "write":
        text = args.get("content", "")
        filepath.write_text(text, encoding="utf-8")
        return CallToolResult(
            content=[TextContent(type="text", text=f"✅ 已写入 {filename}（{len(text)} 字符）")]
        )
    else:
        raise ValueError(f"不支持的 action：{action}")


async def _tool_http_get(args: dict) -> CallToolResult:
    """工具实现：HTTP GET 请求。"""
    url = args["url"]
    headers = args.get("headers", {})
    timeout = 10.0  # 10 秒超时

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        # 尝试解析 JSON，失败则返回纯文本
        try:
            body = json.dumps(response.json(), ensure_ascii=False, indent=2)
        except Exception:
            body = response.text[:2000]  # 截断超长响应

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"🌐 HTTP {response.status_code}  {url}\n\n{body}",
                )
            ]
        )


async def _tool_db_query(args: dict) -> CallToolResult:
    """工具实现：SQLite mock 数据库查询。"""
    keyword = args.get("keyword", "").strip()

    conn = sqlite3.connect(_DB_PATH)
    if keyword:
        rows = conn.execute(
            "SELECT id, name, price, stock FROM products WHERE name LIKE ?",
            (f"%{keyword}%",),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, name, price, stock FROM products").fetchall()
    conn.close()

    if not rows:
        return CallToolResult(
            content=[TextContent(type="text", text="🔍 未找到匹配的商品。")]
        )

    lines = ["| ID | 名称 | 价格 | 库存 |", "|----|------|------|------|"]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | ¥{row[2]} | {row[3]} |")

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(lines))]
    )


# ── 4. 实现 Resources ────────────────────────────────────────────────────────
# Resources = AI 可以读取的数据源（类似只读文件系统）


@app.list_resources()
async def list_resources() -> ListResourcesResult:
    """列出所有可访问的资源。"""
    return ListResourcesResult(
        resources=[
            Resource(
                uri="config://server/settings",
                name="Server 配置",
                description="当前 MCP Server 的运行配置信息",
                mimeType="application/json",
            ),
            Resource(
                uri="data://realtime/stats",
                name="实时统计数据",
                description="每次读取时返回最新的运行时统计（时间戳、请求计数等）",
                mimeType="application/json",
            ),
        ]
    )


# 简单的请求计数器（模拟实时数据）
_request_counter = 0


@app.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    """按 URI 读取具体资源内容。"""
    global _request_counter
    _request_counter += 1

    if uri == "config://server/settings":
        config = {
            "server_name": "demo-mcp-server",
            "version": "1.0.0",
            "work_dir": str(WORK_DIR),
            "auth_enabled": True,
            "supported_transports": ["stdio"],
        }
        return ReadResourceResult(
            contents=[
                TextContent(
                    type="text",
                    text=json.dumps(config, ensure_ascii=False, indent=2),
                )
            ]
        )

    elif uri == "data://realtime/stats":
        stats = {
            "timestamp": datetime.now().isoformat(),
            "resource_reads": _request_counter,
            "db_path": str(_DB_PATH),
            "work_dir_files": [f.name for f in WORK_DIR.glob("*") if f.is_file()],
        }
        return ReadResourceResult(
            contents=[
                TextContent(
                    type="text",
                    text=json.dumps(stats, ensure_ascii=False, indent=2),
                )
            ]
        )

    else:
        raise ValueError(f"未知资源 URI：{uri}")


# ── 5. 实现 Prompts ──────────────────────────────────────────────────────────
# Prompts = 可复用的提示词模板，AI 可以按需调用


@app.list_prompts()
async def list_prompts() -> ListPromptsResult:
    """列出所有提示词模板。"""
    return ListPromptsResult(
        prompts=[
            Prompt(
                name="code_review",
                description="代码审查模板：提交代码片段，获得结构化的审查报告",
                arguments=[
                    PromptArgument(
                        name="language",
                        description="编程语言，例如 Python、JavaScript",
                        required=True,
                    ),
                    PromptArgument(
                        name="code",
                        description="需要审查的代码片段",
                        required=True,
                    ),
                    PromptArgument(
                        name="focus",
                        description="重点关注方向，例如「性能」「安全」「可读性」",
                        required=False,
                    ),
                ],
            )
        ]
    )


@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    """按名称获取并渲染提示词模板。"""
    if name != "code_review":
        raise ValueError(f"未知提示词：{name}")

    args = arguments or {}
    language = args.get("language", "未指定")
    code = args.get("code", "（未提供代码）")
    focus = args.get("focus", "整体质量")

    prompt_text = f"""请对以下 {language} 代码进行专业审查，重点关注：{focus}。

## 代码

```{language.lower()}
{code}
```

## 审查报告格式

请按以下结构输出：

1. **总体评价**（1-2 句话）
2. **发现的问题**（按严重程度排序，每条说明问题+位置+修复建议）
3. **优点**（值得保留或借鉴的地方）
4. **改进后的代码**（如有必要）
"""
    return GetPromptResult(
        description=f"{language} 代码审查（关注：{focus}）",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text),
            )
        ],
    )


# ── 8. 启动 Server（stdio transport）────────────────────────────────────────
# stdio transport 是最常用的本地模式：
# 父进程（Claude Desktop / MCP Inspector）通过标准输入/输出与 Server 通信

async def main() -> None:
    print(f"[MCP Server] 启动中，工作目录：{WORK_DIR}", flush=True)
    print(f"[MCP Server] 认证状态：{'已启用' if verify_api_key() else '未配置（开发模式）'}", flush=True)

    # stdio_server() 返回 (read_stream, write_stream)，交给 app.run() 处理
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
