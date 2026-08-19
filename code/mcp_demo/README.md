# MCP 示例 —— 笔记管家

## 这个示例解决什么问题？

你在学 agent 开发时，会反复遇到一个问题：**AI 怎么连接外部工具？**

以前的做法是「每个 AI 应用 + 每个工具」单独写一套对接代码，
N 个应用 × M 个工具 = N×M 套代码，维护到怀疑人生。

MCP（Model Context Protocol）把这个变成了 N+M：
每个工具写一个 MCP Server，每个 AI 应用内置一个 MCP Client，
大家用同一套协议说话，随便组合。

类比：**MCP 就是 AI 世界的 USB 接口。**
以前每个设备一种插头，现在统一成 USB-C，插上就能用。

---

## 文件说明

| 文件 | 角色 | 作用 |
|------|------|------|
| `server.py` | MCP Server | 「笔记管家」，暴露 3 个工具：添加、列出、搜索笔记 |
| `client.py` | MCP Client | 连接 Server，自动发现工具并调用，打印每步结果 |

两个文件的关系：

```
client.py（Client）                server.py（Server）
     |                                  |
     |  --- 启动子进程 (stdio) ------>   |
     |  --- 握手 (initialize) --------> |
     |  <--- 确认能力 ----------------- |
     |  --- 列出工具 (list_tools) --->  |
     |  <--- 返回 3 个工具 ------------- |
     |  --- 调用 add_note ----------->  |
     |  <--- 返回结果 ----------------- |
     |            ...                   |
```

---

## 运行方法

### 1. 安装依赖

```bash
cd code
uv pip install mcp
```

### 2. 运行 Client（会自动启动 Server）

```bash
cd code/mcp_demo
python client.py
```

你会看到完整的「发现工具 → 调用工具 → 返回结果」过程。

### 3. 单独运行 Server（一般不需要）

```bash
python server.py
```

Server 会等待 stdin 上的 MCP 协议消息，不会主动输出东西。
日常学习直接用 `client.py` 就行，它会自动拉起 Server。

---

## 接入 Claude Desktop（可选）

如果你想让 Claude Desktop 用上这个笔记工具，编辑配置文件：

- Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

加入以下内容：

```json
{
  "mcpServers": {
    "笔记管家": {
      "command": "python",
      "args": ["/这里/填/你/的/绝对路径/code/mcp_demo/server.py"]
    }
  }
}
```

保存后重启 Claude Desktop，它就会自动发现并可以使用「添加笔记」「列出笔记」「搜索笔记」三个工具了。

---

## 关键概念速查

**三个角色：**
- Host（宿主）：运行 AI 的应用，比如 Claude Desktop、Cursor
- Client（客户端）：嵌在 Host 里，负责跟 Server 通信
- Server（服务端）：独立进程，暴露工具给 AI 用

**三种原语：**
- Tools（工具）：AI 主动调用，能执行操作（本示例用的就是这个）
- Resources（资源）：Host 注入上下文，只读数据
- Prompts（提示词）：用户触发的预设模板

**传输方式：**
- stdio：本机子进程通信，最简单最常用（本示例用的就是这个）
- HTTP + SSE：远程服务，跨机器

**和 Function Calling 的区别：**
- Function Calling：每次请求内联定义工具，用完即弃
- MCP：独立部署的工具服务，动态发现、可复用、跨应用共享
