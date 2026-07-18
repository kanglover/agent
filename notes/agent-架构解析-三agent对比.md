# 三 Agent 架构解析与对比

> 日期：2026-07-04

---

## 一、Codex CLI（OpenAI）

### 1.1 定位

Codex CLI 是 OpenAI 于 2025 年开源的终端编程 Agent。核心亮点是「安全优先」：
在执行任何 shell 命令或文件写入之前，都要经过一套「自动判断 + 用户审批」的双重把关机制，
并且把「沙箱执行」当作默认的兜底手段，而不是事后再来补救。

> 说明：Codex 项目后来把 CLI 从 TypeScript 重写成了 Rust（目录从 `codex-cli/src` 迁移到
> `codex-rs/`），当前 GitHub `main` 分支已经找不到本笔记引用的 `.ts` 文件了。
> 以下分析基于该文件被删除前的最后一个版本（commit `75febbdef`，2025-08-08），
> 这也是网上大多数「Codex CLI 源码解析」文章所依据的版本，架构思路对理解 Codex 仍然成立。

### 1.2 ApprovalPolicy（审批模式）

源码位置：`codex-cli/src/approvals.ts`

Codex 定义了三档审批策略（`ApprovalPolicy` 类型）：

| 模式 | 行为 |
|---|---|
| `suggest` | 只有「已知安全」的只读命令（如 `ls`、`cat`、`git status`）会被自动放行，其余一律询问用户 |
| `auto-edit` | 在 `suggest` 的基础上，额外放行「写入范围被限制在可写目录内」的 `apply_patch` 编辑 |
| `full-auto` | 所有命令都自动放行，但强制在沙箱（sandbox）里执行——网络被禁用，文件写入被限制在指定目录 |

判断逻辑由核心函数 `canAutoApprove(command, workdir, policy, writableRoots)` 完成，简化后的决策顺序是：

1. 如果命令是 `apply_patch`：交给 `canAutoApproveApplyPatch()`——`suggest` 下必须询问用户；
   `auto-edit`/`full-auto` 下先检查补丁改动的文件是否都落在「可写目录」内，是则自动批准，
   否则 `full-auto` 仍然放行但强制走沙箱，`auto-edit` 则转为询问用户。
2. 如果命令命中 `isSafeCommand()` 的白名单（`ls`/`cat`/`grep`/`git status`/`git diff` 等只读操作，
   且诸如 `find -exec`、`rg --pre` 这类「可执行外部命令」的危险参数会被专门过滤掉），直接自动批准。
3. 如果是 `bash -lc "..."` 形式的复合命令，会用 `shell-quote` 解析成命令片段，
   只有当每个片段都安全、且片段之间只用 `&&`/`||`/`|`/`;` 等「无副作用」操作符连接时，才整体自动批准
   （用括号 `()`/`{}` 的子 shell 一律视为不安全，交给用户）。
4. 兜底：`full-auto` 下自动批准但跑沙箱；其余模式一律 `ask-user`。

也就是说，Codex 的审批不是简单的「白名单 vs 黑名单」，而是「命令解析 + 路径约束 + 沙箱兜底」三层叠加。

### 1.3 Rollback 机制

源码位置：`codex-cli/src/utils/check-in-git.ts`、`codex-cli/src/utils/get-diff.ts`、
`codex-cli/src/utils/agent/handle-exec-command.ts`

和很多人的预期（自动 `git stash` / 自动打快照）不同，这版 Codex **并没有内置「改之前自动存档、改坏了自动回滚」的机制**，
它选择的是「预防 + 可见性」而不是「事后撤销」：

- **前提检查**：`checkInGit()` 用 `git rev-parse --is-inside-work-tree` 判断当前目录是否在 Git 仓库里。
  Codex 会据此提示用户——你最好在一个受 Git 版本控制的目录里跑，这样任何改动本身就自带「可回滚」的能力
  （用户可以自己 `git diff` / `git checkout` / `git reset` 撤销）。
- **随时可见的 diff**：`getGitDiff()` 会拼出「已跟踪文件的 `git diff`」+「未跟踪文件相对 `/dev/null` 的 diff」，
  提供给终端里的 `/diff` 面板，让用户在批准/拒绝之前，或者事后复查时，随时能看到完整的改动全貌。
- **沙箱限制爆炸半径**：`handle-exec-command.ts` 里的 `getSandbox()` 会在 `runInSandbox=true` 时，
  按平台选择 macOS Seatbelt（`sandbox-exec`）或 Linux Landlock，把写入权限限制在
  `additionalWritableRoots`（默认是当前工作目录）内、并禁用网络。也就是说，
  即使某条命令被 `full-auto` 自动放行，最坏情况也只是把「已知的一小片目录」搞乱，而不是整个系统。
- **会话级白名单**：`handle-exec-command.ts` 里的 `alwaysApprovedCommands` 只是「记住用户这次会话选过『总是批准』的命令类型」，
  用于减少重复询问，跟撤销无关。

一句话总结：Codex 的「rollback」本质上是把 Git 本身当作撤销层，自己负责「让改动始终发生在 Git 能看见、沙箱能兜底的范围内」。

### 1.4 apply-patch

源码位置：`codex-cli/src/utils/agent/apply-patch.ts`（参考自 OpenAI Cookbook 的 `apply_patch.py` 参考实现）

Codex 不用标准的 unified diff（`---`/`+++`/`@@ -1,3 +1,4 @@` 那种格式），而是自定义了一套更适合 LLM 生成的
「V4A diff」纯文本格式，好处是不依赖精确的行号（LLM 数行号很容易数错），改用**上下文匹配**来定位改动位置：

```
*** Begin Patch
*** Update File: path/to/file.py
@@ class BaseClass
@@     def search():
-        pass
+        raise NotImplementedError()
*** End Patch
```

应用流程（`process_patch()`）：

1. `identify_files_needed(text)` 扫描 `*** Update File:` / `*** Delete File:` 这类标记，得到需要读取的文件列表。
2. `load_files()` 把这些文件内容读进内存，作为 `orig`（补丁必须建立在「文件已存在」的假设上）。
3. `text_to_patch(text, orig)` 用一个简单的行扫描 `Parser` 把补丁文本解析成结构化的 `Patch`（每个文件一个 `PatchAction`）。
   定位每个 hunk 时会用 `find_context()` 在原文件里搜索 `[context_before]`/`[context_after]` 这几行上下文，
   并且做了三级容错：先精确匹配，再忽略行尾空白，最后忽略首尾空白全匹配——目的是抵抗 LLM 输出里
   偶尔漏掉的空格、或者中文引号/破折号这类「看起来一样但 Unicode 码位不同」的字符。
4. `patch_to_commit(patch, orig)` 把 `Patch` 转换成 `Commit`（每个文件对应 add/update/delete 三种 `FileChange` 之一，
   `update` 类型会真正拼出新文件全文 `new_content`）。
5. `apply_commit(commit, writeFn, removeFn)` 才是真正落盘的一步：新增/更新调用 `writeFn` 写文件
   （`update` 且指定了 `move_path` 时先写新路径再删旧路径，实现「重命名」），删除调用 `removeFn`。

整个过程中，第 1-4 步都是「纯函数、只读」的，只有最后一步 `apply_commit` 才真正碰磁盘，
这样即便解析失败（`DiffError`）也不会留下「写了一半」的文件。

### 1.5 核心 Loop 伪代码

```
# 用户在终端里发一条消息，Codex 开始一次 run()
def run(conversation_history, model, approval_policy):
    turn_input = conversation_history

    while True:  # 一次 run 里可能有多轮"模型输出 -> 工具调用 -> 结果喂回"
        stream = call_openai_responses_api(model, turn_input)  # 流式请求

        new_turn_input = []
        for event in stream:  # 边接收边处理
            if event 是 "输出项完成":
                if 输出项是 function_call（如 shell/apply_patch）:
                    结果 = handle_function_call(item)   # 见下方审批+执行
                    new_turn_input.append(结果)
                else:
                    展示给用户（普通文本/推理过程）

        if new_turn_input 为空:
            break  # 模型没有再发起新的工具调用，本轮结束
        turn_input = new_turn_input  # 把工具结果喂回，进入下一轮

def handle_function_call(item):
    command = 解析(item.arguments)
    safety = can_auto_approve(command, workdir, approval_policy, writable_roots)
    if safety.type == "ask-user":
        decision = 等待用户在终端里选择(Yes / Always / No / 解释一下)
        if decision 不是 Yes/Always: return "已取消"
    elif safety.type == "reject":
        return "命令被拒绝"
    # auto-approve 或用户已批准 -> 执行（可能带沙箱）
    return exec_or_apply_patch(command, run_in_sandbox=safety.runInSandbox)
```

---

## 二、Claude Code（Anthropic）

### 2.1 定位

Anthropic 官方的终端编程 Agent。核心亮点是「Hook 中间件系统」：在工具真正执行的前后，
可以插入任意自定义逻辑（审计、阻断、修改输入输出），思路上很像 Web 框架里的中间件（middleware）。
整个系统建立在标准的「LLM → 工具调用 → 结果喂回」Agent Loop 之上，
用 Permission Model（权限模型）和 Hook System（钩子系统）两层机制做安全护栏。

### 2.2 Hook 系统

Hook 一共有三个 Hook 时机（PreToolUse、PostToolUse、Stop）+ 权限检查，按执行顺序排列：

1. **权限检查**：先过一遍 `settings.json` 里的 `allow`/`deny` 规则，命中 `deny` 直接拒绝，
   命中 `allow` 直接放行，都不命中则向用户弹出确认框。
2. **PreToolUse Hook**（工具执行前触发）：可以读取工具名和入参，做三件事之一——
   放行（不改变入参）、修改入参后放行、或者直接阻断（返回 `block` 让工具根本不执行）。
   典型用途：把危险命令记进审计日志、拦截触碰敏感文件的写操作。
3. **工具执行**：Hook 没有阻断的话，才真正调用工具的 `execute()`。
4. **PostToolUse Hook**（工具执行后触发）：可以读取工具的执行结果，做修改输出、
   触发副作用（比如执行完 `Write` 后自动跑一遍 `prettier` 格式化）、发通知等操作。
5. **Stop Hook**（整个 Agent 本轮停止时触发）：用来做收尾清理，比如弹一条系统通知。

Hook 在 `settings.json` 里用「matcher（匹配哪个工具，比如 `Bash`/`Write`）+ command（要执行的
shell 命令）」的形式声明，Hook 本身是一条外部命令，通过环境变量（如 `$CLAUDE_TOOL_INPUT`）
拿到工具调用的上下文信息，这样 Hook 逻辑可以用任何语言写，不需要嵌入 Agent 本体的代码里。

### 2.3 工具执行流程

一次完整的工具调用，从 Hook 到执行再到 Hook，链路是：

```
tool_use 请求
   → 权限检查（allow / deny / ask）
   → PreToolUse Hook（可修改入参 / 可阻断）
   → 执行工具 execute(input)
   → PostToolUse Hook（可修改输出 / 触发副作用）
   → 包装成 tool_result，追加进消息历史
   → 喂回 LLM，进入下一轮
```

关键点：权限检查在最外层把关「能不能调这个工具」，Hook 系统在内层把关「调用前后要不要
额外做点什么」，两者职责不同、可以叠加使用。

### 2.4 核心 Loop 伪代码

```python
while True:
    response = llm(messages, tools)          # 调用 LLM，拿到本轮响应
    if end_turn: break                       # 模型没有再发起工具调用，本轮结束
    for tool_call in response.tool_calls:
        args = run_pre_hooks(tool_call.name, tool_call.args)
        if args is None: continue            # Hook 阻断，跳过这次工具调用
        result = execute(tool_call.name, args)      # 真正执行工具
        result = run_post_hooks(tool_call.name, result)  # 执行后再过一遍 Hook
        messages.append(tool_result(result)) # 把结果喂回消息历史
```

---

## 三、Hermes（Anthropic 内部）

### 3.1 定位

Anthropic 内部的多 Agent 编排框架，核心亮点是「上下文隔离的 Subagent」：
一个 Orchestrator（编排者）Agent 把复杂任务拆解成若干子任务，通过 `Task` 工具分发给
专门的 Subagent 去执行，每个 Subagent 都在独立的对话历史里运行，完成后只把一段总结文字
交回给 Orchestrator。

### 3.2 Orchestrator 模式

Orchestrator 的工作分四步：

1. **规划（plan）**：让主 LLM 把用户的复杂请求拆解成一组子任务（比如「代码审查」拆成
   「代码质量检查」「安全漏洞扫描」「测试覆盖分析」「文档完整性检查」四个子任务）。
2. **调度（schedule）**：按子任务之间的依赖关系排序，没有依赖关系的子任务可以并行执行。
3. **分发执行（dispatch）**：Orchestrator 唯一能调用的工具通常就是 `Task`（或类似的
   `dispatch_task`），调用时指定「角色（role）」和「任务描述（task）」，由框架据此
   启动一个全新的 Subagent 实例去跑这个子任务。
4. **汇总（synthesize）**：所有子任务跑完后，Orchestrator 把各个 Subagent 返回的结果
   拼起来，生成最终答复。

### 3.3 上下文隔离的意义

Subagent 不共享 Orchestrator 的对话历史，而是拿到一份全新的、只包含「任务描述 + 必要背景」
的上下文去独立运行 LLM 调用。这样做有几个好处：

- **防止上下文爆炸**：子任务过程中产生的大量中间工具调用结果（比如读了十几个文件），
  只会留在 Subagent 自己的历史里，不会把 Orchestrator 的上下文塞满。
- **故障不级联**：某个 Subagent 中途走了弯路、试错了很多次，这些噪音不会污染
  Orchestrator 的判断，Orchestrator 只看到最终干净的结果。
- **可以并行执行**：多个 Subagent 之间没有共享状态，互不干扰，天然支持并发跑多个子任务。
- **权限可以按需收窄**：可以给不同的 Subagent 配置不同的工具权限边界，
  比如只给「安全扫描」Subagent 开放只读工具。

### 3.4 核心 Loop 伪代码

```python
# Orchestrator（主循环）
while True:
    response = orchestrator_llm(messages, tools=[dispatch_task])
    if end_turn: break
    if tool_call == "dispatch_task":
        result = run_subagent(args.role, args.task)  # 完全独立的 LLM 调用
        messages.append(tool_result(result))

# Subagent（独立运行，不共享 Orchestrator 的历史）
def run_subagent(role, task):
    return llm(system=ROLES[role], messages=[user(task)])
```

---

## 四、三者核心差异对比

| 维度 | Claude Code | Codex CLI | Hermes |
|------|-------------|-----------|--------|
| Agent 数量 | 1 | 1 | 1 Orchestrator + N Subagent |
| 核心创新 | Hook 中间件 | Approval 模式 + 沙箱 | 上下文隔离编排 |
| 工具执行 | Hook 拦截后直接执行 | 审批后执行（高风险需确认） | 分发给专门 Subagent |
| 安全机制 | Hook 可阻断任意工具 | 审批模式 + 可选沙箱 | Subagent 上下文隔离 |
| 适合场景 | 通用编程助手 | 文件修改（安全优先） | 复杂多步骤任务 |

## 五、最小实现对应关系

| 文件 | 对应 Agent | 核心模式 |
|------|-----------|---------|
| `code/min_claude_code.py` | Claude Code | Hook 中间件 |
| `code/min_codex.py` | Codex CLI | Approval 模式 + 快照回滚 |
| `code/min_hermes.py` | Hermes | Orchestrator + Subagent |

---

## 六、运行说明

### 环境准备
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

### 运行三个最小实现

**Claude Code（Hook 中间件）：**
```bash
python code/min_claude_code.py
# 演示：读取 README.md → Hook 拦截记录 → 写入 /tmp/claude_summary.txt
```

**Codex CLI（Approval 审批）：**
```bash
python code/min_codex.py --mode suggest    # 每步等待确认
python code/min_codex.py --mode auto       # 自动快照后执行
python code/min_codex.py --mode full-auto  # 直接执行
```

**Hermes（Orchestrator 编排）：**
```bash
python code/min_hermes.py
# Orchestrator 自动分发：coder → reviewer → writer
```

### 关键学习点
- **min_claude_code.py**：观察 `[HOOK:pre]` 打印，理解 Hook 中间件
- **min_codex.py**：运行 suggest 模式，体验审批流程；输入 `rollback` 体验回滚
- **min_hermes.py**：观察 `Subagent [coder]` 等打印，理解上下文隔离编排

