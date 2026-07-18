"""
Token 预算控制系统
==================
演示如何在实际项目中管理 Claude API 的 Token 用量与成本。

涵盖：
  1. count_tokens API（精确计数）
  2. 字符估算法（快速估算）
  3. 动态预算分配（按任务优先级）
  4. 月度用量追踪（SQLite）
  5. 成本估算（各模型 price per token）
  6. Token 优化策略：去空白、压缩输入、前缀缓存
  7. 分级模型策略（简单任务用 Haiku，复杂用 Opus）
  8. 预算超出预警和自动降级
  9. 用量报表生成
"""

import re
import sqlite3
import textwrap
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import anthropic

# ---------------------------------------------------------------------------
# 1. 模型定价表（美元 / 百万 token，2025 年价格）
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    # model_id -> {"input": usd_per_1m, "output": usd_per_1m, "cache_write": ..., "cache_read": ...}
    "claude-opus-4-5": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-3-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
}

# 任务优先级 -> 推荐模型
PRIORITY_MODEL_MAP: dict[str, str] = {
    "low": "claude-haiku-3-5",       # 简单分类、摘要、格式化
    "medium": "claude-sonnet-4-5",    # 常规问答、代码生成
    "high": "claude-opus-4-5",        # 复杂推理、多步骤规划
}

# ---------------------------------------------------------------------------
# 2. SQLite 用量追踪
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "token_usage.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建用量记录表。"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            month       TEXT    NOT NULL,
            model       TEXT    NOT NULL,
            task_name   TEXT,
            priority    TEXT,
            input_tok   INTEGER NOT NULL,
            output_tok  INTEGER NOT NULL,
            cache_write INTEGER DEFAULT 0,
            cache_read  INTEGER DEFAULT 0,
            cost_usd    REAL    NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_usage(
    conn: sqlite3.Connection,
    model: str,
    input_tok: int,
    output_tok: int,
    task_name: str = "",
    priority: str = "medium",
    cache_write: int = 0,
    cache_read: int = 0,
) -> float:
    """将一次 API 调用的用量写入数据库，返回本次费用（美元）。"""
    cost = estimate_cost(model, input_tok, output_tok, cache_write, cache_read)
    now = datetime.utcnow().isoformat(timespec="seconds")
    month = now[:7]  # "YYYY-MM"
    conn.execute(
        """
        INSERT INTO usage_log
            (ts, month, model, task_name, priority, input_tok, output_tok,
             cache_write, cache_read, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (now, month, model, task_name, priority,
         input_tok, output_tok, cache_write, cache_read, cost),
    )
    conn.commit()
    return cost


# ---------------------------------------------------------------------------
# 3. 成本估算
# ---------------------------------------------------------------------------
def estimate_cost(
    model: str,
    input_tok: int,
    output_tok: int,
    cache_write: int = 0,
    cache_read: int = 0,
) -> float:
    """根据模型和 token 数量估算本次调用费用（美元）。"""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-5"])
    cost = (
        input_tok     * pricing["input"]       / 1_000_000
        + output_tok  * pricing["output"]      / 1_000_000
        + cache_write * pricing["cache_write"] / 1_000_000
        + cache_read  * pricing["cache_read"]  / 1_000_000
    )
    return round(cost, 8)


# ---------------------------------------------------------------------------
# 4. Token 计数
# ---------------------------------------------------------------------------
def count_tokens_exact(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict],
    system: str = "",
) -> int:
    """
    使用 Anthropic count_tokens API 精确统计 prompt 的 token 数量。
    适合在发送真实请求前做预检。
    """
    kwargs: dict = {"model": model, "messages": messages}
    if system:
        kwargs["system"] = system
    response = client.messages.count_tokens(**kwargs)
    return response.input_tokens


def count_tokens_estimate(text: str) -> int:
    """
    字符估算法：快速近似，不需要 API 调用。
    英文约 4 字符/token；中文约 1.5 字符/token（UTF-8 宽字符）。
    混合文本取加权平均。
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    other_chars = len(text) - chinese_chars
    estimated = chinese_chars / 1.5 + other_chars / 4.0
    return max(1, int(estimated))


# ---------------------------------------------------------------------------
# 5. Token 优化策略
# ---------------------------------------------------------------------------
def optimize_prompt(text: str, aggressive: bool = False) -> str:
    """
    Token 优化：去除多余空白、压缩重复换行。
    aggressive=True 时额外删除注释行和多余标点。
    """
    # 去除行尾空白
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # 合并连续空行（超过 2 行压缩为 1 行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除行首缩进（适合非代码场景）
    if aggressive:
        # 删除 # 注释行
        text = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
        # 删除多余空格
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
    return text


def build_cached_system_prompt(base_prompt: str) -> list[dict]:
    """
    构建带 cache_control 的 system prompt，利用前缀缓存降低重复内容费用。
    cache_control={"type": "ephemeral"} 告知 Anthropic 可缓存该块。
    """
    return [
        {
            "type": "text",
            "text": base_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


# ---------------------------------------------------------------------------
# 6. 动态预算分配
# ---------------------------------------------------------------------------
class BudgetManager:
    """
    月度预算管理器。
    - 按任务优先级动态分配可用预算
    - 超出阈值时自动降级模型并发出预警
    """

    def __init__(
        self,
        monthly_budget_usd: float,
        db_conn: sqlite3.Connection,
        warn_threshold: float = 0.80,   # 80% 时预警
        hard_limit: float = 1.00,       # 100% 时强制降级
    ):
        self.monthly_budget = monthly_budget_usd
        self.conn = db_conn
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit

    # ── 查询当月已用金额 ──────────────────────────────────────────────────
    def used_this_month(self) -> float:
        month = date.today().strftime("%Y-%m")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_log WHERE month = ?",
            (month,),
        ).fetchone()
        return float(row[0])

    def remaining_budget(self) -> float:
        return max(0.0, self.monthly_budget - self.used_this_month())

    def usage_ratio(self) -> float:
        return self.used_this_month() / self.monthly_budget if self.monthly_budget else 0.0

    # ── 分级模型选择（含自动降级） ──────────────────────────────────────
    def select_model(self, priority: str) -> tuple[str, str]:
        """
        根据优先级和当前用量选择模型。
        返回 (model_id, reason)。
        """
        ratio = self.usage_ratio()
        preferred = PRIORITY_MODEL_MAP.get(priority, "claude-sonnet-4-5")

        if ratio >= self.hard_limit:
            # 超出硬限制，强制使用最便宜模型
            return "claude-haiku-3-5", f"预算已耗尽({ratio:.0%})，强制降级至 Haiku"
        if ratio >= self.warn_threshold and preferred == "claude-opus-4-5":
            # 接近预警线，Opus 任务降级为 Sonnet
            return "claude-sonnet-4-5", f"预算接近上限({ratio:.0%})，Opus 自动降级至 Sonnet"
        return preferred, f"正常分配（已用 {ratio:.0%}）"

    # ── 预检：判断是否可以执行此次调用 ─────────────────────────────────
    def can_afford(self, estimated_cost: float) -> bool:
        return estimated_cost <= self.remaining_budget()


# ---------------------------------------------------------------------------
# 7. 智能调用入口（整合以上所有功能）
# ---------------------------------------------------------------------------
def smart_call(
    client: anthropic.Anthropic,
    budget_mgr: BudgetManager,
    db_conn: sqlite3.Connection,
    user_message: str,
    system_prompt: str = "",
    priority: str = "medium",
    task_name: str = "unnamed",
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    带预算控制的 Claude 调用：
      1. 优化 prompt
      2. 精确 count tokens
      3. 估算费用并检查预算
      4. 自动选择/降级模型
      5. 发送请求并记录用量
    """
    # Step 1: 优化
    optimized_msg = optimize_prompt(user_message)
    optimized_sys = optimize_prompt(system_prompt) if system_prompt else ""

    # Step 2: 选择模型
    model, reason = budget_mgr.select_model(priority)
    print(f"[模型选择] {model}  原因: {reason}")

    # Step 3: 精确计数
    messages = [{"role": "user", "content": optimized_msg}]
    input_tok = count_tokens_exact(client, model, messages, optimized_sys)
    estimated_cost = estimate_cost(model, input_tok, max_tokens)  # 用 max_tokens 保守估算

    # Step 4: 预算检查
    if not budget_mgr.can_afford(estimated_cost):
        print(
            f"[预警] 预算不足！预估费用 ${estimated_cost:.6f}，"
            f"剩余预算 ${budget_mgr.remaining_budget():.6f}"
        )
        return None

    # Step 5: 发送请求
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if optimized_sys:
        kwargs["system"] = build_cached_system_prompt(optimized_sys)

    resp = client.messages.create(**kwargs)
    output_tok = resp.usage.output_tokens
    cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0)

    # Step 6: 记录
    actual_cost = log_usage(
        db_conn, model,
        resp.usage.input_tokens, output_tok,
        task_name, priority, cache_write, cache_read,
    )
    print(
        f"[用量] input={resp.usage.input_tokens} output={output_tok} "
        f"cost=${actual_cost:.6f}  剩余预算=${budget_mgr.remaining_budget():.4f}"
    )

    return resp.content[0].text


# ---------------------------------------------------------------------------
# 8. 用量报表生成
# ---------------------------------------------------------------------------
def generate_report(conn: sqlite3.Connection, month: Optional[str] = None) -> str:
    """生成月度用量报表（纯文本表格）。"""
    if month is None:
        month = date.today().strftime("%Y-%m")

    rows = conn.execute(
        """
        SELECT model,
               priority,
               COUNT(*)        AS calls,
               SUM(input_tok)  AS total_input,
               SUM(output_tok) AS total_output,
               SUM(cost_usd)   AS total_cost
        FROM usage_log
        WHERE month = ?
        GROUP BY model, priority
        ORDER BY total_cost DESC
        """,
        (month,),
    ).fetchall()

    if not rows:
        return f"[{month}] 暂无用量记录。"

    header = f"\n{'='*70}\n  月度 Token 用量报表  {month}\n{'='*70}"
    cols = f"{'模型':<25} {'优先级':<10} {'调用次数':>8} {'输入Token':>12} {'输出Token':>12} {'费用(USD)':>12}"
    sep = "-" * 70
    lines = [header, cols, sep]

    total_cost = 0.0
    for model, pri, calls, inp, out, cost in rows:
        lines.append(
            f"{model:<25} {(pri or '-'):<10} {calls:>8} {inp:>12,} {out:>12,} {cost:>12.6f}"
        )
        total_cost += cost

    lines.append(sep)
    lines.append(f"{'合计':<48} {total_cost:>12.6f} USD")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. 主演示
# ---------------------------------------------------------------------------
def main():
    client = anthropic.Anthropic()
    conn = init_db()

    # 设置月度预算 $5
    budget = BudgetManager(
        monthly_budget_usd=5.0,
        db_conn=conn,
        warn_threshold=0.80,
    )

    system_prompt = textwrap.dedent("""\
        你是一位专业的 Python 编程助手，擅长写简洁、高效的代码。
        回答要简明扼要，代码示例不超过 20 行。
    """)

    tasks = [
        ("低优先级：格式化任务", "low",  "把这句话改成正式语气：'这个功能很好用'"),
        ("中优先级：代码生成", "medium", "写一个 Python 函数，计算列表的滑动平均值"),
        ("高优先级：复杂推理", "high",   "设计一个生产级别的限流算法，说明数据结构和时间复杂度"),
    ]

    for task_name, priority, prompt in tasks:
        print(f"\n{'─'*60}")
        print(f"任务: {task_name}")

        # 字符估算（无需 API，超快）
        est = count_tokens_estimate(prompt)
        print(f"  字符估算 tokens ≈ {est}")

        result = smart_call(
            client, budget, conn,
            user_message=prompt,
            system_prompt=system_prompt,
            priority=priority,
            task_name=task_name,
            max_tokens=512,
        )

        if result:
            preview = result[:120].replace("\n", " ")
            print(f"  回复预览: {preview}...")
        else:
            print("  [跳过] 预算不足")

    # 生成报表
    print(generate_report(conn))
    conn.close()


if __name__ == "__main__":
    main()
