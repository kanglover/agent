"""
批量处理优化：Anthropic Batch API 完整工作流
============================================================

核心价值：
- 异步处理，无需等待每条请求完成
- 享受 50% 价格折扣（相比实时 API）
- 适合非时间敏感型的大批量任务

成本/速度对比：
┌─────────────┬──────────────┬──────────────┬────────────────────┐
│    方式      │   价格折扣   │   响应延迟   │     适用场景       │
├─────────────┼──────────────┼──────────────┼────────────────────┤
│ 实时 API    │    无折扣    │  秒级响应    │ 交互式对话、实时   │
│ 并发实时 API│    无折扣    │  秒级响应    │ 小批量、需实时反馈 │
│ Batch API   │  -50% 折扣  │  分钟~小时   │ 大批量、离线任务   │
└─────────────┴──────────────┴──────────────┴────────────────────┘

适用场景举例：
  - 数据标注：给 10000 条文本打分类标签
  - 离线分析：对历史对话做情感分析
  - 批量评测：对模型输出做质量评估
  - 内容生成：批量生成商品描述/摘要

官方文档：https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

import asyncio
import time
import json
from typing import Any

import anthropic


# ─────────────────────────────────────────────
# 第一部分：创建 Batch 任务
# ─────────────────────────────────────────────

def create_batch_requests(texts: list[str]) -> list[dict]:
    """
    把一批文本转换成 Batch API 需要的请求格式。

    每条请求需要：
      - custom_id：你自己定的唯一标识，用来对应结果
      - params：和普通 messages.create 参数完全一致
    """
    requests = []
    for idx, text in enumerate(texts):
        requests.append({
            "custom_id": f"task-{idx:04d}",          # 自定义 ID，方便后续对应结果
            "params": {
                "model": "claude-opus-4-5",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": f"用一句话概括以下文本的情感（正面/负面/中性）并给出理由：\n\n{text}"
                    }
                ]
            }
        })
    return requests


def submit_batch(texts: list[str]) -> str:
    """
    提交 Batch 任务，返回 batch_id。

    注意：提交后立即返回，不会等待处理完成。
    处理时间取决于任务量，通常几分钟到几小时。
    """
    client = anthropic.Anthropic()

    requests = create_batch_requests(texts)

    print(f"正在提交 Batch 任务，共 {len(requests)} 条请求...")

    batch = client.messages.batches.create(requests=requests)

    print(f"Batch 任务已提交！")
    print(f"  batch_id  : {batch.id}")
    print(f"  状态      : {batch.processing_status}")
    print(f"  创建时间  : {batch.created_at}")
    print(f"  请求总数  : {batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored}")

    return batch.id


# ─────────────────────────────────────────────
# 第二部分：轮询 Batch 状态
# ─────────────────────────────────────────────

def poll_batch_until_done(batch_id: str, interval_seconds: int = 30) -> Any:
    """
    轮询 Batch 状态，直到处理完成。

    processing_status 取值：
      - "in_progress"  ：正在处理中
      - "canceling"    ：正在取消
      - "ended"        ：已结束（成功 / 部分失败 / 全部失败 均为此状态）
    """
    client = anthropic.Anthropic()

    print(f"\n开始轮询 Batch 状态（每 {interval_seconds} 秒检查一次）...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired

        print(
            f"[{time.strftime('%H:%M:%S')}] 状态: {batch.processing_status} | "
            f"处理中: {counts.processing} | "
            f"成功: {counts.succeeded} | "
            f"失败: {counts.errored} | "
            f"共: {total}"
        )

        if batch.processing_status == "ended":
            print("\nBatch 处理完成！")
            return batch

        time.sleep(interval_seconds)


# ─────────────────────────────────────────────
# 第三部分：下载和解析 Batch 结果
# ─────────────────────────────────────────────

def download_and_parse_results(batch_id: str) -> dict[str, Any]:
    """
    下载 Batch 结果并按 custom_id 整理成字典。

    结果格式：
      {
        "task-0001": {"status": "succeeded", "text": "..."},
        "task-0002": {"status": "errored",   "error": "..."},
        ...
      }
    """
    client = anthropic.Anthropic()

    print(f"\n正在下载 Batch 结果...")

    results: dict[str, Any] = {}

    # results_stream 返回一个可迭代的结果流，逐条处理，不会一次性加载全部到内存
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            # 成功：取出模型回复的文本
            message = result.result.message
            text_content = ""
            for block in message.content:
                if hasattr(block, "text"):
                    text_content += block.text
            results[custom_id] = {
                "status": "succeeded",
                "text": text_content,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }

        elif result.result.type == "errored":
            # 失败：记录错误信息（部分失败不影响其他条目）
            results[custom_id] = {
                "status": "errored",
                "error": result.result.error.type,
            }

        elif result.result.type == "expired":
            # 超时：Batch 任务在 24 小时内未处理完会过期
            results[custom_id] = {
                "status": "expired",
            }

    print(f"共解析 {len(results)} 条结果")
    return results


# ─────────────────────────────────────────────
# 第四部分：错误处理（部分失败）
# ─────────────────────────────────────────────

def analyze_results(results: dict[str, Any]) -> None:
    """
    分析结果，统计成功/失败情况，并打印示例。

    Batch API 的重要特性：部分失败不影响整体，
    每条请求独立成功或失败。
    """
    succeeded = {k: v for k, v in results.items() if v["status"] == "succeeded"}
    errored   = {k: v for k, v in results.items() if v["status"] == "errored"}
    expired   = {k: v for k, v in results.items() if v["status"] == "expired"}

    total_input_tokens  = sum(v.get("input_tokens", 0)  for v in succeeded.values())
    total_output_tokens = sum(v.get("output_tokens", 0) for v in succeeded.values())

    print("\n" + "=" * 50)
    print("结果汇总")
    print("=" * 50)
    print(f"  成功条数   : {len(succeeded)}")
    print(f"  失败条数   : {len(errored)}")
    print(f"  过期条数   : {len(expired)}")
    print(f"  消耗 Token : 输入 {total_input_tokens} / 输出 {total_output_tokens}")

    if succeeded:
        print("\n前 3 条成功结果示例：")
        for key, val in list(succeeded.items())[:3]:
            print(f"  [{key}] {val['text'][:80]}...")

    if errored:
        print(f"\n失败条目（需要重试）：")
        for key, val in errored.items():
            print(f"  [{key}] 错误类型: {val['error']}")

    # 对失败条目的建议处理方式
    if errored:
        failed_ids = list(errored.keys())
        print(f"\n建议：对 {len(failed_ids)} 条失败条目单独重试（实时 API 或新建 Batch）")


# ─────────────────────────────────────────────
# 第五部分：asyncio.gather 并发实时 API（小批量）
# ─────────────────────────────────────────────

async def process_single_async(client: anthropic.AsyncAnthropic, text: str, idx: int) -> dict:
    """异步处理单条文本（实时 API）。"""
    message = await client.messages.create(
        model="claude-haiku-4-5",       # 小批量用 Haiku，速度快成本低
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"用一句话概括情感（正面/负面/中性）：{text}"
        }]
    )
    return {
        "idx": idx,
        "text": message.content[0].text,
        "tokens": message.usage.input_tokens + message.usage.output_tokens,
    }


async def process_batch_concurrent(texts: list[str], max_concurrent: int = 5) -> list[dict]:
    """
    用 asyncio.gather + 信号量 并发调用实时 API。

    适合小批量（< 100 条）且需要快速拿到结果的场景。
    max_concurrent 控制并发数，避免触发速率限制。
    """
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)   # 最多同时 5 个请求

    async def guarded_task(text: str, idx: int) -> dict:
        async with semaphore:
            return await process_single_async(client, text, idx)

    print(f"\n并发实时 API 模式：{len(texts)} 条请求，最大并发 {max_concurrent}")

    start = time.time()
    tasks = [guarded_task(text, idx) for idx, text in enumerate(texts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    # 过滤异常
    successes = [r for r in results if isinstance(r, dict)]
    errors    = [r for r in results if isinstance(r, Exception)]

    print(f"完成！耗时 {elapsed:.1f}s，成功 {len(successes)} 条，失败 {len(errors)} 条")
    return successes


# ─────────────────────────────────────────────
# 第六部分：选择策略
# ─────────────────────────────────────────────

def print_strategy_guide() -> None:
    """
    打印选择策略指南：什么时候用 Batch，什么时候用并发实时 API。
    """
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║                    选择策略速查表                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  用 Batch API 当：                                               ║
║    ✓ 任务量 > 100 条                                             ║
║    ✓ 不需要立刻拿到结果（可以等几分钟到几小时）                  ║
║    ✓ 预算敏感（享受 50% 折扣）                                   ║
║    ✓ 离线任务：数据标注、评测、批量生成                          ║
║    ✓ 定时任务：每天/每周跑一次的分析                             ║
║                                                                  ║
║  用并发实时 API 当：                                             ║
║    ✓ 任务量 < 100 条                                             ║
║    ✓ 需要秒级响应，用户在等待                                    ║
║    ✓ 任务依赖上一步结果（链式任务）                              ║
║    ✓ 需要流式输出（streaming）                                   ║
║                                                                  ║
║  经验法则：                                                      ║
║    - 省钱首选 Batch；速度首选并发实时                            ║
║    - Batch 单次最多 10,000 条请求                                ║
║    - Batch 有效期 29 天（过期自动取消）                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(guide)


# ─────────────────────────────────────────────
# 主流程演示
# ─────────────────────────────────────────────

# 示例数据：模拟一批待分析的用户评论
SAMPLE_TEXTS = [
    "这款手机的拍照效果非常出色，电池续航也很棒，强烈推荐！",
    "快递慢得离谱，包装破损，客服态度差，非常失望。",
    "产品质量一般，价格适中，没有特别惊喜但也没有大问题。",
    "第二次购买了，每次都很满意，会继续支持这个品牌。",
    "颜色和图片差距太大，退货流程也很麻烦，不会再买了。",
]


def demo_batch_api() -> None:
    """演示完整的 Batch API 工作流（需要真实 API Key）。"""
    print("=" * 60)
    print("Batch API 完整工作流演示")
    print("=" * 60)

    # 步骤 1：提交 Batch 任务
    batch_id = submit_batch(SAMPLE_TEXTS)

    # 步骤 2：轮询状态（生产环境可以先做别的事，过一会再查）
    # 这里为了演示缩短等待时间；实际可以跑一个单独的脚本来轮询
    batch = poll_batch_until_done(batch_id, interval_seconds=10)

    # 步骤 3：下载并解析结果
    results = download_and_parse_results(batch_id)

    # 步骤 4：分析结果（含错误处理）
    analyze_results(results)

    # 步骤 5：保存结果到本地（可选）
    output_path = "/tmp/batch_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")


def demo_concurrent_api() -> None:
    """演示 asyncio 并发实时 API（小批量快速处理）。"""
    print("=" * 60)
    print("并发实时 API 演示（小批量）")
    print("=" * 60)

    results = asyncio.run(
        process_batch_concurrent(SAMPLE_TEXTS[:3], max_concurrent=3)
    )

    print("\n结果：")
    for r in results:
        print(f"  [{r['idx']}] {r['text'][:80]}")


if __name__ == "__main__":
    # 打印选择策略速查表
    print_strategy_guide()

    # 选择运行模式
    print("请选择演示模式：")
    print("  1. Batch API 完整工作流（适合大批量，有 50% 折扣）")
    print("  2. 并发实时 API（适合小批量，响应更快）")
    print("  3. 仅打印策略指南")

    choice = input("\n输入 1/2/3（默认 3）：").strip() or "3"

    if choice == "1":
        demo_batch_api()
    elif choice == "2":
        demo_concurrent_api()
    else:
        print("\n已打印策略指南，无需调用 API。")
        print("取消注释 demo_batch_api() 或 demo_concurrent_api() 即可运行真实演示。")
