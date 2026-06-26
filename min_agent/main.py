# min_agent/main.py
from pathlib import Path
from min_agent.agent import run_agent
from min_agent.tasks import TASKS


def main():
    # 每次运行前清空 trace 文件
    Path("trace.jsonl").write_text("", encoding="utf-8")

    print("=" * 55)
    print("   最小 Agent Loop — observe → think → act → observe")
    print("=" * 55)

    for i, task in enumerate(TASKS, 1):
        print(f"\n📋 任务 {i}/{len(TASKS)}: {task}")
        print("-" * 45)

        result = run_agent(task)

        if result["success"]:
            preview = result["result"][:80]
            print(f"✅ 成功（共 {result['steps']} 步）")
            print(f"   {preview}{'...' if len(result['result']) > 80 else ''}")
        else:
            print(f"❌ 失败（{result['steps']} 步后停止）")
            print(f"   原因：{result['error']}")

    print("\n" + "=" * 55)
    print("✅ 全部任务完成！Trace 已写入 trace.jsonl")
    print("   运行 `cat trace.jsonl | python3 -m json.tool` 查看详情")


if __name__ == "__main__":
    main()
