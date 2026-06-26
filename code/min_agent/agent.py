# min_agent/agent.py
import anthropic
from min_agent.tools import TOOL_DEFINITIONS, run_tool
from min_agent.tracer import write_trace, build_trace_entry

MAX_STEPS = 5
client = anthropic.Anthropic()


def run_agent(task: str) -> dict:
    """
    运行最小 Agent Loop：observe → think → act → observe

    每一轮：
      1. THINK：调用 Claude，获取下一步决策
      2. 若 stop_reason == "end_turn"：任务完成，退出
      3. 若 stop_reason == "tool_use"：
           ACT：执行工具
           OBSERVE：把结果喂回给 Claude，继续下一轮

    Returns:
        成功：{"success": True,  "result": str, "steps": int}
        失败：{"success": False, "error": str,  "steps": int}
    """
    messages = [{"role": "user", "content": task}]
    step = 0

    try:
        while step < MAX_STEPS:

            # ── THINK ────────────────────────────────────────────
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1000,
                system=(
                    "你是一个助手，使用提供的工具完成任务。"
                    "先用 search_notes 搜索相关笔记，再用 write_summary 保存摘要。"
                ),
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # ── OBSERVE：任务完成 ────────────────────────────────
            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    "任务完成",
                )
                write_trace(build_trace_entry(
                    step=step,
                    thought_summary="任务完成，直接回复用户",
                    tool=None,
                    args=None,
                    observation=final_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ))
                return {"success": True, "result": final_text, "steps": step + 1}

            # ── ACT：调用工具 ────────────────────────────────────
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_args = block.input
                    observation = run_tool(tool_name, tool_args)

                    thought = next(
                        (b.text for b in response.content if hasattr(b, "text")),
                        f"决定调用 {tool_name}",
                    )

                    write_trace(build_trace_entry(
                        step=step,
                        thought_summary=thought,
                        tool=tool_name,
                        args=tool_args,
                        observation=observation,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": observation,
                    })

                # ── OBSERVE：把工具结果喂回给 Claude ────────────
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                step += 1

        return {
            "success": False,
            "error": f"超过最大步数限制（{MAX_STEPS} 步），任务未完成",
            "steps": step,
        }

    except anthropic.APIError as e:
        return {"success": False, "error": f"API 错误：{str(e)}", "steps": step}
    except Exception as e:
        return {"success": False, "error": f"未知错误：{str(e)}", "steps": step}
