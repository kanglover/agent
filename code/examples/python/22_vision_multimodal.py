"""
22_vision_multimodal.py
视觉与多模态 —— Claude Vision API 完整示例

涵盖：
  1. 图片URL分析
  2. base64图片编码和分析
  3. 多图片对比（两张图对比分析）
  4. 图片+文字混合输入
  5. 用Claude分析图表（提取数据）
  6. 设计稿分析（提取颜色、字体、布局）
  7. 代码截图转文字代码
  8. PDF截图分析（将PDF页转为图片）
  9. 视觉Agent（看图执行操作）

运行前请确保：
  pip install anthropic pillow pymupdf requests
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import anthropic
import base64
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 初始化客户端
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()
MODEL = "claude-opus-4-5"


# ---------------------------------------------------------------------------
# 工具函数：将本地图片读取为 base64 字符串
# ---------------------------------------------------------------------------
def image_to_base64(image_path: str) -> tuple[str, str]:
    """
    读取本地图片文件，返回 (base64字符串, media_type)。
    支持 jpeg / png / gif / webp。
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def build_image_block_url(url: str) -> dict:
    """构造来自 URL 的图片 content block。"""
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


def build_image_block_base64(image_path: str) -> dict:
    """构造来自本地文件（base64）的图片 content block。"""
    data, media_type = image_to_base64(image_path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


# ---------------------------------------------------------------------------
# 1. 图片 URL 分析
# ---------------------------------------------------------------------------
def demo_url_analysis():
    """直接传入公开 URL，让 Claude 描述图片内容。"""
    print("\n" + "=" * 60)
    print("【1】图片 URL 分析")
    print("=" * 60)

    # 使用 Anthropic 官方文档示例图片
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_url(image_url),
                    {"type": "text", "text": "请用中文描述这张图片里有什么。"},
                ],
            }
        ],
    )
    print("Claude 的描述：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 2. base64 图片编码和分析
# ---------------------------------------------------------------------------
def demo_base64_analysis(image_path: str):
    """
    将本地图片编码为 base64 后发送给 Claude。
    如果没有本地图片，会先生成一个简单的测试 PNG。
    """
    print("\n" + "=" * 60)
    print("【2】base64 图片编码和分析")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"  未找到 {image_path}，跳过本节（请提供真实图片路径）。")
        return

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_base64(image_path),
                    {"type": "text", "text": "请用中文描述这张图片的内容和风格。"},
                ],
            }
        ],
    )
    print("Claude 的分析：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 3. 多图片对比（两张图对比分析）
# ---------------------------------------------------------------------------
def demo_multi_image_compare(url_a: str, url_b: str):
    """
    同时传入两张图片，让 Claude 对比分析异同。
    """
    print("\n" + "=" * 60)
    print("【3】多图片对比")
    print("=" * 60)

    message = client.messages.create(
        model=MODEL,
        max_tokens=768,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "以下是两张图片，请对比它们的异同："},
                    {"type": "text", "text": "图片 A："},
                    build_image_block_url(url_a),
                    {"type": "text", "text": "图片 B："},
                    build_image_block_url(url_b),
                    {
                        "type": "text",
                        "text": (
                            "请从内容、颜色、构图三个维度对比这两张图片，"
                            "并用中文输出结论。"
                        ),
                    },
                ],
            }
        ],
    )
    print("对比结果：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 4. 图片 + 文字混合输入
# ---------------------------------------------------------------------------
def demo_mixed_input(image_url: str):
    """
    同时提供背景文字说明和图片，实现上下文丰富的分析。
    """
    print("\n" + "=" * 60)
    print("【4】图片 + 文字混合输入")
    print("=" * 60)

    background = (
        "这是我们公司官网首页截图。"
        "我们是一家专注于 AI 教育的初创企业，目标用户是非技术背景的职场人。"
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=768,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": background},
                    build_image_block_url(image_url),
                    {
                        "type": "text",
                        "text": (
                            "结合上述背景，请评价这张截图的视觉设计是否符合目标用户的期望，"
                            "并给出3条改进建议。"
                        ),
                    },
                ],
            }
        ],
    )
    print("混合分析结果：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 5. 用 Claude 分析图表（提取数据）
# ---------------------------------------------------------------------------
def demo_chart_extraction(chart_url: str):
    """
    发送图表截图，要求 Claude 以 JSON 格式提取数据。
    """
    print("\n" + "=" * 60)
    print("【5】图表分析：提取数据")
    print("=" * 60)

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_url(chart_url),
                    {
                        "type": "text",
                        "text": (
                            "请从这张图表中提取所有可见的数据，"
                            "以如下 JSON 格式返回：\n"
                            "{\n"
                            '  "chart_type": "...",\n'
                            '  "title": "...",\n'
                            '  "x_axis_label": "...",\n'
                            '  "y_axis_label": "...",\n'
                            '  "data_points": [\n'
                            '    {"label": "...", "value": ...},\n'
                            "    ...\n"
                            "  ],\n"
                            '  "key_insight": "..."\n'
                            "}"
                        ),
                    },
                ],
            }
        ],
    )
    raw = message.content[0].text
    print("提取结果（原始）：")
    print(raw)

    # 尝试解析 JSON
    try:
        # 如果 Claude 在 JSON 外面包了 markdown 代码块，去掉它
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        print("\n解析后的结构化数据：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("（JSON 解析失败，原始文本已打印）")


# ---------------------------------------------------------------------------
# 6. 设计稿分析（提取颜色、字体、布局）
# ---------------------------------------------------------------------------
def demo_design_analysis(design_url: str):
    """
    分析设计稿，提取主色调、字体规格、布局结构。
    """
    print("\n" + "=" * 60)
    print("【6】设计稿分析")
    print("=" * 60)

    prompt = """请分析这张设计稿，提取以下信息并以 Markdown 格式返回：

## 主色调
列出 3~5 种主要颜色，尽量给出近似的 HEX 色值。

## 字体规格
- 标题字体：大小、粗细、颜色
- 正文字体：大小、粗细、颜色

## 布局结构
用简洁的文字描述整体布局（如：顶部导航 + 左侧边栏 + 右侧内容区）。

## 设计风格
一句话概括整体设计风格（如：扁平简约 / 拟物化 / 深色极客风等）。
"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_url(design_url),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 7. 代码截图转文字代码
# ---------------------------------------------------------------------------
def demo_code_ocr(code_screenshot_url: str):
    """
    将代码截图还原为可复制的文字代码。
    """
    print("\n" + "=" * 60)
    print("【7】代码截图 → 文字代码")
    print("=" * 60)

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_url(code_screenshot_url),
                    {
                        "type": "text",
                        "text": (
                            "请将截图中的代码完整转录为文字，"
                            "用对应语言的代码块包裹，"
                            "保持缩进和格式不变。"
                            "如果看到语法错误，请原样保留（不要自动修正）。"
                        ),
                    },
                ],
            }
        ],
    )
    print("转录结果：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 8. PDF 截图分析（将 PDF 页转为图片后分析）
# ---------------------------------------------------------------------------
def demo_pdf_page_analysis(pdf_path: str, page_index: int = 0):
    """
    使用 PyMuPDF 将 PDF 某页渲染为 PNG，再交给 Claude 分析。
    依赖：pip install pymupdf
    """
    print("\n" + "=" * 60)
    print("【8】PDF 截图分析")
    print("=" * 60)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  未安装 pymupdf，跳过本节。请运行：pip install pymupdf")
        return

    if not os.path.exists(pdf_path):
        print(f"  未找到 {pdf_path}，跳过本节。")
        return

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        print(f"  PDF 只有 {len(doc)} 页，page_index={page_index} 超出范围。")
        return

    page = doc[page_index]
    # 以 2x 分辨率渲染，提高 OCR 准确率
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    tmp_png = "/tmp/pdf_page_snapshot.png"
    pix.save(tmp_png)
    print(f"  已将第 {page_index + 1} 页渲染为 {tmp_png}")

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    build_image_block_base64(tmp_png),
                    {
                        "type": "text",
                        "text": (
                            "这是一份 PDF 文档的某一页截图，"
                            "请提取该页所有文字内容，"
                            "并简要说明页面的结构（标题、正文、表格、图表等）。"
                        ),
                    },
                ],
            }
        ],
    )
    print("分析结果：")
    print(message.content[0].text)


# ---------------------------------------------------------------------------
# 9. 视觉 Agent（看图执行操作）
# ---------------------------------------------------------------------------
def visual_agent(task: str, image_url: str, max_steps: int = 5):
    """
    简单的视觉 Agent：
    - 接收一个任务描述和图片
    - 用 ReAct 模式循环：思考 → 行动 → 观察
    - 预定义两个工具：describe_image / extract_text

    这里用 tool_use 模拟 Agent 的工具调用循环。
    """
    print("\n" + "=" * 60)
    print("【9】视觉 Agent")
    print("=" * 60)
    print(f"任务：{task}")
    print(f"图片：{image_url}\n")

    tools = [
        {
            "name": "describe_image",
            "description": "对图片进行全面的视觉描述，包括主体、背景、颜色、布局等。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "重点关注的区域或元素，如 'top-left corner' 或 'text area'",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "extract_text",
            "description": "提取图片中所有可见的文字内容（OCR）。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "要提取文字的区域，如 'full', 'header', 'footer'",
                    }
                },
                "required": [],
            },
        },
    ]

    # 系统提示：让 Agent 知道它在看一张图
    system = (
        "你是一个视觉分析 Agent。你会收到一张图片和一个任务。"
        "请使用提供的工具逐步完成任务，每次只调用一个工具。"
        "当你认为已经获得足够信息可以完成任务时，直接给出最终答案。"
    )

    # 初始消息包含图片
    messages = [
        {
            "role": "user",
            "content": [
                build_image_block_url(image_url),
                {"type": "text", "text": f"任务：{task}"},
            ],
        }
    ]

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )

        # 将 assistant 回复加入历史
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Agent 已给出最终答案
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"最终答案：\n{block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  调用工具：{block.name}，参数：{block.input}")
                    # 模拟工具执行结果（真实场景中这里执行真正的逻辑）
                    if block.name == "describe_image":
                        result = (
                            "[模拟] 图片包含：色彩丰富的UI界面，顶部有导航栏，"
                            "中央有主标题文字，底部有按钮区域。"
                        )
                    elif block.name == "extract_text":
                        result = "[模拟] 提取到的文字：首页 | 关于我们 | 联系我们 | 开始使用"
                    else:
                        result = "[模拟] 未知工具"

                    print(f"  工具返回：{result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # 把工具结果喂回 Agent
            messages.append({"role": "user", "content": tool_results})
        else:
            print(f"  stop_reason={response.stop_reason}，退出循环。")
            break


# ---------------------------------------------------------------------------
# 主函数：依次运行所有示例
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  视觉与多模态 Claude Vision API 示例")
    print("=" * 60)

    # 公开测试图片 URL（维基共享资源）
    sample_png = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )
    sample_chart = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/"
        "Pie_chart_of_w3schools.png/320px-Pie_chart_of_w3schools.png"
    )
    sample_code_screenshot = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/"
        "Python_3.11_source_code_example.png/320px-Python_3.11_source_code_example.png"
    )

    # 1. 图片 URL 分析
    demo_url_analysis()

    # 2. base64 分析（需要本地图片，这里传一个占位路径）
    demo_base64_analysis("/tmp/sample_local_image.jpg")

    # 3. 多图片对比
    img_b = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/"
        "Aster_Tataricus.jpg/320px-Aster_Tataricus.jpg"
    )
    demo_multi_image_compare(sample_png, img_b)

    # 4. 图片 + 文字混合输入
    demo_mixed_input(sample_png)

    # 5. 图表分析：提取数据
    demo_chart_extraction(sample_chart)

    # 6. 设计稿分析
    demo_design_analysis(sample_png)

    # 7. 代码截图转文字代码
    demo_code_ocr(sample_code_screenshot)

    # 8. PDF 截图分析（需要本地 PDF 文件）
    demo_pdf_page_analysis("/tmp/sample.pdf", page_index=0)

    # 9. 视觉 Agent
    visual_agent(
        task="请描述这张图片的主要内容，并识别其中所有可见文字。",
        image_url=sample_png,
    )

    print("\n所有示例运行完毕。")


if __name__ == "__main__":
    main()
