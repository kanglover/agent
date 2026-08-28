"""工具层：定义工具基类、注册表与三个演示工具（计算器/搜索/文件读取）。"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    output: str = ""
    error: Optional[str] = None


class Tool(ABC):
    """工具基类: 子类必须定义 NAME/DESCRIPTION/PARAMETERS 并实现 _run。"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 格式，如 {"type": "object", "properties": {...}, "required": [...]}"""
        ...

    @abstractmethod
    def _run(self, **kwargs) -> str:
        """返回工具输出字符串；抛异常表示失败。"""
        ...

    def execute(self, **kwargs) -> ToolResult:
        """包装 _run: 成功返回 ToolResult(True, output)；异常返回失败。"""
        # 先检查必需参数
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs or kwargs[param] is None:
                return ToolResult(success=False, error=f"Missing required parameter: {param}")
        try:
            output = self._run(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def to_openai_format(self) -> dict:
        """转为 OpenAI 工具定义格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：管理工具的注册、查询与执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具，重名时抛 ValueError。"""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """执行指定工具；工具不存在返回失败。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(**arguments)


class CalculatorTool(Tool):
    """计算器工具：用 eval 计算受限的算术表达式。"""

    _ALLOWED_PATTERN = re.compile(r"^[\d\s\+\-\*/\(\)\.\%\//]+$")

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算算术表达式，支持加减乘除、幂运算( **)、取余( %)、整除( //)。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的算术表达式，如 '3.14159 * 5 ** 2'",
                }
            },
            "required": ["expression"],
        }

    def _run(self, expression: str) -> str:
        # 去除首尾空白后校验
        expr = expression.strip()
        if not expr:
            raise ValueError("Expression is empty")
        if not self._ALLOWED_PATTERN.match(expr):
            raise ValueError(
                f"Invalid characters in expression: {expression!r}. "
                "Only digits, + - * / ** ( ) . % // and spaces are allowed."
            )
        # 二次防御：禁止任何字母或下划线
        if re.search(r"[a-zA-Z_]", expr):
            raise ValueError(
                f"Invalid characters in expression: {expression!r}. "
                "Letters and underscores are not allowed."
            )
        result = eval(expr, {"__builtins__": {}}, {})
        return f"Result: {result}"


class SearchTool(Tool):
    """搜索工具：基于预置小知识库做关键词匹配的 Mock 实现。"""

    # 预置知识库
    _KNOWLEDGE_BASE: dict[str, list[dict]] = {
        "circle area formula": [
            {
                "title": "Circle - Wikipedia",
                "snippet": "The area of a circle is π * r², where r is the radius. π ≈ 3.14159.",
            },
            {
                "title": "Geometry Formulas Handbook",
                "snippet": "A = π * r². For r=5, A ≈ 78.54. The formula is fundamental in geometry.",
            },
            {
                "title": "Math Is Fun - Circle",
                "snippet": "The area enclosed by a circle is A = πr². The circumference is C = 2πr.",
            },
        ],
        "python": [
            {
                "title": "Python.org Official Site",
                "snippet": "Python is a high-level, general-purpose programming language.",
            },
            {
                "title": "Python Tutorial - W3Schools",
                "snippet": "Python is a popular programming language used for web, data science and AI.",
            },
            {
                "title": "Real Python",
                "snippet": "Learn Python programming with tutorials for beginners and experts.",
            },
        ],
    }

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索网络知识，按关键词返回模拟搜索结果。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词",
                }
            },
            "required": ["query"],
        }

    def _run(self, query: str) -> str:
        query_lower = query.strip().lower()
        results: list[dict] = []

        # 精确匹配优先
        if query_lower in self._KNOWLEDGE_BASE:
            results = self._KNOWLEDGE_BASE[query_lower]
        else:
            # 模糊匹配：知识库 key 中包含查询词，或查询词包含 key
            for key, entries in self._KNOWLEDGE_BASE.items():
                if query_lower in key or key in query_lower:
                    results.extend(entries)

        if not results:
            return f"No results found for: {query}"

        lines: list[str] = []
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. [{item['title']}] {item['snippet']}")
        return "\n".join(lines)


class FileReadTool(Tool):
    """文件读取工具：读取真实文件的前 500 字符（utf-8）。"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取指定路径文件的前 500 字符内容（UTF-8 编码）。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                }
            },
            "required": ["path"],
        }

    def _run(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(500)
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {path}: {e}")
