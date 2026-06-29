# code/context_builder/compressor.py
import hashlib

MAX_CHARS_DEFAULT = 500


class Compressor:
    """
    对长内容进行截断压缩：
    - 短内容（≤ max_chars）：原样返回，ref_id 为 ""
    - 长内容（> max_chars）：截断 + 生成 ref_id + 存储原文
    ref_id = "ref:" + MD5(content)[:8]，相同内容 → 相同 ID（可重现）
    """

    def __init__(self, max_chars: int = MAX_CHARS_DEFAULT):
        self._max_chars = max_chars
        self._store: dict[str, str] = {}

    def compress(self, content: str, source: str = "") -> tuple[str, str, int]:
        """
        Returns:
            (summary, ref_id, original_char_count)
        """
        original_len = len(content)

        if original_len <= self._max_chars:
            return content, "", original_len

        ref_id = "ref:" + hashlib.md5(content.encode()).hexdigest()[:8]
        self._store[ref_id] = content
        summary = (
            content[: self._max_chars]
            + f"…[截断，共 {original_len} 字，引用：{ref_id}]"
        )
        return summary, ref_id, original_len

    def retrieve(self, ref_id: str) -> str | None:
        """根据 ref_id 取回原始内容"""
        return self._store.get(ref_id)

    def is_compressed(self, ref_id: str) -> bool:
        """判断 ref_id 是否对应一条压缩记录"""
        return bool(ref_id) and ref_id in self._store
