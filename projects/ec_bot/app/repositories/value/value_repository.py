"""
字段取值检索仓储接口

取值召回在这套工程里有两个可替换落地方式：Elasticsearch 全文索引与 MySQL 全文索引。
上层（依赖注入、元数据构建、召回节点）只依赖这里的协议，不感知具体存储。
"""

from typing import Protocol, runtime_checkable

from app.entities.value_info import ValueInfo


@runtime_checkable
class ValueRepository(Protocol):
    """字段取值检索的统一接口"""

    async def ensure_index(self) -> None:
        """确保取值索引结构存在，幂等"""
        ...

    async def index(self, value_infos: list[ValueInfo], batch_size: int = 20) -> None:
        """批量写入字段取值，按 id 覆盖写入"""
        ...

    async def search(
        self,
        keyword: str,
        score_threshold: float | None = None,
        limit: int | None = None,
    ) -> list[ValueInfo]:
        """按关键词检索字段取值，返回匹配度达到阈值的取值列表"""
        ...
