"""
字段取值 MySQL 仓储

用 MySQL 自身的全文索引承载字段取值召回，替换「Elasticsearch + IK 分词插件」
这套依赖，部署时就不必再额外托管一个 ES 实例。

召回分两步：
1. ``MATCH ... AGAINST`` 借助 ngram 解析器拿候选行。ngram 默认按 2 字符切词，
   中文场景不需要再装分词插件
2. 在应用层用「包含判定 + 字符序列相似度」打分并按阈值过滤，对齐原 ES ``min_score``
   只想保留高质量命中的语义

若目标 MySQL 建不出 ngram 全文索引，会自动退化成 LIKE 匹配，保证取值召回这条链路
不会因为存储能力差异整体不可用。
"""

import difflib

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.conf.app_config import app_config
from app.core.log import logger
from app.entities.value_info import ValueInfo

# MySQL 不支持 CREATE INDEX IF NOT EXISTS，索引是否已建要查 information_schema
_CHECK_FULLTEXT_SQL = """
SELECT COUNT(*)
FROM information_schema.statistics
WHERE table_schema = DATABASE()
  AND table_name = :table_name
  AND index_name = :index_name
"""

_FULLTEXT_INDEX_NAME = "ft_value"

# 全文索引可用性是库级别的事实，进程内缓存一次即可，避免每次请求都重新探测。
# None 表示尚未探测，False 表示已确认不可用并进入降级模式。
_fulltext_available: bool | None = None


def score_value(keyword: str, value: str) -> float:
    """计算关键词与字段取值的匹配度，取值区间 0~1

    完全相等或互相包含直接判定为 1，否则退化成字符序列相似度。
    这里刻意不要求词序一致，是为了兼容「华北」命中「华北地区」这类业务值。
    """

    normalized_keyword = keyword.strip().lower()
    normalized_value = value.strip().lower()

    if not normalized_keyword or not normalized_value:
        return 0.0
    if (
        normalized_keyword == normalized_value
        or normalized_keyword in normalized_value
        or normalized_value in normalized_keyword
    ):
        return 1.0

    return difflib.SequenceMatcher(None, normalized_keyword, normalized_value).ratio()


class ValueMySQLRepository:
    """基于 MySQL 全文索引的字段取值检索仓储"""

    def __init__(self, session: AsyncSession, table_name: str | None = None):
        self.session = session
        self.table_name = table_name or app_config.value_store.table_name

        # 表名只能来自配置，这里做一次标识符校验，避免配置写错导致 SQL 拼装异常
        if not self.table_name.replace("_", "").isalnum():
            raise ValueError(f"非法的取值索引表名：{self.table_name}")

    async def ensure_index(self) -> None:
        """确保取值索引表与 ngram 全文索引存在，可重复执行"""
        global _fulltext_available

        await self.session.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name}
                (
                    id        VARCHAR(512) PRIMARY KEY COMMENT '取值编号，column_id.value',
                    value     VARCHAR(512) NOT NULL COMMENT '字段真实取值',
                    column_id VARCHAR(64) NOT NULL COMMENT '所属字段编号',
                    FULLTEXT INDEX {_FULLTEXT_INDEX_NAME} (value) WITH PARSER ngram
                ) DEFAULT CHARSET = utf8mb4 COMMENT = '字段取值全文索引'
                """
            )
        )

        exists = await self.session.execute(
            text(_CHECK_FULLTEXT_SQL),
            {"table_name": self.table_name, "index_name": _FULLTEXT_INDEX_NAME},
        )

        # 表是历史遗留且缺全文索引时补建；失败说明当前 MySQL 不支持 ngram 解析器
        if not exists.scalar():
            try:
                await self.session.execute(
                    text(
                        f"ALTER TABLE {self.table_name} "
                        f"ADD FULLTEXT INDEX {_FULLTEXT_INDEX_NAME} (value) WITH PARSER ngram"
                    )
                )
            except Exception as error:
                _fulltext_available = False
                logger.warning(f"取值索引无法启用 ngram 全文索引，退化为 LIKE 匹配：{error}")

        # DDL 在 MySQL 中是隐式提交，这里显式收口，避免把事务留给后续的 begin()
        await self.session.commit()

    async def index(self, value_infos: list[ValueInfo], batch_size: int = 20) -> None:
        """批量写入字段取值；重复写入按 id 覆盖，便于反复重建索引"""
        if not value_infos:
            return

        statement = text(
            f"""
            INSERT INTO {self.table_name} (id, value, column_id)
            VALUES (:id, :value, :column_id)
            ON DUPLICATE KEY UPDATE value = VALUES(value), column_id = VALUES(column_id)
            """
        )

        for start in range(0, len(value_infos), batch_size):
            batch = value_infos[start : start + batch_size]
            await self.session.execute(
                statement,
                [
                    {"id": info.id, "value": info.value, "column_id": info.column_id}
                    for info in batch
                ],
            )

        await self.session.commit()

    async def search(
        self,
        keyword: str,
        score_threshold: float | None = None,
        limit: int | None = None,
    ) -> list[ValueInfo]:
        """按关键词检索字段取值，返回匹配度不低于阈值的取值列表"""

        keyword = (keyword or "").strip()
        if not keyword:
            return []

        threshold = (
            app_config.value_store.score_threshold
            if score_threshold is None
            else score_threshold
        )
        query_limit = app_config.value_store.limit if limit is None else limit
        # 先多取候选再在应用层打分过滤，避免过滤后条数不够
        candidate_limit = max(query_limit * 5, 50)

        candidates = await self._query_candidates(keyword, candidate_limit)

        scored = sorted(
            ((score_value(keyword, candidate.value), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )

        # 同一取值可能被多个关键词命中，按 id 去重后再按上限截断
        matched: dict[str, ValueInfo] = {}
        for score, candidate in scored:
            if score < threshold:
                continue
            matched.setdefault(candidate.id, candidate)

        return list(matched.values())[:query_limit]

    async def _query_candidates(self, keyword: str, limit: int) -> list[ValueInfo]:
        """取出候选取值：优先走全文索引，降级模式下走 LIKE"""

        global _fulltext_available

        # ngram 解析器按 2 字符切词，单字关键词拿不到任何 token，直接走 LIKE 更稳
        if _fulltext_available is not False and len(keyword) >= 2:
            try:
                result = await self.session.execute(
                    text(
                        f"SELECT id, value, column_id FROM {self.table_name} "
                        "WHERE MATCH(value) AGAINST (:keyword IN NATURAL LANGUAGE MODE) "
                        "LIMIT :limit"
                    ),
                    {"keyword": keyword, "limit": limit},
                )
                return [ValueInfo(**dict(row)) for row in result.mappings().fetchall()]
            except Exception as error:
                # 只在第一次失败时告警并降级，后续请求不再重复撞全文索引
                _fulltext_available = False
                logger.warning(f"取值全文检索失败，退化为 LIKE 匹配：{error}")
                # 语句报错后事务可能已进入失败态，先回滚再执行兜底查询
                await self.session.rollback()

        result = await self.session.execute(
            text(
                f"SELECT id, value, column_id FROM {self.table_name} "
                "WHERE value LIKE :pattern LIMIT :limit"
            ),
            {"pattern": f"%{keyword}%", "limit": limit},
        )
        return [ValueInfo(**dict(row)) for row in result.mappings().fetchall()]
