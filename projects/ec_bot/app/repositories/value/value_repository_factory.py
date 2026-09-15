"""
字段取值检索仓储工厂

按 ``value_store.provider`` 决定取值召回落在 Elasticsearch 还是 MySQL 上，
让依赖注入层和元数据构建脚本用同一段逻辑拿到仓储实例，避免两处各写一次分支。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.es_client_manager import es_client_manager
from app.conf.app_config import app_config
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.value.value_mysql_repository import ValueMySQLRepository
from app.repositories.value.value_repository import ValueRepository


def create_value_repository(session: AsyncSession) -> ValueRepository:
    """创建字段取值检索仓储

    Args:
        session: 复用的 Meta MySQL 会话；MySQL 实现需要它来读写取值索引表，
            ES 实现不使用该参数。
    """

    if app_config.value_store.provider == "es":
        client = es_client_manager.client
        if client is None:
            raise RuntimeError("value_store.provider=es 但 Elasticsearch 客户端尚未初始化")
        return ValueESRepository(client)

    return ValueMySQLRepository(session)
