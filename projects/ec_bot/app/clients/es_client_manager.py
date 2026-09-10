"""
Elasticsearch 客户端管理器
"""

import asyncio
from typing import Optional

from elasticsearch import AsyncElasticsearch

from app.conf.app_config import ESConfig, app_config

class ESClientManager:
    """Elasticsearch 客户端管理器类，用于管理 Elasticsearch 的异步连接。

    Attributes:
        client (AsyncElasticsearch): Elasticsearch 的异步客户端实例。
    """

    def __init__(self, es_config: ESConfig = app_config.es) -> None:
        """初始化 ESClientManager 类。

        Args:
            es_config (ESConfig): Elasticsearch 的配置信息，默认为全局的 Elasticsearch 配置。
        """
        self.es_config = es_config
        self.client: Optional[AsyncElasticsearch] = None

    def get_url(self) -> str:
        """获取 Elasticsearch 的连接 URL。

        Returns:
            str: 连接的 URL，格式为 "http://host:port"。
        """
        return f"http://{self.es_config.host}:{self.es_config.port}"

    def init(self) -> None:
        """初始化 Elasticsearch 的异步连接。"""
        self.client = AsyncElasticsearch(hosts=[self.get_url()])

    async def close(self) -> None:
        """关闭与 Elasticsearch 的连接。

        如果当前没有打开的连接，此方法将不做任何操作。
        """
        if self.client is not None:
            await self.client.close()

es_client_manager = ESClientManager(app_config.es)

if __name__ == "__main__":
    es_client_manager.init()

    async def test():
        client = es_client_manager.client

        try:
            if client is None:
                print("Error: Elasticsearch client is not initialized")
                return

            if not await client.indices.exists(index="my-books"):
                # 这里同时显式定义了字段结构
                # dynamic=False 表示关闭动态映射，要求写入数据必须符合当前定义
                await client.indices.create(
                    index="my-books",
                    mappings={
                        "dynamic": False,
                        "properties": {
                            "name": {"type": "text"},
                            "author": {"type": "text"},
                            "release_date": {"type": "date", "format": "yyyy-MM-dd"},
                            "page_count": {"type": "integer"},
                        },
                    },
                )
                # 插入数据
                # bulk 采用“操作说明 + 数据本体”交替出现的格式
                # 适合一次性写入多条文档
                await client.bulk(
                    operations=[
                        {"index": {"_index": "my-books"}},
                        {
                            "name": "Revelation Space",
                            "author": "Alastair Reynolds",
                            "release_date": "2000-03-15",
                            "page_count": 585,
                        },
                        {"index": {"_index": "my-books"}},
                        {
                            "name": "1984",
                            "author": "George Orwell",
                            "release_date": "1985-06-01",
                            "page_count": 328,
                        },
                        {"index": {"_index": "my-books"}},
                        {
                            "name": "Fahrenheit 451",
                            "author": "Ray Bradbury",
                            "release_date": "1953-10-15",
                            "page_count": 227,
                        },
                        {"index": {"_index": "my-books"}},
                        {
                            "name": "Brave New World",
                            "author": "Aldous Huxley",
                            "release_date": "1932-06-01",
                            "page_count": 268,
                        },
                        {"index": {"_index": "my-books"}},
                        {
                            "name": "The Handmaids Tale",
                            "author": "Margaret Atwood",
                            "release_date": "1985-06-01",
                            "page_count": 311,
                        },
                    ],
                )

                resp = await client.search(
                    index="my-books",
                    query={"match": {"name": "brave"}},
                )
                print(resp)

        except Exception as e:
            print(e)
        finally:
            # 成功或异常都关闭连接，避免 aiohttp ClientSession 未关闭告警
            await es_client_manager.close()

    asyncio.run(test())