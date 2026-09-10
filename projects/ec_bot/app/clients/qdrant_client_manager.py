"""
Qdrant 客户端管理器
"""

import asyncio
import random
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import QdrantConfig, app_config

class QdrantClientManager:
    """管理 Qdrant 客户端的初始化与关闭"""

    def __init__(self, qdrant_config: QdrantConfig):
        self.qdrant_config = qdrant_config
        self.client: Optional[AsyncQdrantClient] = None

    def __get_url(self) -> str:
        """拼接 Qdrant 服务地址"""
        return f"http://{self.qdrant_config.host}:{self.qdrant_config.port}"

    def init(self):
        """
        初始化 Qdrant 客户端
        这里不在 __init__ 中初始化，是为了和项目的生命周期管理保持一致
        """
        self.client = AsyncQdrantClient(self.__get_url())

    async def close(self):
        """关闭 Qdrant 客户端连接"""
        if self.client is not None:
            await self.client.close()

qdrant_client_manager = QdrantClientManager(app_config.qdrant)

if __name__ == "__main__":
    qdrant_client_manager.init()

    async def test():
        client = qdrant_client_manager.client

        if client is None:
            print("Error: Qdrant client is not initialized")
            return

        if not await client.collection_exists("my_collection"):
            await client.create_collection(
                collection_name="my_collection",
                vectors_config=models.VectorParams(
                    # 当前集合中的向量维度是 10
                    size=10,
                    # 使用余弦相似度作为距离计算方式
                    distance=models.Distance.COSINE,
                ),
            )

        # 向集合中写入 100 个随机 point
        # 每个 point 都有一个 id 和一个 10 维向量
        await client.upsert(
            collection_name="my_collection",
            points=[
                models.PointStruct(
                    id=i,
                    vector=[random.random() for _ in range(10)],
                )
                for i in range(100)
            ],
        )

        # 用一个随机生成的查询向量做相似度检索
        # limit=10 表示最多返回 10 条结果
        # score_threshold=0.8 表示只保留分数不低于 0.8 的结果
        res = await client.query_points(
            collection_name="my_collection",
            query=[random.random() for _ in range(10)],  # type: ignore
            limit=10,
            score_threshold=0.8,
        )

        print(res)

    asyncio.run(test())