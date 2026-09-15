"""
Embedding 客户端管理器

按 ``embedding.provider`` 在两种实现之间切换，业务层只依赖 aembed_documents /
aembed_query 这套统一接口：

- ``local``：本地 TEI 推理服务，沿用 ``HuggingFaceEndpointEmbeddings`` + ``/embed`` 端点
- ``openai``：任意 OpenAI 兼容的 embedding 接口，免去本地加载 1GB+ 模型的资源开销

这样部署环境只改配置就能换掉 embedding 后端，召回链路的代码不用动。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from typing import Optional

from huggingface_hub import AsyncInferenceClient, InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    """负责按当前配置创建并持有 Embedding 客户端"""

    def __init__(self, config: EmbeddingConfig):
        # 客户端在模块导入阶段先不立即创建，避免启动时就发起外部依赖连接
        self.client: Optional[Embeddings] = None
        # 保存 Embedding 服务配置，供 init() 时组装客户端使用
        self.config = config

    def _get_url(self) -> str:
        # 本地 TEI 部署通过 host + port 访问已启动的推理服务
        return f"http://{self.config.host}:{self.config.port}"

    def _build_local_client(self) -> Embeddings:
        """构建指向本地 TEI 服务的客户端"""
        # HuggingFaceEndpointEmbeddings 的 model 参数只接受 HF repo ID，
        # 实际的推理端点 URL 需要直接注入到底层 InferenceClient 中
        client = HuggingFaceEndpointEmbeddings(model=self.config.model)
        # TEI 的 embedding 端点是 /embed，InferenceClient 会直接向 model URL 发 POST
        endpoint_url = f"{self._get_url()}/embed"
        client.client = InferenceClient(model=endpoint_url)
        client.async_client = AsyncInferenceClient(model=endpoint_url)
        return client

    def _build_openai_client(self) -> Embeddings:
        """构建指向 OpenAI 兼容接口的客户端"""
        if not self.config.api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_KEY")
        if not self.config.base_url:
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_BASE")

        return OpenAIEmbeddings(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            # 关闭按 tiktoken 预切分的长度校验：非 OpenAI 官方接口拿不到本地分词器，
            # 打开会导致调用直接失败；关掉后请求体会原样发给兼容网关
            check_embedding_ctx_length=False,
        )

    def init(self) -> None:
        """在应用启动阶段显式调用，完成真正的客户端初始化"""
        if self.config.provider == "openai":
            self.client = self._build_openai_client()
        else:
            self.client = self._build_local_client()


# 模块级单例，供其他模块按需复用同一个客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == "__main__":
    # 本地调试入口：初始化客户端后执行一次最小化向量化调用
    embedding_client_manager.init()
    client = embedding_client_manager.client

    async def test():
        if client is None:
            raise ValueError("Embedding client not initialized")

        # 使用示例文本验证 Embedding 服务是否可正常响应
        text = "What is deep learning?"
        query_result = await client.aembed_query(text)
        # 只打印前 3 个维度与总维度，便于快速确认返回结果结构正确
        print(query_result[:3], len(query_result))

    # 运行调试测试
    asyncio.run(test())
