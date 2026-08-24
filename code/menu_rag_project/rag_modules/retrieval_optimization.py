import logging
import hashlib
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """
    Retrieval Optimization Module.
    """

    def __init__(self, vectorstore: FAISS, documents: list[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.setup_retrievers()

    def setup_retrievers(self):
        """
        Setup the retrievers for the vectorstore.
        """
        logger.info("Setting up retrievers for the vectorstore")

        # 向量检索
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # BM25 检索
        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=5)

        # reranker 重排模型
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        logger.info("Retrievers setup completed")

    def hybrid_search(self, query: str, top_k: int = 3) -> list[Document]:
        """
        Perform hybrid retrieval using both vector and BM25 retrievers.

        Args:
            query (str): The query string for retrieval.
            top_k (int): The number of top documents to retrieve from each retriever.

        Returns:
            list[Document]: A list of retrieved Document objects.
        """
        logger.info(f"Performing hybrid retrieval for query: {query}")

        # 向量检索
        vector_results = self.vector_retriever.invoke(query)
        logger.info(f"Vector retrieval returned {len(vector_results)} documents")

        # BM25 检索
        bm25_results = self.bm25_retriever.invoke(query)
        logger.info(f"BM25 retrieval returned {len(bm25_results)} documents")

        # RRF 重排
        reranked_docs = self.rrf_rerank(vector_results, bm25_results)
        # return reranked_docs[:top_k]

        candidates = reranked_docs[:top_k * 3]
        return self._rerank(query, candidates, top_k=top_k)

    def metadata_filtered_search(
        self, query: str, filters: dict[str, Any], top_k: int = 5
    ) -> list[Document]:
        """
        Perform a metadata-filtered search using the vector retriever.

        Args:
            query (str): The query string for retrieval.
            filters (dict): A dictionary specifying metadata filters.
            top_k (int): The number of top documents to retrieve.
        """
        # 混合检索获取候选集
        docs = self.hybrid_search(query, top_k * 3)

        filtered_docs = []
        for doc in docs:
            if not self._match_filters(doc, filters):
                continue
            filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break

        return filtered_docs

    def _match_filters(self, doc: Document, filters: dict) -> bool:
        """检查单个文档是否满足所有过滤条件"""
        for key, value in filters.items():
            doc_value = doc.metadata.get(key)
            if doc_value is None:
                return False
            # 统一处理：单值包装成列表，避免分支重复
            allowed = value if isinstance(value, list) else [value]
            if doc_value not in allowed:
                return False
        return True

    def rrf_rerank(
        self, vector_results: list[Document], bm25_results: list[Document], k: int = 3
    ) -> list[Document]:
        """
        Rerank the documents using RRF (Randomized Reciprocal Ranking).

        Args:
            vector_results (list[Document]): Documents retrieved by vector retriever.
            bm25_results (list[Document]): Documents retrieved by BM25 retriever.
            top_k (int): The number of top documents to return.

        Returns:
            list[Document]: A reranked list of Document objects.
        """

        logger.info("Starting RRF reranking")

        # 计算每个文档的 RRF 分数
        doc_scores = {}
        doc_objects = {}
        for rank, doc in enumerate(vector_results, 1):
            # 使用文档内容的确定性哈希作为唯一标识
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_objects[doc_id] = doc
            logger.debug(f"向量检索 - 文档{rank} {doc.metadata['dish_name']}: RRF分数 = {1 / (k + rank):.4f}")

        for rank, doc in enumerate(bm25_results, 1):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_objects[doc_id] = doc
            logger.debug(f"BM25检索 - 文档{rank} {doc.metadata['dish_name']}: RRF分数 = {1 / (k + rank):.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata["rrf_score"] = score
                reranked_docs.append(doc)

        logger.info(
            f"RRF reranking completed, returning reranked {len(reranked_docs)} documents"
        )
        return reranked_docs

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Cross-Encoder 精排"""
        if not docs:
            return []

        # 构造 (query, document) 对
        pairs = [(query, doc.page_content) for doc in docs]

        # 批量打分
        scores = self.reranker.predict(pairs)

        # 按分数降序排列，取 top_k
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # 可选：设置最低分数阈值
        THRESHOLD = 0.1
        final = [doc for doc, score in scored_docs[:top_k] if score > THRESHOLD]

        # 将精排分数写入 metadata（便于调试和日志）
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)

        return final