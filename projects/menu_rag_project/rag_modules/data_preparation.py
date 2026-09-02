import logging
import hashlib
import re
from pathlib import Path
from typing import Any, ClassVar

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """
    数据准备模块。
    """

    CATEGORY_MAPPING: ClassVar[dict[str, str]] = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }

    CATEGORY_LABELS: ClassVar[list[str]] = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_MAPPING: ClassVar[dict[int, str]] = {
        5: "非常困难",
        4: "困难",
        3: "中等",
        2: "简单",
        1: "非常简单",
    }
    DIFFICULTY_LABELS: ClassVar[list[str]] = list(set(DIFFICULTY_MAPPING.values()))

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents: list[Document] = []  # 父文档（完整食谱）
        self.chunks: list[Document] = []  # 子文档（按标题分割的小块）
        self.parent_child_map: dict[str, str] = {}  # 子块ID -> 父文档ID的映射

    def load_documents(self) -> list[Document]:
        """
        加载文档。
        """
        documents = []
        for file_path in self.data_path.glob("**/*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                try:
                    data_root = self.data_path.resolve()
                    relative_path = (
                        file_path.resolve().relative_to(data_root).as_posix()
                    )
                except ValueError:
                    relative_path = file_path.as_posix()

                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "parent_id": parent_id,
                        "doc_type": "parent",  # 标记为父文档
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"读取文件时出错 {file_path}: {e}")

        for doc in documents:
            doc = self._enhance_metadata(doc)
            self.documents.append(doc)
            # logger.info(f"文档元数据 {doc.metadata}")

        logger.info(f"加载了 {len(documents)} 个文档")
        return self.documents

    def _enhance_metadata(self, doc: Document) -> Document:
        """
        增强文档的元数据。
        """
        metadata = doc.metadata.copy()
        content = doc.page_content
        file_path = Path(metadata.get("source", ""))
        path_parts = file_path.parts

        # 提取类别
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                metadata["category"] = value
                break
        else:
            metadata["category"] = "其他"

        # 提取菜名
        metadata["dish_name"] = file_path.stem  # 文件名作为菜名

        # 提取难度
        difficulty_match = re.search(r"★+", content)
        if difficulty_match:
            star_count = len(difficulty_match.group())
            metadata["difficulty"] = self.DIFFICULTY_MAPPING.get(star_count, "未知")
        else:
            metadata["difficulty"] = "未知"

        return Document(page_content=content, metadata=metadata)

    @classmethod
    def get_supported_categories(cls) -> list[str]:
        """
        获取支持的类别。
        """
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> list[str]:
        """
        获取支持的难度等级。
        """
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Markdown 结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info(f"开始对 {len(documents)} 个文档进行分块")

        if not documents:
            raise ValueError("文档列表为空，无法进行分块")

        chunks = self._markdown_header_split(documents)

        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk_id = str(uuid.uuid4())
                chunk.metadata["chunk_id"] = chunk_id
                logger.warning(f"分块 {i} 缺少 'chunk_id'，已自动生成: {chunk_id}")
            chunk.metadata["batch_index"] = i  # 添加批次索引，整个批次全局
            chunk.metadata["chunk_size"] = len(chunk.page_content)  # 添加块大小信息

        self.chunks = chunks
        logger.info(f"分块完成，总共生成 {len(chunks)} 块")
        return chunks

    def _markdown_header_split(self, documents: list[Document]) -> list[Document]:
        """
        根据 Markdown 标题进行分块。
        """

        headers_to_split_on = [
            ("#", "主标题"),  # 菜品名称
            ("##", "二级标题"),  # 必备原料、计算、操作等
            ("###", "三级标题"),  # 简易版本、复杂版本等
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # 保留标题
        )

        chunks = []

        for doc in documents:
            try:
                doc_chunks = markdown_splitter.split_text(doc.page_content)
                logger.info(
                    f"文档 {doc.metadata.get('source')} 分块成功，生成 {len(doc_chunks)} 块"
                )

                if len(doc_chunks) <= 1:
                    logger.warning(
                        f"文档 {doc.metadata.get('source')} 分块后只有 1 块，可能内容过少或格式不规范"
                    )

                parent_id = doc.metadata.get("parent_id", "")  # 父文档ID

                for i, chunk in enumerate(doc_chunks):
                    child_id = str(uuid.uuid4())

                    chunk.metadata.update(doc.metadata)  # 继承父文档的元数据
                    chunk.metadata.update(
                        {
                            "chunk_index": i,
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",  # 标记为子文档
                        }
                    )

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                chunks.extend(doc_chunks)

            except Exception as e:
                logger.error(f"文档 {doc.metadata.get('source')} 分块失败: {e}")
                chunks.append(doc)  # 添加原始文档到结果列表

        logger.info(f"总共生成 {len(chunks)} 块")
        return chunks

    def filter_documents_by_category(
        self, documents: list[Document], category: str
    ) -> list[Document]:
        """
        根据类别过滤文档。
        """
        filtered_docs = [
            doc for doc in documents if doc.metadata.get("category") == category
        ]
        logger.info(f"根据类别 '{category}' 过滤后，剩余 {len(filtered_docs)} 个文档")
        return filtered_docs

    def filter_documents_by_difficulty(
        self, documents: list[Document], difficulty: str
    ) -> list[Document]:
        """
        根据难度过滤文档。
        """
        filtered_docs = [
            doc for doc in documents if doc.metadata.get("difficulty") == difficulty
        ]
        logger.info(f"根据难度 '{difficulty}' 过滤后，剩余 {len(filtered_docs)} 个文档")
        return filtered_docs

    def get_statistics(self) -> dict[str, Any]:
        """
        获取数据统计信息。
        """

        if not self.documents:
            return {"total_documents": 0, "total_chunks": 0}

        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": {
                cat: 0 for cat in self.CATEGORY_LABELS
            },  # 初始化类别计数器，默认为0
            "difficulties": {
                diff: 0 for diff in self.DIFFICULTY_LABELS
            },  # 初始化难度计数器，默认为0
            "avg_chunk_size": sum(
                chunk.metadata.get("chunk_size", 0) for chunk in self.chunks
            )
            / len(self.chunks)
            if self.chunks
            else 0,
        }

        for doc in self.documents:
            category = doc.metadata.get("category", "其他")
            difficulty = doc.metadata.get("difficulty", "未知")
            if category in stats["categories"]:
                stats["categories"][category] += 1
            if difficulty in stats["difficulties"]:
                stats["difficulties"][difficulty] += 1

        return stats

    def export_metadata(self, output_path: str) -> None:
        """
        导出元数据到 JSON 文件。
        """
        import json

        metadata = {
            "source": [doc.metadata.get("source") for doc in self.documents],
            "dish_name": [doc.metadata.get("dish_name") for doc in self.documents],
            "category": [doc.metadata.get("category") for doc in self.documents],
            "difficulty": [doc.metadata.get("difficulty") for doc in self.documents],
            "content_length": [len(doc.page_content) for doc in self.documents],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"元数据已导出到 {output_path}")

    # 这个函数的核心作用是：从一批检索到的子块（child chunks）反查出它们所属的父文档，并按"命中子块数量"排序返回去重后的父文档列表。
    # 这是 RAG 中经典的 Parent Document Retrieval 模式的关键环节。下面逐段拆解：
    # 输入: [子块A, 子块B, 子块C, 子块D]     ← 向量检索命中的结果
    #         ↓         ↓       ↓       ↓
    #       父文档1   父文档2  父文档1  父文档3
    #         ↓
    # 输出: [父文档1(2块), 父文档2(1块), 父文档3(1块)]  ← 去重+排序
    def get_parent_documents(self, child_chunks: list[Document]) -> list[Document]:
        """
        根据子块获取对应的父文档。
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                # ① 计数：父文档被命中了多少个子块
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                # ② 缓存：第一次遇到时查找并缓存父文档，避免重复遍历
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],  # 按命中次数排序
            reverse=True,  # 降序：命中多的排前面
        )

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(
            f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}"
        )
        return parent_docs
