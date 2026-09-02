import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """
    Recipe RAG System.
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY", ""):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    def initialize_system(self):
        """
        Initialize the RAG system.
        """
        logger.info("Initializing Recipe RAG System")

        # 初始化数据模块
        self.data_module = DataPreparationModule(data_path=self.config.data_path)
        logger.info(f"数据模块初始化成功{self.config.embedding_model}")
        # 初始化索引构建模块
        self.index_module = IndexConstructionModule(
            embedding_model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        # 初始化生成集成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        logger.info("Recipe RAG System initialized successfully")

    def build_knowledge_base(self):
        """
        Build the knowledge base.
        """
        logger.info("Building Knowledge Base")

        if not (self.index_module and self.data_module and self.generation_module):
            raise RuntimeError("系统未初始化，请先调用 initialize_system()")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            logger.info("已加载现有索引")
            print("加载食谱文档...")
            documents = self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents(documents)
        else:
            # 2. 若不存在，则构建新的索引
            print("加载食谱文档...")
            documents = self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents(documents)
            print("构建索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            print("保存索引...")
            self.index_module.save_index()
            logger.info("新索引构建并保存成功")

        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """
        Ask a question to the RAG system.
        """
        if not (self.retrieval_module and self.generation_module and self.data_module):
            raise RuntimeError("检索优化模块或生成集成模块未初始化，请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 根据问题类型选择查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        rewritten_query = question
        if route_type != "list":
            rewritten_query = self.generation_module.query_rewrite(question)
            print(f"重写的查询: {rewritten_query}")

        # 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        print(f"应用元数据过滤: {filters}")

        if filters:
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith("#"):
                    title_end = (
                        content_preview.find("\n")
                        if "\n" in content_preview
                        else len(content_preview)
                    )
                    section_title = content_preview[:title_end].replace("#", "").strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name} - {content_preview}")

            print(f"检索到 {len(relevant_chunks)} 个相关文档: {', '.join(chunk_info)}")
        else:
            print(f"检索到 {len(relevant_chunks)} 个相关文档")

        if not relevant_chunks:
            print("未找到相关信息，请尝试其他问题。")
            return ""

        # 生成答案
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = [doc.metadata.get("dish_name", "未知菜品") for doc in relevant_docs]

        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")
        elif route_type != "list":
            print(f"对应 {len(relevant_docs)} 个完整文档")

        # 根据路由类型分发处理
        if route_type == "list":
            print("📋 生成菜品列表...")
            return self.generation_module.generate_list_answer(question, relevant_docs)

        print("✍️ 生成详细回答...")
        generate_fn = {
            (
                "detail",
                True,
            ): self.generation_module.generate_step_by_step_answer_stream,
            ("detail", False): self.generation_module.generate_step_by_step_answer,
            ("general", True): self.generation_module.generate_basic_answer_stream,
            ("general", False): self.generation_module.generate_basic_answer,
        }[(route_type, stream)]

        return generate_fn(question, relevant_docs)

    def _extract_filters_from_query(self, query: str):
        """
        Extract metadata filters from the user's question.
        """
        filters = {}

        # 分类：直接匹配（类别间通常互斥，首个命中即可）
        for cat in DataPreparationModule.get_supported_categories():
            if cat in query:
                filters["category"] = cat
                break

        # 难度：按长度降序匹配，避免"简单"误匹配"简单易学"中的"简单"
        for diff in sorted(
            DataPreparationModule.get_supported_difficulties(), key=len, reverse=True
        ):
            if diff in query:
                filters["difficulty"] = diff
                break

        return filters

    def run_interactive(self):
        """
        Run an interactive session with the RAG system.
        """
        self.initialize_system()
        self.build_knowledge_base()

        while True:
            try:
                user_input = input("\n🤖 请输入你的问题（或 'exit' 退出）: ")
                if user_input.lower() == "exit":
                    print("再见！")
                    break

                stream_choice = (
                    input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                )
                use_stream = stream_choice != "n"

                answer = self.ask_question(user_input, stream=use_stream)
                if use_stream:
                    for chunk in answer:
                        print(chunk, end="", flush=True)
                else:
                    print("\n📝 回答:")
                    print(answer)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"发生错误：{e}")
                continue


def main():
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"发生错误：{e}")


if __name__ == "__main__":
    main()
