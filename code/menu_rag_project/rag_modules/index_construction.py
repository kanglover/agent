import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """
    Index Construction Module.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
    ):
        self.embedding_model_name = embedding_model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """
        Setup the embeddings model.
        """
        logger.info(f"Initializing embeddings model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Embeddings model {self.embedding_model_name} initialized")

    def build_vector_index(self, documents: list[Document]) -> FAISS:
        """
        Build a vector index from the provided documents.
        """
        logger.info(f"Building vector index with {len(documents)} documents")

        if not documents:
            raise ValueError("No documents provided for index construction.")

        if not self.embeddings:
            raise ValueError("Embeddings model is not initialized.")

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vector index built with {len(documents)} documents")
        return self.vectorstore

    def add_documents(self, new_documents: list[Document]):
        """
        Add new documents to the existing vector index.
        """
        if not self.vectorstore:
            raise ValueError("Vector index is not initialized. Build it first.")

        if not new_documents:
            logger.warning("No new documents provided to add.")
            return

        self.vectorstore.add_documents(new_documents)
        logger.info(f"Added {len(new_documents)} new documents to the vector index")

    def save_index(self):
        """
        Save the vector index to disk.
        """
        if not self.vectorstore:
            raise ValueError("Vector index is not initialized. Build it first.")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"Vector index saved to {self.index_save_path}")

    def load_index(self):
        """
        Load the vector index from disk.
        """
        if not self.embeddings:
            raise ValueError(
                "Embeddings model is not initialized. Initialize it first."
            )

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Vector index loaded from {self.index_save_path}")
            return self.vectorstore
        except Exception as e:
            logger.error(
                f"Failed to load vector index from {self.index_save_path}: {e}"
            )
            raise e

    def similarity_search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Perform a similarity search on the vector index.
        """
        if not self.vectorstore:
            raise ValueError("Vector index is not initialized. Build or load it first.")

        if not query:
            raise ValueError("Query string is empty.")

        results = self.vectorstore.similarity_search(query, k=top_k)
        logger.info(f"Performed similarity search for query: '{query}'")
        return results
