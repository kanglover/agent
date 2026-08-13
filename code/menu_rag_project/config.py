"""
RAG Configuration file.
"""

from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class RAGConfig:
    """
    Configuration for RAG.
    """
    data_path: str = "./data"
    index_save_path: str = "./vector_index"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = ""

    top_k: int = 3

    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RAGConfig":
        """
        Create a RAGConfig instance from a dictionary.
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the RAGConfig instance to a dictionary.
        """
        return asdict(self)

DEFAULT_CONFIG = RAGConfig()