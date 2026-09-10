"""
应用主配置
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from omegaconf import OmegaConf

@dataclass
class File:
    """文件日志配置"""
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str

@dataclass
class Console:
    """控制台日志配置"""
    enable: bool
    level: str

@dataclass
class LoggingConfig:
    """日志总配置"""
    file: File
    console: Console

@dataclass
class DBConfig:
    """MySQL 连接配置"""

    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class QdrantConfig:
    """Qdrant 连接与向量维度配置"""
    host: str
    port: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    """Embedding 服务配置"""
    host: str
    port: int
    model: str


@dataclass
class ESConfig:
    """Elasticsearch 配置"""
    host: str
    port: int
    index_name: str


@dataclass
class LLMConfig:
    """大模型调用配置"""
    model_name: str
    api_key: str
    base_url: str


@dataclass
class AppConfig:
    """项目级总配置入口"""
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig

project_root = Path(__file__).parents[2]
config_file = project_root / "conf" / "app_config.yaml"

load_dotenv(project_root / ".env")

# 读取 YAML 配置内容
context = OmegaConf.load(config_file)

# 根据 AppConfig 生成结构化配置 schema
schema = OmegaConf.structured(AppConfig)

# 把配置结构和配置值合并，再转换成可以直接按属性访问的对象
app_config: AppConfig = cast(AppConfig, OmegaConf.to_object(OmegaConf.merge(schema, context)))

if __name__ == "__main__":
    print(app_config)