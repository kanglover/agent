"""
应用主配置
"""

import os
import re
import sys
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
    # 托管实例的连接数通常很紧，池大小需要按目标环境下调
    pool_size: int
    # 云上连接容易被中间层断开，定期回收可避免取到失效连接
    pool_recycle: int
    # 托管 MySQL 基本都强制 TLS，本地容器不需要
    ssl: bool
    # 是否校验服务端证书；云上证书链异常时可临时关闭，仅保留加密
    ssl_verify: bool


@dataclass
class QdrantConfig:
    """Qdrant 连接与向量维度配置"""

    # provider 取 local 或 cloud，用于区分本地容器与 Qdrant Cloud 的鉴权方式
    provider: str
    host: str
    port: int
    api_key: str
    use_https: bool
    timeout: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    """Embedding 服务配置"""

    # provider 取 local（TEI 推理服务）或 openai（任意 OpenAI 兼容接口）
    provider: str
    host: str
    port: int
    model: str
    api_key: str
    base_url: str
    batch_size: int


@dataclass
class ValueStoreConfig:
    """字段取值检索配置"""

    # provider 取 mysql 或 es，决定取值召回走哪种存储
    provider: str
    table_name: str
    score_threshold: float
    limit: int


@dataclass
class ESConfig:
    """Elasticsearch 配置"""
    host: str
    port: int
    index_name: str


@dataclass
class CORSConfig:
    """跨域访问配置

    前后端分离部署时前端与后端不同源，浏览器会拦截跨域请求，
    需要显式声明允许的前端来源。
    """

    # 逗号分隔的来源白名单，由 main.py 拆分后交给 CORSMiddleware
    allow_origins: str
    # 正则来源，用于匹配 Vercel 预览域名这类不固定的地址
    allow_origin_regex: str
    # 是否允许携带凭据；开启后来源不能用 *
    allow_credentials: bool


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
    value_store: ValueStoreConfig
    es: ESConfig
    cors: CORSConfig
    llm: LLMConfig

def _check_required_env(yaml_text: str) -> None:
    """
    提前检查必填环境变量是否都在

    YAML 里写成 ${oc.env:XXX}（不给默认值）的都算必填。
    这类变量缺失时 OmegaConf 只会抛一段很深的插值异常，在容器里表现为
    uvicorn 直接退出、日志里全是 omegaconf 内部调用栈，很难看出到底缺了什么。
    这里在加载配置前先做一次检查，把缺失项一次性列清楚。
    """

    # 只匹配 ${oc.env:XXX} 这种没有逗号默认值的写法；
    # 写成 ${oc.env:XXX,默认值} 的不算必填，所以不会命中
    required = set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}", yaml_text))
    missing = sorted(name for name in required if not os.environ.get(name))

    if missing:
        lines = [
            "",
            "配置加载失败：以下必填环境变量未设置 ——",
            *[f"  - {name}" for name in missing],
            "",
            "本地开发：把它们写进项目根目录的 .env（可参考 .env.example）",
            "容器部署：在平台的环境变量里补齐（Render 见根目录 render.yaml 的 envVars）",
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1)


project_root = Path(__file__).parents[2]
config_file = project_root / "conf" / "app_config.yaml"

load_dotenv(project_root / ".env")

# 读取 YAML 原文：既用于缺变量检查，也用于后续解析
config_text = config_file.read_text(encoding="utf-8")
_check_required_env(config_text)

# 读取 YAML 配置内容
context = OmegaConf.load(config_file)

# 根据 AppConfig 生成结构化配置 schema
schema = OmegaConf.structured(AppConfig)

# 把配置结构和配置值合并，再转换成可以直接按属性访问的对象
app_config: AppConfig = cast(AppConfig, OmegaConf.to_object(OmegaConf.merge(schema, context)))

if __name__ == "__main__":
    print(app_config)