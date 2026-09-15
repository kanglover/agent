# 问数智能体后端镜像
#
# 基础镜像自带 uv 与 Python 3.12，依赖用 uv.lock 锁定以保证构建可复现。
# 说明：Elasticsearch 与本地 Embedding 服务都不进这个镜像——
# 取值召回走 MySQL 全文索引，向量化走 OpenAI 兼容接口，二者都由外部托管。

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# 先只拷贝依赖描述文件，代码变动时这一层仍能命中缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 再拷贝应用代码
COPY . .

# 容器文件系统是临时的，文件日志没有意义，统一走标准输出交给平台采集
ENV LOG_FILE_ENABLE=false \
    LOG_CONSOLE_ENABLE=true

EXPOSE 8000

# Render 等平台会注入 PORT，本地直接跑默认 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
