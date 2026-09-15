"""
FastAPI 应用入口

负责创建后端应用实例，注册应用生命周期函数，并把各业务模块中的 router
挂载到同一个 app 上。HTTP 请求会先进入这里创建的 app，再按路由分发到
具体的接口处理函数。
"""

import uuid
from typing import cast
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.lifespan import lifespan
from app.api.routers.query_router import query_router
from app.conf.app_config import app_config
from app.core.context import request_id_ctx_var
from app.core.log import logger

# lifespan 交给 FastAPI 管理，用于在服务启动和关闭时统一初始化与释放外部客户端
app = FastAPI(lifespan=lifespan)


def _split_origins(raw: str) -> list[str]:
    """把逗号分隔的来源字符串拆成列表，忽略空白项"""

    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# 前端部署在 Vercel、后端部署在 Render，两者不同源；
# 不加这层中间件时，浏览器会把 /api/query 的请求直接判定为跨域而拦截。
_cors_origins = _split_origins(app_config.cors.allow_origins)
if not _cors_origins and not app_config.cors.allow_origin_regex:
    logger.warning(
        "未配置 CORS_ALLOW_ORIGINS，浏览器将拦截来自前端域名的跨域请求；"
        "部署前后端分离架构时请务必设置该变量"
    )

# CORSMiddleware 会自动响应 OPTIONS 预检请求，SSE 流式接口发出的
# application/json POST 也会先走一次预检，因此这里必须挂载。
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # 为空时传 None，避免把空字符串当成非法正则
    allow_origin_regex=app_config.cors.allow_origin_regex or None,
    allow_credentials=app_config.cors.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 把查询路由注册进应用；没有挂载时，/docs 和真实 HTTP 请求都访问不到该接口
app.include_router(query_router)


@app.get("/health")
async def health():
    """健康检查接口

    托管平台（如 Render）会用这个地址探活，只返回进程存活状态，
    不触发任何外部依赖访问，避免外部服务抖动导致容器被误判为不健康。
    """

    return {"status": "ok"}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # 请求被处理之前
    request_id = uuid.uuid4()
    request_id_ctx_var.set(cast(str, request_id))
    response = await call_next(request)
    # 请求被处理之后
    return response
