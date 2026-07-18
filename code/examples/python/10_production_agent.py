"""
生产级 Agent 模板
=================
功能清单：
  - FastAPI 应用（POST /chat、GET /health）
  - 异步 Agent（asyncio）
  - 并发会话管理（字典存储多用户）
  - 结构化日志（trace_id、user_id、token_count）
  - 请求追踪（x-trace-id header）
  - 配置管理（pydantic BaseSettings + .env）
  - Graceful shutdown
  - 错误处理中间件

运行方式：
  pip install fastapi uvicorn anthropic pydantic-settings python-dotenv
  uvicorn 10_production_agent:app --reload --port 8000

测试：
  curl -X POST http://localhost:8000/chat \
       -H "Content-Type: application/json" \
       -H "x-trace-id: my-trace-001" \
       -d '{"user_id": "alice", "message": "你好，请介绍一下你自己"}'

  curl http://localhost:8000/health
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 1. 配置管理（pydantic BaseSettings）
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """从环境变量 / .env 文件读取配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = ""
    model: str = "claude-opus-4-5"
    max_tokens: int = 1024
    system_prompt: str = "你是一个有帮助的 AI 助手，请用简洁清晰的中文回答问题。"

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    max_history_turns: int = 20          # 每个会话保留的最大轮次
    session_ttl_seconds: int = 3600      # 会话超时时间（秒）


settings = Settings()


# ---------------------------------------------------------------------------
# 2. 结构化日志
# ---------------------------------------------------------------------------

class StructuredLogger:
    """输出 JSON 格式日志，包含 trace_id、user_id、token_count 等字段。"""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            **kwargs,
        }
        self._logger.log(
            getattr(logging, level),
            json.dumps(record, ensure_ascii=False),
        )

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)


logger = StructuredLogger("production_agent")


# ---------------------------------------------------------------------------
# 3. 会话管理
# ---------------------------------------------------------------------------

class Session:
    """单个用户的对话会话。"""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.history: list[dict[str, str]] = []
        self.created_at = time.time()
        self.last_active = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.last_active = time.time()
        # 超出限制时裁剪最早的轮次（保留系统完整性：成对裁剪）
        while len(self.history) > settings.max_history_turns * 2:
            self.history.pop(0)
            self.history.pop(0)

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > settings.session_ttl_seconds

    def stats(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "turns": len(self.history) // 2,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "created_at": datetime.utcfromtimestamp(self.created_at).isoformat() + "Z",
            "last_active": datetime.utcfromtimestamp(self.last_active).isoformat() + "Z",
        }


class SessionStore:
    """线程安全（asyncio）的多用户会话存储。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, user_id: str) -> Session:
        async with self._lock:
            session = self._sessions.get(user_id)
            if session is None or session.is_expired():
                session = Session(user_id)
                self._sessions[user_id] = session
                logger.info("session_created", user_id=user_id)
            return session

    async def delete(self, user_id: str) -> None:
        async with self._lock:
            self._sessions.pop(user_id, None)

    async def purge_expired(self) -> int:
        async with self._lock:
            expired = [uid for uid, s in self._sessions.items() if s.is_expired()]
            for uid in expired:
                del self._sessions[uid]
            return len(expired)

    def count(self) -> int:
        return len(self._sessions)


session_store = SessionStore()


# ---------------------------------------------------------------------------
# 4. 异步 Agent
# ---------------------------------------------------------------------------

class ProductionAgent:
    """异步 Agent，使用 Anthropic Messages API。"""

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None  # None → 读取环境变量
        )

    async def chat(
        self,
        session: Session,
        user_message: str,
        trace_id: str,
    ) -> str:
        session.add_turn("user", user_message)

        t0 = time.perf_counter()
        response = await self._client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=settings.system_prompt,
            messages=session.history,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        assistant_text = response.content[0].text
        session.add_turn("assistant", assistant_text)

        # 累计 token 计数
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        session.total_input_tokens += input_tokens
        session.total_output_tokens += output_tokens

        logger.info(
            "agent_response",
            trace_id=trace_id,
            user_id=session.user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed_ms,
            model=settings.model,
        )

        return assistant_text


agent = ProductionAgent()


# ---------------------------------------------------------------------------
# 5. 后台清理任务
# ---------------------------------------------------------------------------

async def _cleanup_loop() -> None:
    """每 5 分钟清理一次过期会话。"""
    while True:
        await asyncio.sleep(300)
        removed = await session_store.purge_expired()
        if removed:
            logger.info("sessions_purged", count=removed, active=session_store.count())


# ---------------------------------------------------------------------------
# 6. FastAPI 生命周期（含 Graceful Shutdown）
# ---------------------------------------------------------------------------

_background_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 / 关闭钩子。"""
    logger.info("app_starting", model=settings.model, port=settings.port)

    # 启动后台任务
    task = asyncio.create_task(_cleanup_loop())
    _background_tasks.append(task)

    # 注册系统信号以支持 graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))

    yield  # 应用运行期间

    # 关闭时取消后台任务
    for t in _background_tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    logger.info("app_stopped")


async def _shutdown() -> None:
    logger.info("graceful_shutdown_initiated")
    # 给进行中的请求最多 5 秒完成
    await asyncio.sleep(5)
    sys.exit(0)


# ---------------------------------------------------------------------------
# 7. FastAPI 应用与中间件
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Production Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 错误处理中间件
@app.middleware("http")
async def error_and_tracing_middleware(request: Request, call_next):
    # 生成或透传 trace_id
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
    request.state.trace_id = trace_id

    t0 = time.perf_counter()
    try:
        response: Response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "http_request",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        response.headers["x-trace-id"] = trace_id
        return response
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "unhandled_exception",
            trace_id=trace_id,
            path=request.url.path,
            error=str(exc),
            elapsed_ms=elapsed_ms,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "trace_id": trace_id},
            headers={"x-trace-id": trace_id},
        )


# ---------------------------------------------------------------------------
# 8. 请求 / 响应模型
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    message: str = Field(..., min_length=1, description="用户输入的消息")
    reset: bool = Field(False, description="是否重置当前会话历史")


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    trace_id: str
    session_stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    model: str
    timestamp: str


# ---------------------------------------------------------------------------
# 9. 路由
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request) -> ChatResponse:
    trace_id: str = getattr(request.state, "trace_id", str(uuid.uuid4()))

    logger.info(
        "chat_request",
        trace_id=trace_id,
        user_id=body.user_id,
        message_len=len(body.message),
    )

    if body.reset:
        await session_store.delete(body.user_id)
        logger.info("session_reset", trace_id=trace_id, user_id=body.user_id)

    session = await session_store.get_or_create(body.user_id)

    try:
        reply = await agent.chat(session, body.message, trace_id)
    except anthropic.AuthenticationError:
        logger.error("auth_error", trace_id=trace_id, user_id=body.user_id)
        raise HTTPException(status_code=401, detail="Anthropic API key 无效或未配置")
    except anthropic.RateLimitError:
        logger.warning("rate_limit", trace_id=trace_id, user_id=body.user_id)
        raise HTTPException(status_code=429, detail="请求频率超限，请稍后重试")
    except anthropic.APIError as exc:
        logger.error("api_error", trace_id=trace_id, user_id=body.user_id, error=str(exc))
        raise HTTPException(status_code=502, detail=f"上游 API 错误：{exc}")

    return ChatResponse(
        user_id=body.user_id,
        reply=reply,
        trace_id=trace_id,
        session_stats=session.stats(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    return HealthResponse(
        status="ok",
        active_sessions=session_store.count(),
        model=settings.model,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.delete("/session/{user_id}", status_code=204)
async def delete_session(user_id: str, request: Request) -> None:
    """手动删除指定用户的会话（方便测试）。"""
    trace_id: str = getattr(request.state, "trace_id", str(uuid.uuid4()))
    await session_store.delete(user_id)
    logger.info("session_deleted_manual", trace_id=trace_id, user_id=user_id)


# ---------------------------------------------------------------------------
# 10. 直接运行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "10_production_agent:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # 生产环境改用 workers=4，关闭 reload
        reload=True,
    )
