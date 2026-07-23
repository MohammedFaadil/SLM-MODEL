"""Application entrypoint.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Your platform points its OpenAI base_url at  http://<host>:8000/v1
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .backends.base import BackendError
from .backends.factory import close_backend, get_backend
from .config import settings
from .logging_conf import configure_logging, get_logger
from .openai_api.router import router as openai_router
from .persistence import db, repo
from .routes.domain_routes import router as domain_router

configure_logging(settings.log_level)
log = get_logger(__name__)


async def _warmup() -> None:
    """Load the model into (V)RAM so the first real request isn't a cold start."""
    if settings.llm_backend != "openai_upstream":
        return
    try:
        t0 = time.time()
        await get_backend().chat_completion({
            "model": settings.model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        })
        log.info("Warmup complete in %.1fs (model resident).", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Warmup skipped/failed (%s). Model will load on first request.", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("SLM Gateway starting")
    log.info("  backend=%s  model=%s", settings.llm_backend, settings.model_name)
    log.info("  upstream=%s", settings.upstream_base_url)
    log.info("  embeddings=%s (%s)", settings.embeddings_mode, settings.embedding_model)
    log.info("  ocr=%s version=%s  auth=%s",
             settings.ocr_enabled, settings.ocr_version, settings.auth_required)

    # Optional MSSQL persistence (connect + create tables); safe if disabled.
    await run_in_threadpool(db.init)

    if settings.warmup_on_start:
        asyncio.create_task(_warmup())  # non-blocking: service is up immediately

    yield

    await close_backend()
    await run_in_threadpool(db.dispose)
    log.info("SLM Gateway stopped")


app = FastAPI(
    title="SLM Gateway (Qwen3-8B, OpenAI-compatible)",
    version=__version__,
    description="OpenAI-compatible SLM for recruiting: resume OCR, candidate "
    "summaries, and job↔candidate matching. Drop-in replacement — point your "
    "platform's OpenAI base_url at this server's /v1.",
    lifespan=lifespan,
)

# CORS. Defaults to "*" for local dev; set CORS_ALLOW_ORIGINS in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _access_log(request: Request, call_next):
    """Best-effort request/latency audit to MSSQL (no-op when DB disabled)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    if settings.persist_requests and db.available():
        path = request.url.path
        if path.startswith(("/v1", "/api")):
            latency_ms = int((time.perf_counter() - t0) * 1000)
            try:
                await run_in_threadpool(
                    repo.save_request_log,
                    path=path,
                    method=request.method,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    client=request.client.host if request.client else "",
                )
            except Exception:  # noqa: BLE001
                pass
    return response


@app.exception_handler(BackendError)
async def _backend_error_handler(request: Request, exc: BackendError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.err_type,
                "code": exc.status_code,
                "upstream_body": exc.body,
            }
        },
    )


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": "SLM Gateway",
        "version": __version__,
        "model": settings.served_model_id,
        "openai_base_url": "/v1",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "ocr": "/api/ocr/parse",
            "resume": "/api/resume/parse",
            "candidate_summary": "/api/candidate/summary",
            "jobs": "/api/jobs",
            "match": "/api/match",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
@app.get("/healthz", tags=["meta"])
async def health() -> JSONResponse:
    backend_health = await get_backend().health()
    body = {
        "status": "ok" if backend_health.get("ok") else "degraded",
        "version": __version__,
        "backend": backend_health,
        "features": {
            "ocr_enabled": settings.ocr_enabled,
            "ocr_version": settings.ocr_version,
            "embeddings_mode": settings.embeddings_mode,
            "candidate_summary": True,
        },
        "model": settings.served_model_id,
    }
    code = 200 if backend_health.get("ok") else 503
    return JSONResponse(body, status_code=code)


app.include_router(openai_router)
app.include_router(domain_router)
