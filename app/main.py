"""Application entrypoint.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Your platform points its OpenAI base_url at  http://<host>:8000/v1
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .backends.base import BackendError
from .backends.factory import close_backend, get_backend
from .config import settings
from .logging_conf import configure_logging, get_logger
from .openai_api.router import router as openai_router
from .routes.domain_routes import router as domain_router

configure_logging(settings.log_level)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("SLM Gateway starting")
    log.info("  backend=%s  model=%s", settings.llm_backend, settings.model_name)
    log.info("  upstream=%s", settings.upstream_base_url)
    log.info("  embeddings=%s (%s)", settings.embeddings_mode, settings.embedding_model)
    log.info("  ocr=%s version=%s  auth=%s",
             settings.ocr_enabled, settings.ocr_version, settings.auth_required)
    yield
    await close_backend()
    log.info("SLM Gateway stopped")


app = FastAPI(
    title="SLM Gateway (Qwen3-8B, OpenAI-compatible)",
    version="0.1.0",
    description="OpenAI-compatible SLM for recruiting: resume OCR, candidate "
    "summaries, and job↔candidate matching. Drop-in replacement — point your "
    "platform's OpenAI base_url at this server's /v1.",
    lifespan=lifespan,
)

# Permissive CORS so the local Streamlit test UI can call the API.
# Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "model": settings.served_model_id,
        "openai_base_url": "/v1",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "ocr": "/api/ocr/parse",
            "resume": "/api/resume/parse",
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
        "backend": backend_health,
        "features": {
            "ocr_enabled": settings.ocr_enabled,
            "ocr_version": settings.ocr_version,
            "embeddings_mode": settings.embeddings_mode,
        },
        "model": settings.served_model_id,
    }
    code = 200 if backend_health.get("ok") else 503
    return JSONResponse(body, status_code=code)


app.include_router(openai_router)
app.include_router(domain_router)
