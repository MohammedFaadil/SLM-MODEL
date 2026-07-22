"""OpenAI-compatible endpoints.

This is the surface your existing platform talks to. Point its OpenAI
`base_url` at `<gateway>/v1` and keep everything else the same.

Implemented:
  POST /v1/chat/completions   (streaming + non-streaming, tools, JSON mode)
  POST /v1/completions        (legacy)
  POST /v1/embeddings         (local sentence-transformers or upstream)
  GET  /v1/models
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..backends.base import BackendError
from ..backends.factory import get_backend
from ..config import settings
from ..logging_conf import get_logger
from ..security import require_api_key
from .schemas import ModelCard, ModelList

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


def _normalize_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route every request to the configured model.

    With FORCE_MODEL=true, whatever the platform sends (e.g. "gpt-4o") is
    replaced by MODEL_NAME so no platform-side change is needed.
    """
    payload = dict(payload)
    if settings.force_model or not payload.get("model"):
        payload["model"] = settings.model_name
    return payload


async def _guarded_stream(gen: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Wrap the upstream byte stream so a mid-stream backend error is delivered
    as a clean SSE error event instead of a broken connection."""
    try:
        async for chunk in gen:
            yield chunk
    except BackendError as exc:
        err = {"error": {"message": exc.message, "type": exc.err_type, "code": exc.status_code}}
        yield f"data: {json.dumps(err)}\n\n".encode()
        yield b"data: [DONE]\n\n"


@router.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request) -> Any:
    raw = await request.json()
    payload = _normalize_model(raw)
    backend = get_backend()

    if payload.get("stream"):
        gen = backend.chat_completion_stream(payload)
        return StreamingResponse(
            _guarded_stream(gen),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    data = await backend.chat_completion(payload)
    # Advertise our public model id rather than the internal engine tag.
    if isinstance(data, dict):
        data["model"] = settings.served_model_id
    return JSONResponse(data)


@router.post("/completions", dependencies=[Depends(require_api_key)])
async def completions(request: Request) -> Any:
    raw = await request.json()
    payload = _normalize_model(raw)
    backend = get_backend()

    if payload.get("stream"):
        # Reuse chat streaming shape is not valid here; most modern platforms use
        # chat. Fall back to non-streaming for the legacy endpoint.
        payload["stream"] = False
    data = await backend.completion(payload)
    if isinstance(data, dict):
        data["model"] = settings.served_model_id
    return JSONResponse(data)


@router.post("/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(request: Request) -> Any:
    raw = await request.json()
    backend = get_backend()

    # Prefer local sentence-transformers unless explicitly set to upstream.
    if settings.embeddings_mode == "upstream":
        payload = _normalize_model(raw)
        data = await backend.embeddings(payload)
        return JSONResponse(data)

    try:
        from ..embeddings.embedder import get_embedder

        embedder = get_embedder()
        data = embedder.openai_response(raw.get("input"), model=settings.embedding_model)
        return JSONResponse(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("Local embeddings unavailable (%s); trying upstream.", exc)
        try:
            payload = _normalize_model(raw)
            return JSONResponse(await backend.embeddings(payload))
        except Exception as exc2:  # noqa: BLE001
            raise BackendError(
                f"Embeddings unavailable. Install requirements-embeddings.txt "
                f"or set EMBEDDINGS_MODE=upstream with an embedding model. ({exc2})",
                status_code=501,
                err_type="embeddings_unavailable",
            ) from exc2


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models() -> ModelList:
    cards = [ModelCard(id=settings.served_model_id)]
    # Also advertise the raw engine tag for clients that request it directly.
    if settings.model_name != settings.served_model_id:
        cards.append(ModelCard(id=settings.model_name))
    return ModelList(data=cards)
