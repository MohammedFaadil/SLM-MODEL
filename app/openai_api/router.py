"""OpenAI-compatible endpoints — the surface the product talks to.

Point the product's OpenAI `base_url` at `<gateway>/v1`. The gateway is a faithful
transparent proxy: the product's prompts and params (streaming, JSON mode, tools,
temperature, seed, chat_template_kwargs, ...) are forwarded to the model untouched.
The gateway adds NO prompts and never overrides the product's sampling — accuracy is
entirely the product's choice.

Implemented:
  POST /v1/chat/completions   (streaming + non-streaming, tools, JSON mode, ...)
  POST /v1/completions        (streaming + non-streaming)
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

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _normalize_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route every request to the configured model, touching ONLY the model field.

    With FORCE_MODEL=true, whatever the product sends (e.g. "gpt-4o") is replaced by
    MODEL_NAME. All other fields (sampling, tools, response_format, ...) pass through.
    """
    payload = dict(payload)
    if settings.force_model or not payload.get("model"):
        payload["model"] = settings.model_name
    return payload


def _sse_error(exc: BackendError) -> bytes:
    """A terminal SSE error event, preserving a genuine OpenAI-shaped upstream body."""
    body = (
        exc.body
        if isinstance(exc.body, dict) and "error" in exc.body
        else {"error": {"message": exc.message, "type": exc.err_type, "param": None, "code": None}}
    )
    return f"data: {json.dumps(body)}\n\n".encode() + b"data: [DONE]\n\n"


async def _sse_response(gen: AsyncIterator[bytes]) -> StreamingResponse:
    """Stream an upstream SSE byte stream to the client.

    The generator is PRIMED here so a pre-token upstream error (bad params, unknown
    model, 401/429, unreachable) raises BackendError before the response starts —
    the app handler then returns the real HTTP status + OpenAI error body, instead of
    a misleading 200. Genuine mid-stream failures become a clean terminal SSE error.
    """
    it = gen.__aiter__()
    try:
        first = await it.__anext__()  # BackendError here -> app exception handler
    except StopAsyncIteration:
        first = None

    async def body() -> AsyncIterator[bytes]:
        if first is None:  # empty upstream stream -> well-formed terminator
            yield b"data: [DONE]\n\n"
            return
        yield first
        try:
            async for chunk in it:
                yield chunk
        except BackendError as exc:
            yield _sse_error(exc)

    return StreamingResponse(body(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request) -> Any:
    payload = _normalize_model(await request.json())
    backend = get_backend()
    if payload.get("stream"):
        return await _sse_response(backend.chat_completion_stream(payload))
    data = await backend.chat_completion(payload)
    if isinstance(data, dict):
        data["model"] = settings.served_model_id  # advertise our public id
    return JSONResponse(data)


@router.post("/completions", dependencies=[Depends(require_api_key)])
async def completions(request: Request) -> Any:
    payload = _normalize_model(await request.json())
    backend = get_backend()
    if payload.get("stream"):
        return await _sse_response(backend.completion_stream(payload))
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
        return JSONResponse(await backend.embeddings(_normalize_model(raw)))

    try:
        from ..embeddings.embedder import get_embedder

        embedder = get_embedder()
        data = embedder.openai_response(
            raw.get("input"),
            model=settings.embedding_model,
            encoding_format=raw.get("encoding_format") or "float",
            dimensions=raw.get("dimensions"),
        )
        return JSONResponse(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("Local embeddings unavailable (%s); trying upstream.", exc)
        try:
            return JSONResponse(await backend.embeddings(_normalize_model(raw)))
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
    if settings.model_name != settings.served_model_id:
        cards.append(ModelCard(id=settings.model_name))
    return ModelList(data=cards)
