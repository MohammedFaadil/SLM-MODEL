"""Transparent proxy to any OpenAI-compatible inference server.

Locally this points at Ollama (`http://localhost:11434/v1`); in the cloud it
points at vLLM (`http://vllm:8000/v1`). Same code either way — the only thing
that changes is UPSTREAM_BASE_URL + MODEL_NAME.

Streaming responses are passed through byte-for-byte, so tool calls, JSON mode,
and SSE framing behave exactly as the upstream engine emits them.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx

from ..config import Settings
from ..logging_conf import get_logger
from .base import BackendError, LLMBackend

log = get_logger(__name__)


class OpenAIUpstreamBackend(LLMBackend):
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.upstream_base_url
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {settings.upstream_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.request_timeout, connect=15.0),
        )

    # -- helpers ----------------------------------------------------------- #
    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise BackendError(
                f"Cannot reach model backend at {self._base_url}. "
                f"Is Ollama/vLLM running? ({exc})",
                status_code=503,
                err_type="backend_unreachable",
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(
                "Model backend timed out. CPU inference can be slow — raise "
                "REQUEST_TIMEOUT or use a smaller/faster model.",
                status_code=504,
                err_type="backend_timeout",
            ) from exc

        if resp.status_code >= 400:
            body: Any
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise BackendError(
                f"Backend returned {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )
        return resp.json()

    # -- interface --------------------------------------------------------- #
    async def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post_json("/chat/completions", payload)

    async def chat_completion_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        payload = {**payload, "stream": True}
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    raise BackendError(
                        f"Backend returned {resp.status_code} on stream",
                        status_code=resp.status_code,
                        body=raw.decode("utf-8", "replace"),
                    )
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.ConnectError as exc:
            raise BackendError(
                f"Cannot reach model backend at {self._base_url} ({exc})",
                status_code=503,
                err_type="backend_unreachable",
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(
                "Model backend timed out while streaming.",
                status_code=504,
                err_type="backend_timeout",
            ) from exc

    async def completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post_json("/completions", payload)

    async def embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post_json("/embeddings", payload)

    async def health(self) -> Dict[str, Any]:
        try:
            resp = await self._client.get("/models")
            ok = resp.status_code < 400
            return {
                "ok": ok,
                "backend": "openai_upstream",
                "upstream": self._base_url,
                "status_code": resp.status_code,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "backend": "openai_upstream",
                "upstream": self._base_url,
                "error": str(exc),
            }

    async def aclose(self) -> None:
        await self._client.aclose()
