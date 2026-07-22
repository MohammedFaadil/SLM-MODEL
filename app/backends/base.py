"""Backend abstraction.

A backend turns an OpenAI-shaped request dict into an OpenAI-shaped response.
The gateway does model-name remapping and auth *before* calling the backend, so
backends receive an already-normalized payload.
"""
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, Optional


class BackendError(Exception):
    """Raised when the upstream engine returns an error or is unreachable.

    Carries an HTTP status so the gateway can surface an OpenAI-shaped error
    with the right code back to the calling platform.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        body: Optional[Any] = None,
        err_type: str = "upstream_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.err_type = err_type


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    async def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming chat completion -> OpenAI response dict."""

    @abc.abstractmethod
    def chat_completion_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Streaming chat completion -> raw SSE byte chunks (``data: {...}\\n\\n``)."""

    @abc.abstractmethod
    async def completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming legacy completion."""

    @abc.abstractmethod
    async def embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Embeddings via the upstream engine (may be unsupported)."""

    @abc.abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Return {'ok': bool, ...} describing backend reachability."""

    async def aclose(self) -> None:  # pragma: no cover - trivial default
        return None
