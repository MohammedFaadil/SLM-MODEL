"""Backend selection + lifecycle (single shared instance per process)."""
from __future__ import annotations

from typing import Optional

from ..config import Settings, settings
from ..logging_conf import get_logger
from .base import LLMBackend
from .mock import MockBackend
from .openai_upstream import OpenAIUpstreamBackend

log = get_logger(__name__)

_backend: Optional[LLMBackend] = None


def build_backend(cfg: Settings) -> LLMBackend:
    if cfg.llm_backend == "mock":
        log.info("LLM backend: mock (no model)")
        return MockBackend(model_id=cfg.served_model_id)
    log.info(
        "LLM backend: openai_upstream -> %s (model=%s)",
        cfg.upstream_base_url,
        cfg.model_name,
    )
    return OpenAIUpstreamBackend(cfg)


def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = build_backend(settings)
    return _backend


async def close_backend() -> None:
    global _backend
    if _backend is not None:
        await _backend.aclose()
        _backend = None
