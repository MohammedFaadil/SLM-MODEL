"""Deterministic mock backend.

No model, no GPU, no network — used to test the gateway wiring, domain routes,
and UI before Ollama/Qwen is installed. When JSON output is requested it returns
an empty object so the domain layer's heuristic fallbacks take over (those same
fallbacks are the production safety net for malformed model output).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict

from .base import LLMBackend


def _wants_json(payload: Dict[str, Any]) -> bool:
    rf = payload.get("response_format") or {}
    if isinstance(rf, dict) and rf.get("type") in {"json_object", "json_schema"}:
        return True
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str) and "json" in content.lower():
            return True
    return False


def _last_user_text(payload: Dict[str, Any]) -> str:
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


class MockBackend(LLMBackend):
    def __init__(self, model_id: str = "mock-slm") -> None:
        self._model_id = model_id

    def _content_for(self, payload: Dict[str, Any]) -> str:
        if _wants_json(payload):
            return "{}"
        preview = _last_user_text(payload)[:280]
        return (
            "[mock backend] No model is loaded. This is a deterministic stub so "
            "you can verify the gateway and UI end-to-end. Install Ollama + "
            f"Qwen3-8B for real output.\n\nEcho of your prompt: {preview}"
        )

    def _envelope(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        content = self._content_for(payload)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", self._model_id),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(content.split()),
                "total_tokens": len(content.split()),
            },
        }

    async def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._envelope(payload)

    async def chat_completion_stream(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        env = self._envelope(payload)
        content = env["choices"][0]["message"]["content"]
        base = {
            "id": env["id"],
            "object": "chat.completion.chunk",
            "created": env["created"],
            "model": env["model"],
        }
        # A couple of chunks, then the [DONE] sentinel.
        for piece in (content[: len(content) // 2], content[len(content) // 2 :]):
            chunk = {**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = "[mock backend] no model loaded."
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": payload.get("model", self._model_id),
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inp = payload.get("input")
        items = inp if isinstance(inp, list) else [inp]
        data = [
            {"object": "embedding", "index": i, "embedding": [0.0] * 8}
            for i, _ in enumerate(items)
        ]
        return {
            "object": "list",
            "data": data,
            "model": payload.get("model", self._model_id),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "backend": "mock"}
