"""Thin helper for calling our own backend for structured / free-form output.

Handles the two things that make small-model output reliable:
  * Qwen3 "thinking" control (off by default -> faster, cleaner JSON on CPU).
  * Lenient JSON parsing (strips <think> blocks and ```json fences, repairs
    trailing commas, extracts the outermost object) so a slightly-off model
    response still yields usable data instead of crashing.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..backends.factory import get_backend
from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _apply_thinking(messages: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    if settings.enable_thinking:
        return
    # vLLM honours chat_template_kwargs; Ollama ignores it harmlessly.
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    # Ollama/Qwen3 honour the /no_think directive in the prompt.
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0]['content']} /no_think"
    else:
        messages.insert(0, {"role": "system", "content": "/no_think"})


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object from a model response."""
    if not text:
        return {}
    text = strip_reasoning(text)

    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Direct parse first.
    for candidate in (text, _outermost_object(text)):
        if not candidate:
            continue
        for attempt in (candidate, _repair(candidate)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


def _outermost_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def _repair(text: str) -> str:
    # Remove trailing commas before } or ].
    return re.sub(r",\s*([}\]])", r"\1", text)


async def chat_text(
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    _apply_thinking(messages, payload)
    resp = await get_backend().chat_completion(payload)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return strip_reasoning(content)


async def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1600,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    _apply_thinking(messages, payload)
    resp = await get_backend().chat_completion(payload)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json(content)
