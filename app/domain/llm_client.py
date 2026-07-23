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


def _apply_thinking(
    messages: List[Dict[str, Any]], payload: Dict[str, Any], think: Optional[bool]
) -> None:
    """Enable/disable Qwen3 reasoning for this call.

    think=None -> fall back to the global ENABLE_THINKING default.
    """
    if think is None:
        think = settings.enable_thinking
    if think:
        # vLLM honours chat_template_kwargs; Ollama/Qwen3 think by default.
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        return
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    # Ollama/Qwen3 honour the /no_think directive in the prompt.
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0]['content']} /no_think"
    else:
        messages.insert(0, {"role": "system", "content": "/no_think"})


def _apply_sampling(payload: Dict[str, Any], temperature: float) -> None:
    """Force reproducible decoding when DETERMINISTIC is on.

    Greedy (temperature 0) + fixed seed => same input, same output every time.
    """
    if settings.deterministic:
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
    else:
        payload["temperature"] = temperature
    payload["seed"] = settings.llm_seed


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

    # Direct parse first, then repaired, then truncation-recovered.
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

    # Last resort: the model output was cut off mid-JSON (token limit). Balance
    # the open strings/brackets and try again — recovers most of the object.
    salvaged = _repair(_close_truncated(text))
    if salvaged:
        try:
            obj = json.loads(salvaged)
            if isinstance(obj, dict):
                log.warning("Recovered a truncated JSON response (%d chars).", len(text))
                return obj
        except Exception:
            pass
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


def _close_truncated(text: str) -> str:
    """Repair JSON cut off by the token limit: close any open string and balance
    the remaining brackets so the largest valid prefix parses."""
    start = text.find("{")
    if start == -1:
        return ""
    s = text[start:]
    stack, in_str, esc = [], False, False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            if in_str:
                esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    out = s + ('"' if in_str else "")
    stripped = out.rstrip()
    if stripped.endswith(","):
        out = stripped[:-1]
    elif stripped.endswith(":"):
        out = stripped + " null"
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


async def chat_text(
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
    think: Optional[bool] = None,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    _apply_sampling(payload, temperature)
    _apply_thinking(messages, payload, think)
    resp = await get_backend().chat_completion(payload)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return strip_reasoning(content)


async def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1600,
    think: Optional[bool] = None,
) -> Dict[str, Any]:
    # Default JSON calls to NON-thinking: cleaner, guaranteed-parseable output.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    _apply_sampling(payload, temperature)
    _apply_thinking(messages, payload, False if think is None else think)
    resp = await get_backend().chat_completion(payload)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json(content)
