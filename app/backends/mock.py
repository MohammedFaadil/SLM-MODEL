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
            text = _last_user_text(payload)
            if "JOB INPUT" in text:
                import json
                mock_jd = {
                    "title": "Software Developer",
                    "seniority": "Junior",
                    "min_years_experience": 1.0,
                    "required_skills": ["C#", ".NET", "SQL Server", "HTML", "CSS", "JavaScript", "Git & GitHub", "REST APIs", "Object-Oriented Programming (OOP)", "Problem-solving and debugging"],
                    "preferred_skills": ["Blazor", "Python", "Docker", "Azure or AWS", "CI/CD pipelines", "Agile/Scrum methodology"],
                    "responsibilities": [
                        "Develop, test, and maintain web applications.",
                        "Write clean, efficient, and reusable code.",
                        "Collaborate with designers, developers, and product managers.",
                        "Debug and resolve software defects.",
                        "Optimize application performance and scalability.",
                        "Participate in code reviews and technical discussions.",
                        "Integrate third-party APIs and databases.",
                        "Prepare technical documentation when required."
                    ],
                    "qualifications": [
                        "Bachelor's degree in Computer Science, Information Technology, or a related field.",
                        "Strong understanding of software development principles.",
                        "Good communication and teamwork skills."
                    ],
                    "location": "Chennai, Tamil Nadu (Hybrid)",
                    "employment_type": "Full-Time",
                    "description": "# Job Description: Software Developer\n\n### Job Title\nSoftware Developer\n\n### Location\nChennai, Tamil Nadu (Hybrid)\n\n### Employment Type\nFull-Time\n\n### Experience\n1–3 Years\n\n## Job Summary\nWe are looking for a passionate and motivated Software Developer to design, develop, test, and maintain high-quality software applications. The ideal candidate should have strong problem-solving skills, a solid understanding of programming concepts, and the ability to work collaboratively in an agile development environment.\n\n## Key Responsibilities\n* Develop, test, and maintain web applications.\n* Write clean, efficient, and reusable code.\n* Collaborate with designers, developers, and product managers.\n* Debug and resolve software defects.\n* Optimize application performance and scalability.\n* Participate in code reviews and technical discussions.\n* Integrate third-party APIs and databases.\n* Prepare technical documentation when required.\n\n## Required Skills\n* C#\n* .NET / ASP.NET Core\n* SQL Server\n* HTML, CSS, JavaScript\n* Git & GitHub\n* REST APIs\n* Object-Oriented Programming (OOP)\n* Problem-solving and debugging\n\n## Preferred Skills\n* Blazor\n* Python\n* Docker\n* Azure or AWS\n* CI/CD pipelines\n* Agile/Scrum methodology\n\n## Qualifications\n* Bachelor's degree in Computer Science, Information Technology, or a related field.\n* Strong understanding of software development principles.\n* Good communication and teamwork skills.\n\n## Benefits\n* Competitive salary\n* Health insurance\n* Paid leave\n* Flexible working hours\n* Learning and certification support\n* Career growth opportunities"
                }
                return json.dumps(mock_jd)
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
