"""In-process smoke test — no model, no GPU, no network required.

Forces the mock backend and exercises the OpenAI-compatible endpoints + the OCR
endpoint via Starlette's TestClient.

Run:  ./.venv/Scripts/python.exe scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys

# Must be set BEFORE importing the app (settings are read at import time).
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDINGS_MODE", "off")
os.environ.setdefault("OCR_ENABLED", "false")
os.environ.setdefault("GATEWAY_API_KEYS", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {extra}" if extra else ""))


def main() -> int:
    client = TestClient(app)

    print("\n== meta ==")
    r = client.get("/health")
    check("GET /health", r.status_code == 200, r.json().get("version"))
    r = client.get("/v1/models")
    ids = [m["id"] for m in r.json().get("data", [])]
    check("GET /v1/models", r.status_code == 200 and len(ids) >= 1, str(ids))

    print("\n== chat completions ==")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    body = r.json()
    check("POST /v1/chat/completions", r.status_code == 200 and "choices" in body)
    check("  any model name routed to served id", body.get("model") == "slm-qwen3-8b")

    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as s:
        text = "".join(line for line in s.iter_lines())
    check("POST /v1/chat/completions (stream)", "data:" in text and "[DONE]" in text)

    print("\n== legacy completions ==")
    r = client.post("/v1/completions", json={"model": "x", "prompt": "hi"})
    check("POST /v1/completions", r.status_code == 200 and "choices" in r.json())
    with client.stream("POST", "/v1/completions",
                       json={"model": "x", "prompt": "hi", "stream": True}) as s:
        text = "".join(line for line in s.iter_lines())
    check("POST /v1/completions (stream)", "data:" in text and "[DONE]" in text)

    print("\n== embeddings (off -> degrades cleanly) ==")
    r = client.post("/v1/embeddings", json={"model": "x", "input": "hello"})
    check("POST /v1/embeddings", r.status_code in (200, 501), str(r.status_code))

    print("\n== ocr endpoint ==")
    r = client.post("/api/ocr/parse",
                    files={"file": ("resume.txt", b"Hello OCR gateway", "text/plain")})
    check("POST /api/ocr/parse", r.status_code == 200 and "text" in r.json(),
          r.json().get("status"))

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
