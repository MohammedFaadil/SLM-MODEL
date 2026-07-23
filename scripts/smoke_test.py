"""In-process smoke test — no model, no GPU, no network required.

Forces the mock backend + fuzzy matching, then exercises the OpenAI-compatible
endpoints and the recruiting domain flow end to end via Starlette's TestClient.

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

SAMPLE_RESUME = """
Jane Doe
Senior Machine Learning Engineer
jane.doe@example.com | +1 (555) 123-4567 | San Francisco, CA
linkedin.com/in/janedoe | github.com/janedoe

SUMMARY
ML engineer with 7 years building NLP and computer vision systems in production.

EXPERIENCE
Senior ML Engineer, Acme AI (2020 - Present)
- Built LLM-powered document extraction with PyTorch and Hugging Face transformers.
- Deployed models on AWS using Docker and Kubernetes.

Data Scientist, DataCorp (2017 - 2020)
- Developed scikit-learn and TensorFlow models; built pipelines with Airflow and Spark.

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, NLP, AWS, Docker, Kubernetes, SQL, FastAPI

EDUCATION
M.S. Computer Science, Stanford University, 2017
"""

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    mark = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f"  -> {extra}" if extra else ""))


def main() -> int:
    client = TestClient(app)

    print("\n== meta ==")
    r = client.get("/health")
    check("GET /health", r.status_code == 200, r.json().get("status"))
    r = client.get("/v1/models")
    ids = [m["id"] for m in r.json().get("data", [])]
    check("GET /v1/models", r.status_code == 200 and len(ids) >= 1, str(ids))

    print("\n== openai chat ==")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    body = r.json()
    check("POST /v1/chat/completions", r.status_code == 200 and "choices" in body,
          body.get("model"))
    check("  model remapped to served id", body.get("model") == "slm-qwen3-8b")

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
              "stream": True},
    ) as s:
        chunks = [line for line in s.iter_lines() if line]
    check("POST /v1/chat/completions (stream)",
          any("data:" in c for c in chunks) and any("[DONE]" in c for c in chunks),
          f"{len(chunks)} SSE lines")

    print("\n== embeddings (off -> should 501 cleanly) ==")
    r = client.post("/v1/embeddings", json={"model": "x", "input": "hello"})
    check("POST /v1/embeddings degrades gracefully", r.status_code in (200, 501),
          str(r.status_code))

    print("\n== domain: job creation ==")
    r = client.post(
        "/api/jobs",
        json={"title": "Senior ML Engineer",
              "skills": ["Python", "PyTorch", "AWS", "Kubernetes"],
              "min_years_experience": 5, "seniority": "Senior", "enrich": True},
    )
    job = r.json()
    check("POST /api/jobs", r.status_code == 200 and job.get("title") == "Senior ML Engineer",
          f"required={job.get('required_skills')}")

    print("\n== domain: resume parse (text) ==")
    r = client.post("/api/resume/parse-text", json={"text": SAMPLE_RESUME})
    cand = r.json()
    check("POST /api/resume/parse-text", r.status_code == 200)
    check("  extracted email", cand.get("contact", {}).get("email") == "jane.doe@example.com",
          cand.get("contact", {}).get("email"))
    check("  extracted skills (heuristic lexicon)", len(cand.get("skills", [])) >= 4,
          str(cand.get("skills")))

    print("\n== domain: candidate summary ==")
    r = client.post("/api/candidate/summary", json=cand)
    summ = r.json()
    check("POST /api/candidate/summary", r.status_code == 200 and "summary" in summ,
          f"skills={len(summ.get('skill_experience', []))}")
    check("  skill_experience present", isinstance(summ.get("skill_experience"), list))

    print("\n== domain: match ==")
    r = client.post("/api/match", json={"job": job, "candidate": cand, "justify": True})
    m = r.json()
    check("POST /api/match", r.status_code == 200 and "overall_score" in m,
          f"score={m.get('overall_score')} verdict={m.get('verdict')}")
    check("  matched_skills present", len(m.get("matched_skills", [])) >= 1)
    check("  justification present", bool(m.get("justification")))

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
