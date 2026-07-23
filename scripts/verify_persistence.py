"""Verify the persistence layer end-to-end against SQLite.

This exercises the SAME SQLAlchemy models/repo used for MSSQL — only the
connection URL differs — so a green run here means the MSSQL wiring is correct
too (portable column types, auto-created tables, history endpoints).

Requires SQLAlchemy (pip install sqlalchemy). No MSSQL/ODBC needed.
Run:  ./.venv/Scripts/python.exe scripts/verify_persistence.py
"""
from __future__ import annotations

import os
import sys
import tempfile

_tmp = os.path.join(tempfile.gettempdir(), "slm_verify.sqlite")
if os.path.exists(_tmp):
    os.remove(_tmp)

# Configure BEFORE importing the app.
os.environ["LLM_BACKEND"] = "mock"
os.environ["EMBEDDINGS_MODE"] = "off"
os.environ["OCR_ENABLED"] = "false"
os.environ["GATEWAY_API_KEYS"] = ""
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}"
os.environ["WARMUP_ON_START"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SAMPLE = """Jane Doe
jane.doe@example.com | +1 555 123 4567
Senior ML Engineer, 7 years. Skills: Python, PyTorch, AWS, Kubernetes, NLP.
"""

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -> {extra}" if extra else ""))


# Using TestClient as a context manager runs lifespan -> db.init() creates tables.
with TestClient(app) as client:
    from app.persistence import db

    check("MSSQL/SQLite persistence enabled at startup", db.available())

    job = client.post("/api/jobs", json={
        "title": "Senior ML Engineer",
        "skills": ["Python", "PyTorch", "AWS", "Kubernetes"],
        "min_years_experience": 5, "enrich": True,
    }).json()
    cand = client.post("/api/resume/parse-text", json={"text": SAMPLE}).json()
    client.post("/api/match", json={"job": job, "candidate": cand, "justify": True})

    jobs = client.get("/api/history/jobs").json()["items"]
    cands = client.get("/api/history/candidates").json()["items"]
    matches = client.get("/api/history/matches").json()["items"]

    check("job persisted + listed", len(jobs) >= 1, f"{len(jobs)} rows")
    check("candidate persisted + listed", len(cands) >= 1, f"{len(cands)} rows")
    check("match persisted + listed", len(matches) >= 1,
          f"score={matches[0].get('overall_score') if matches else '-'}")
    check("request audit log populated",
          len(client.get("/api/history/matches").json()["items"]) >= 1)

print("\nPersistence verification:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
