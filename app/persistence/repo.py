"""Best-effort persistence helpers.

Every function is a no-op when the DB is disabled and swallows its own errors, so
a database hiccup can never break an API response. Call the write helpers via
`fastapi.concurrency.run_in_threadpool` since the driver is synchronous.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..logging_conf import get_logger
from . import db

log = get_logger(__name__)


def _dump(model: Any) -> str:
    try:
        if hasattr(model, "model_dump"):
            return json.dumps(model.model_dump(), ensure_ascii=False, default=str)
        return json.dumps(model, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def save_request_log(**kw: Any) -> None:
    if not db.available():
        return
    try:
        from .models import RequestLog

        with db.session() as s:
            s.add(RequestLog(**kw))
    except Exception as exc:  # noqa: BLE001
        log.debug("request-log write skipped: %s", exc)


def save_job(spec: Any) -> Optional[int]:
    if not db.available():
        return None
    try:
        from .models import JobRecord

        with db.session() as s:
            row = JobRecord(
                title=getattr(spec, "title", "") or "",
                seniority=getattr(spec, "seniority", "") or "",
                min_years_experience=float(getattr(spec, "min_years_experience", 0) or 0),
                spec_json=_dump(spec),
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as exc:  # noqa: BLE001
        log.debug("job write skipped: %s", exc)
        return None


def save_candidate(profile: Any) -> Optional[int]:
    if not db.available():
        return None
    try:
        from .models import CandidateRecord

        contact = getattr(profile, "contact", None)
        with db.session() as s:
            row = CandidateRecord(
                name=(getattr(contact, "name", "") or "") if contact else "",
                email=(getattr(contact, "email", "") or "") if contact else "",
                total_years_experience=float(getattr(profile, "total_years_experience", 0) or 0),
                profile_json=_dump(profile),
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as exc:  # noqa: BLE001
        log.debug("candidate write skipped: %s", exc)
        return None


def save_match(job: Any, candidate: Any, result: Any) -> Optional[int]:
    if not db.available():
        return None
    try:
        from .models import MatchRecord

        contact = getattr(candidate, "contact", None)
        with db.session() as s:
            row = MatchRecord(
                job_title=getattr(job, "title", "") or "",
                candidate_name=(getattr(contact, "name", "") or "") if contact else "",
                overall_score=float(getattr(result, "overall_score", 0) or 0),
                verdict=getattr(result, "verdict", "") or "",
                result_json=_dump(result),
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as exc:  # noqa: BLE001
        log.debug("match write skipped: %s", exc)
        return None


def _list(model_name: str, limit: int) -> List[Dict[str, Any]]:
    if not db.available():
        return []
    try:
        from sqlalchemy import select

        from . import models as m

        model = getattr(m, model_name)
        with db.session() as s:
            rows = s.execute(
                select(model).order_by(model.created_at.desc()).limit(limit)
            ).scalars().all()
            out = []
            for r in rows:
                d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
                if d.get("created_at") is not None:
                    d["created_at"] = d["created_at"].isoformat()
                out.append(d)
            return out
    except Exception as exc:  # noqa: BLE001
        log.debug("list %s skipped: %s", model_name, exc)
        return []


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    return _list("JobRecord", limit)


def list_candidates(limit: int = 50) -> List[Dict[str, Any]]:
    return _list("CandidateRecord", limit)


def list_matches(limit: int = 50) -> List[Dict[str, Any]]:
    return _list("MatchRecord", limit)
