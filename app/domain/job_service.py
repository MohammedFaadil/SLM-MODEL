"""Create/enrich a JobSpec from a title + seed skills + experience level."""
from __future__ import annotations

from typing import Any, Dict, List

from ..logging_conf import get_logger
from .llm_client import chat_json
from .prompts import JOB_CREATE_SYSTEM, job_create_user
from .schemas import JobCreateRequest, JobSpec
from .skills import normalize_skills

log = get_logger(__name__)


def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _basic_spec(req: JobCreateRequest) -> JobSpec:
    """Deterministic spec used when enrichment is off or the model is offline."""
    return JobSpec(
        title=req.title,
        seniority=req.seniority,
        min_years_experience=req.min_years_experience,
        required_skills=normalize_skills(req.skills),
        preferred_skills=[],
        responsibilities=[],
        qualifications=[],
        location=req.location,
        employment_type=req.employment_type,
        description=(req.notes or f"{req.title} role.").strip(),
    )


async def create_job(req: JobCreateRequest) -> JobSpec:
    if not req.enrich:
        return _basic_spec(req)

    payload = {
        "title": req.title,
        "skills": req.skills,
        "min_years_experience": req.min_years_experience,
        "seniority": req.seniority,
        "location": req.location,
        "employment_type": req.employment_type,
        "notes": req.notes,
    }

    data: Dict[str, Any] = {}
    try:
        data = await chat_json(JOB_CREATE_SYSTEM, job_create_user(payload), max_tokens=1600)
    except Exception as exc:  # noqa: BLE001
        log.warning("Job enrichment failed (%s); returning basic spec.", exc)

    if not data:
        return _basic_spec(req)

    # Merge: the model enriches, but the user's explicit inputs win.
    required = normalize_skills(_as_list(data.get("required_skills")) or req.skills)
    # Guarantee every seed skill the user asked for is present as required.
    for s in normalize_skills(req.skills):
        if s not in required:
            required.append(s)

    return JobSpec(
        title=req.title or data.get("title") or "Untitled role",
        seniority=req.seniority or data.get("seniority"),
        min_years_experience=(
            req.min_years_experience
            if req.min_years_experience is not None
            else _to_float(data.get("min_years_experience"))
        ),
        required_skills=required,
        preferred_skills=normalize_skills(_as_list(data.get("preferred_skills"))),
        responsibilities=_as_list(data.get("responsibilities")),
        qualifications=_as_list(data.get("qualifications")),
        location=req.location or data.get("location"),
        employment_type=req.employment_type or data.get("employment_type"),
        description=(data.get("description") or req.notes or f"{req.title} role.").strip(),
    )


def _to_float(v: Any):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
