"""Create/enrich a JobSpec from a title + seed skills + experience + free-text prompt.

Two stages, so the model never has to stuff a long Markdown description inside a
JSON string (which breaks small-model JSON):
  1. structured fields  -> strict JSON  (chat_json)
  2. rich description    -> Markdown text (chat_text, reasoning-friendly)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from ..config import settings
from ..logging_conf import get_logger
from .llm_client import chat_json, chat_text
from .prompts import (
    JOB_DESCRIPTION_SYSTEM,
    JOB_FIELDS_SYSTEM,
    job_description_user,
    job_fields_user,
)
from .schemas import JobCreateRequest, JobSpec
from .skills import normalize_skills

log = get_logger(__name__)


def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _to_float(v: Any):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _payload(req: JobCreateRequest) -> Dict[str, Any]:
    return {
        "title": req.title,
        "skills": req.skills,
        "prompt": req.prompt,               # <-- the user's detailed requirements
        "min_years_experience": req.min_years_experience,
        "seniority": req.seniority,
        "location": req.location,
        "employment_type": req.employment_type,
        "notes": req.notes,
    }


def _compose_description_fallback(spec: JobSpec, req: JobCreateRequest) -> str:
    """Deterministic Markdown JD if the description call is unavailable/empty."""
    lines = [f"# Job Description: {spec.title}", ""]
    if spec.seniority:
        lines += ["### Experience Level", f"{spec.seniority}", ""]
    if req.prompt or req.notes:
        lines += ["## Job Summary", (req.prompt or req.notes).strip(), ""]
    if spec.responsibilities:
        lines += ["## Key Responsibilities", *[f"* {r}" for r in spec.responsibilities], ""]
    if spec.required_skills:
        lines += ["## Required Skills", *[f"* {s}" for s in spec.required_skills], ""]
    if spec.preferred_skills:
        lines += ["## Preferred Skills", *[f"* {s}" for s in spec.preferred_skills], ""]
    if spec.qualifications:
        lines += ["## Qualifications", *[f"* {q}" for q in spec.qualifications], ""]
    return "\n".join(lines).strip()


def _basic_spec(req: JobCreateRequest) -> JobSpec:
    """Used when enrichment is off or the model is offline."""
    spec = JobSpec(
        title=req.title,
        seniority=req.seniority,
        min_years_experience=req.min_years_experience,
        required_skills=normalize_skills(req.skills),
        location=req.location,
        employment_type=req.employment_type,
    )
    spec.description = _compose_description_fallback(spec, req)
    return spec


async def create_job(req: JobCreateRequest) -> JobSpec:
    if not req.enrich:
        return _basic_spec(req)

    payload = _payload(req)

    # Structured fields (JSON) and the long description (text) both derive from the
    # same inputs, so run them CONCURRENTLY to roughly halve job-creation latency.
    fields_res, desc_res = await asyncio.gather(
        chat_json(JOB_FIELDS_SYSTEM, job_fields_user(payload), max_tokens=1200),
        chat_text(
            JOB_DESCRIPTION_SYSTEM,
            job_description_user(payload, {}),
            temperature=0.5,
            max_tokens=1800,
            think=settings.domain_reasoning,
        ),
        return_exceptions=True,
    )
    fields: Dict[str, Any] = fields_res if isinstance(fields_res, dict) else {}
    if isinstance(fields_res, Exception):
        log.warning("Job fields generation failed (%s).", fields_res)
    description = desc_res if isinstance(desc_res, str) else ""
    if isinstance(desc_res, Exception):
        log.warning("Job description generation failed (%s).", desc_res)

    if not fields:
        # No structured data at all -> still give a useful basic spec.
        spec = _basic_spec(req)
    else:
        required = normalize_skills(_as_list(fields.get("required_skills")) or req.skills)
        for s in normalize_skills(req.skills):  # user's seed skills are always required
            if s not in required:
                required.append(s)
        spec = JobSpec(
            title=req.title or "Untitled role",
            seniority=req.seniority or fields.get("seniority"),
            min_years_experience=(
                req.min_years_experience
                if req.min_years_experience is not None
                else _to_float(fields.get("min_years_experience"))
            ),
            required_skills=required,
            preferred_skills=normalize_skills(_as_list(fields.get("preferred_skills"))),
            responsibilities=_as_list(fields.get("responsibilities")),
            qualifications=_as_list(fields.get("qualifications")),
            location=req.location or fields.get("location"),
            employment_type=req.employment_type or fields.get("employment_type"),
        )

    spec.description = description.strip() or _compose_description_fallback(spec, req)
    return spec
