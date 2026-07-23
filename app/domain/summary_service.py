"""Comprehensive candidate summary (on-demand 'AI Summary').

Combines deterministic experience math (total + per-skill years) with an
HR-facing narrative written by the model. The numbers are authoritative and
computed in code; the model narrates them.
"""
from __future__ import annotations

import datetime as _dt

from ..config import settings
from ..logging_conf import get_logger
from .experience import annotate_durations, skill_experience, total_years
from .llm_client import chat_text
from .prompts import CANDIDATE_SUMMARY_SYSTEM, candidate_summary_user
from .schemas import CandidateProfile, CandidateSummary

log = get_logger(__name__)


def _fallback_summary(profile: CandidateProfile, total, skill_exp) -> str:
    name = profile.contact.name or "The candidate"
    role = profile.current_title or profile.headline or "professional"
    yrs = f" with about {total:g} years of professional experience" if total else ""
    top = ", ".join(f"{s.skill} (~{s.years:g}y)" for s in skill_exp[:5] if s.evidenced)
    top_txt = f" Their strongest evidenced areas include {top}." if top else ""
    return f"{name} is a {role}{yrs}.{top_txt}"


async def candidate_summary(profile: CandidateProfile) -> CandidateSummary:
    annotate_durations(profile.experience)
    total = total_years(profile.experience)
    if total is None:
        total = profile.total_years_experience

    skill_exp = skill_experience(profile)
    today = _dt.date.today().isoformat()
    skill_years_payload = [
        {"skill": s.skill, "years": s.years, "evidenced": s.evidenced}
        for s in skill_exp[:30]
    ]

    summary_text = ""
    try:
        summary_text = await chat_text(
            CANDIDATE_SUMMARY_SYSTEM,
            candidate_summary_user(profile.model_dump(), skill_years_payload, total, today),
            temperature=0.35,
            max_tokens=1600,
            think=settings.domain_reasoning,
        )
        summary_text = summary_text.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Candidate summary generation failed (%s).", exc)

    if not summary_text or len(summary_text) < 40:
        summary_text = _fallback_summary(profile, total, skill_exp)

    return CandidateSummary(
        name=profile.contact.name,
        headline=profile.headline or profile.current_title,
        summary=summary_text,
        total_years_experience=total,
        skills=profile.skills,
        skill_experience=skill_exp,
        strengths=profile.strengths,
    )
