"""Resume text -> structured CandidateProfile + recruiter summary.

Primary path is a single SLM call (extraction + summary together, to save CPU
time). Regex/lexicon heuristics fill any gaps and act as a full fallback when no
model is available, so the endpoint always returns something usable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..config import settings
from ..logging_conf import get_logger
from .llm_client import chat_json, chat_text
from .prompts import (
    RESUME_PARSE_SYSTEM,
    RESUME_SUMMARY_SYSTEM,
    resume_parse_user,
    resume_summary_user,
)
from .schemas import (
    CandidateProfile,
    ContactInfo,
    EducationItem,
    ExperienceItem,
)
from .skills import extract_skills_heuristic, normalize_skills

log = get_logger(__name__)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(\+?\d[\d\s().\-]{7,}\d)")
_LINK = re.compile(r"(https?://[^\s]+|(?:www\.)?(?:linkedin\.com|github\.com)/[^\s,]+)", re.I)
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def _as_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _profile_from_llm(data: Dict[str, Any]) -> CandidateProfile:
    c = data.get("contact") or {}
    contact = ContactInfo(
        name=(c.get("name") or None),
        email=(c.get("email") or None),
        phone=(c.get("phone") or None),
        location=(c.get("location") or None),
        links=_as_list(c.get("links")),
    )

    experience = []
    for e in data.get("experience") or []:
        if not isinstance(e, dict):
            continue
        experience.append(
            ExperienceItem(
                title=e.get("title") or None,
                company=e.get("company") or None,
                start=str(e.get("start")) if e.get("start") else None,
                end=str(e.get("end")) if e.get("end") else None,
                highlights=_as_list(e.get("highlights")),
                skills=normalize_skills(_as_list(e.get("skills"))),
            )
        )

    education = []
    for ed in data.get("education") or []:
        if not isinstance(ed, dict):
            continue
        education.append(
            EducationItem(
                degree=ed.get("degree") or None,
                field=ed.get("field") or None,
                institution=ed.get("institution") or None,
                year=str(ed.get("year")) if ed.get("year") else None,
            )
        )

    years = data.get("total_years_experience")
    try:
        years = float(years) if years is not None else None
    except (TypeError, ValueError):
        years = None

    return CandidateProfile(
        contact=contact,
        headline=data.get("headline") or None,
        current_title=data.get("current_title") or None,
        total_years_experience=years,
        skills=normalize_skills(_as_list(data.get("skills"))),
        experience=experience,
        education=education,
        certifications=_as_list(data.get("certifications")),
        strengths=_as_list(data.get("strengths")),
        summary=(data.get("summary") or "").strip(),
    )


def _estimate_years(text: str) -> Optional[float]:
    years = [int(y) for y in _YEAR.findall(text)]
    if len(years) < 2:
        return None
    span = max(years) - min(years)
    # Guard against education years inflating the span.
    return float(span) if 0 < span <= 30 else None


def _fill_heuristics(profile: CandidateProfile, text: str) -> None:
    if not profile.contact.email:
        m = _EMAIL.search(text)
        if m:
            profile.contact.email = m.group(0)
    if not profile.contact.phone:
        m = _PHONE.search(text)
        if m:
            profile.contact.phone = m.group(0).strip()
    if not profile.contact.links:
        profile.contact.links = list(dict.fromkeys(_LINK.findall(text)))[:5]
    if not profile.contact.name:
        for line in text.splitlines():
            s = line.strip()
            if 2 <= len(s.split()) <= 4 and s.replace(" ", "").isalpha() and s[:1].isupper():
                profile.contact.name = s
                break
    if not profile.skills:
        profile.skills = extract_skills_heuristic(text)
    if profile.total_years_experience is None:
        profile.total_years_experience = _estimate_years(text)


def _fallback_summary(profile: CandidateProfile) -> str:
    name = profile.contact.name or "The candidate"
    role = profile.current_title or profile.headline or "professional"
    yrs = profile.total_years_experience
    yrs_txt = f" with ~{yrs:g} years of experience" if yrs else ""
    top = ", ".join(profile.skills[:6]) or "a range of relevant skills"
    return f"{name} is a {role}{yrs_txt}. Core skills include {top}."


async def parse_resume(text: str) -> CandidateProfile:
    data: Dict[str, Any] = {}
    try:
        # Generous budget: full structured extraction (roles + per-role skills +
        # highlights) must not be truncated, or experience is lost.
        data = await chat_json(RESUME_PARSE_SYSTEM, resume_parse_user(text), max_tokens=4096)
    except Exception as exc:  # noqa: BLE001
        log.warning("Resume LLM parse failed (%s); using heuristics.", exc)

    if not data:
        log.warning(
            "Resume extraction returned no JSON (text=%d chars). Falling back to "
            "heuristics — experience/roles will be limited. Consider a larger model.",
            len(text),
        )

    profile = _profile_from_llm(data) if data else CandidateProfile()
    _fill_heuristics(profile, text)

    # Deterministic experience math from the parsed date ranges (more accurate
    # and reproducible than the model's estimate).
    from .experience import annotate_durations, total_years

    annotate_durations(profile.experience)
    det_total = total_years(profile.experience)
    if det_total is not None:
        profile.total_years_experience = det_total

    log.info(
        "Parsed resume: %d roles, %d skills, total_years=%s (deterministic=%s).",
        len(profile.experience), len(profile.skills),
        profile.total_years_experience, det_total,
    )

    # Always generate a detailed recruiter assessment in a dedicated text call
    # (richer + more reliable than squeezing it into the extraction JSON).
    try:
        summary = await chat_text(
            RESUME_SUMMARY_SYSTEM,
            resume_summary_user(profile.model_dump(), text),
            temperature=0.35,
            max_tokens=1200,
            think=settings.domain_reasoning,
        )
        summary = summary.strip()
        # Guard against the model refusing / returning a stub.
        if len(summary) >= 40:
            profile.summary = summary
    except Exception as exc:  # noqa: BLE001
        log.warning("Resume summary generation failed (%s).", exc)

    if not profile.summary or len(profile.summary) < 20:
        profile.summary = _fallback_summary(profile)

    return profile
