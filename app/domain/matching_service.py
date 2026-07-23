"""Job <-> candidate matching.

Two-stage by design:
  1. Deterministic, explainable scoring (skill coverage + experience) computed in
     code so results are stable and reproducible.
  2. The SLM writes the human justification, grounded ONLY in the numbers from
     stage 1 — so the narrative can never contradict the score.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import settings
from ..logging_conf import get_logger
from .experience import skill_experience
from .llm_client import chat_text
from .prompts import MATCH_JUSTIFY_SYSTEM, match_justify_user
from .schemas import CandidateProfile, JobSpec, MatchResult, SkillMatch
from .skills import SkillMatcher

log = get_logger(__name__)


def _verdict(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Potential match"
    return "Weak match"


def _score(
    skill_cov: float, pref_cov: float, cand_years: Optional[float], req_years: Optional[float]
) -> tuple[float, Optional[bool]]:
    if req_years and req_years > 0 and cand_years is not None:
        exp_factor = min(1.0, cand_years / req_years)
        overall = skill_cov * 68 + pref_cov * 10 + exp_factor * 22
        exp_ok = (cand_years + 0.25) >= req_years
    else:
        # Experience unknown -> weight entirely on skills.
        overall = skill_cov * 88 + pref_cov * 12
        exp_ok = None
    return round(min(100.0, max(0.0, overall)), 1), exp_ok


def _fallback_justification(result: MatchResult, job: JobSpec) -> tuple[str, str]:
    matched = [m.skill for m in result.matched_skills if m.matched]
    strong = ", ".join(matched[:5]) or "few of the required skills"
    gaps = ", ".join(result.missing_skills[:5])
    exp_txt = ""
    if result.required_years is not None and result.candidate_years is not None:
        exp_txt = (
            f" Experience: {result.candidate_years:g}y vs {result.required_years:g}y required"
            f" ({'meets' if result.experience_ok else 'below'} the bar)."
        )
    justification = (
        f"{result.verdict} ({result.overall_score:.0f}/100) for {job.title}. "
        f"Matches {len(matched)}/{len(result.matched_skills)} required skills including {strong}."
        + (f" Missing: {gaps}." if gaps else "")
        + exp_txt
    )
    if result.overall_score >= 75:
        rec = "Strong fit — advance to technical interview."
    elif result.overall_score >= 50:
        rec = f"Possible fit — screen for {gaps or 'gaps'}."
    else:
        rec = "Not a strong fit for this role."
    return justification, rec


async def match_candidate(
    job: JobSpec, candidate: CandidateProfile, *, justify: bool = True
) -> MatchResult:
    matcher = SkillMatcher()

    req_matches, missing, additional = matcher.match(job.required_skills, candidate.skills)
    matched_skills = [
        SkillMatch(skill=m.required, matched=m.matched, evidence=m.evidence, score=m.score)
        for m in req_matches
    ]
    skill_cov = (
        sum(1 for m in req_matches if m.matched) / len(req_matches) if req_matches else 1.0
    )

    pref_cov = 0.0
    if job.preferred_skills:
        pref_matches, _, _ = matcher.match(job.preferred_skills, candidate.skills)
        pref_cov = (
            sum(1 for m in pref_matches if m.matched) / len(pref_matches)
            if pref_matches
            else 0.0
        )

    cand_years = candidate.total_years_experience
    req_years = job.min_years_experience
    overall, exp_ok = _score(skill_cov, pref_cov, cand_years, req_years)

    result = MatchResult(
        overall_score=overall,
        verdict=_verdict(overall),
        skill_coverage=round(skill_cov, 3),
        experience_ok=exp_ok,
        candidate_years=cand_years,
        required_years=req_years,
        matched_skills=matched_skills,
        missing_skills=missing,
        additional_skills=additional[:15],
    )

    # -- Justification: deterministic recommendation + SLM-written explanation -- #
    justification, recommendation = _fallback_justification(result, job)
    if justify:
        facts: Dict[str, Any] = {
            "job_title": job.title,
            "job_seniority": job.seniority,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
            "skill_coverage_pct": round(skill_cov * 100),
            "matched_required_skills": [
                {"skill": m.skill, "evidence": m.evidence}
                for m in matched_skills if m.matched
            ],
            "missing_required_skills": result.missing_skills,
            "preferred_skills_coverage_pct": round(pref_cov * 100) if job.preferred_skills else None,
            "candidate_additional_skills": result.additional_skills,
            "candidate_years_experience": cand_years,
            "required_years_experience": req_years,
            "experience_meets_requirement": exp_ok,
            "candidate_title": candidate.current_title or candidate.headline,
            "candidate_strengths": candidate.strengths[:6],
            "candidate_summary": (candidate.summary or "")[:700],
            "candidate_skill_years": [
                {"skill": s.skill, "years": s.years}
                for s in skill_experience(candidate)[:15] if s.evidenced
            ],
            "key_responsibilities": job.responsibilities[:6],
        }
        try:
            text = await chat_text(
                MATCH_JUSTIFY_SYSTEM,
                match_justify_user(facts),
                temperature=0.3,
                max_tokens=1100,
                think=settings.domain_reasoning,
            )
            text = text.strip()
            if len(text) >= 40:
                justification = text
        except Exception as exc:  # noqa: BLE001
            log.warning("Match justification LLM call failed (%s); using fallback.", exc)

    result.justification = justification
    result.recommendation = recommendation
    return result
