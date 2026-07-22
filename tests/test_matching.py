from app.domain.matching_service import _score, _verdict, match_candidate
from app.domain.schemas import CandidateProfile, JobSpec


def test_verdict_thresholds():
    assert _verdict(90) == "Strong match"
    assert _verdict(60) == "Potential match"
    assert _verdict(30) == "Weak match"


def test_score_experience_weighting():
    # Full skills, meets experience -> high.
    high, ok = _score(1.0, 1.0, cand_years=8, req_years=5)
    assert high >= 90 and ok is True
    # Full skills, half the required experience -> penalized.
    mid, ok2 = _score(1.0, 0.0, cand_years=2, req_years=5)
    assert 60 <= mid < 90 and ok2 is False
    # Unknown experience -> skills-only weighting, no exp flag.
    unk, ok3 = _score(0.5, 0.0, cand_years=None, req_years=None)
    assert ok3 is None and 40 <= unk <= 50


async def test_match_candidate_end_to_end():
    job = JobSpec(
        title="ML Engineer",
        min_years_experience=5,
        required_skills=["python", "pytorch", "aws", "kubernetes"],
        preferred_skills=["nlp"],
    )
    candidate = CandidateProfile(
        current_title="Senior ML Engineer",
        total_years_experience=7,
        skills=["python", "pytorch", "aws", "kubernetes", "nlp", "docker"],
    )
    # justify=False so no model call is needed.
    result = await match_candidate(job, candidate, justify=False)

    assert result.overall_score >= 90
    assert result.verdict == "Strong match"
    assert result.missing_skills == []
    assert result.experience_ok is True
    assert "docker" in result.additional_skills
    assert result.justification  # fallback justification always present
