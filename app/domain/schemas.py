"""Domain models: Job, Candidate, and Match results."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  Job
# --------------------------------------------------------------------------- #
class JobCreateRequest(BaseModel):
    title: str
    skills: List[str] = Field(default_factory=list)
    prompt: Optional[str] = None
    min_years_experience: Optional[float] = None
    seniority: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    notes: Optional[str] = None
    enrich: bool = True  # use the SLM to expand into a full JD


class JobSpec(BaseModel):
    title: str
    seniority: Optional[str] = None
    min_years_experience: Optional[float] = None
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    employment_type: Optional[str] = None
    description: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Candidate
# --------------------------------------------------------------------------- #
class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: List[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)  # skills/tools used in this role
    duration_years: Optional[float] = None           # computed from start/end


class EducationItem(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class CandidateProfile(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    headline: Optional[str] = None
    summary: str = ""  # AI-generated candidate summary / justification
    total_years_experience: Optional[float] = None
    current_title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
class SkillExperience(BaseModel):
    skill: str
    years: float                    # evidenced years of experience for this skill
    evidenced: bool = True          # False = listed but not tied to a dated role


class CandidateSummary(BaseModel):
    name: Optional[str] = None
    headline: Optional[str] = None
    summary: str = ""               # comprehensive HR-facing narrative
    total_years_experience: Optional[float] = None
    skills: List[str] = Field(default_factory=list)
    skill_experience: List[SkillExperience] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    top_domains: List[str] = Field(default_factory=list)


class SkillMatch(BaseModel):
    skill: str
    matched: bool
    evidence: Optional[str] = None  # the candidate skill/phrase that satisfied it
    score: float = 0.0              # 0..1 semantic similarity


class MatchResult(BaseModel):
    overall_score: float            # 0..100
    verdict: str                    # Strong match | Potential match | Weak match
    skill_coverage: float           # 0..1 fraction of required skills satisfied
    experience_ok: Optional[bool] = None
    candidate_years: Optional[float] = None
    required_years: Optional[float] = None
    matched_skills: List[SkillMatch] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    additional_skills: List[str] = Field(default_factory=list)
    justification: str = ""
    recommendation: str = ""
