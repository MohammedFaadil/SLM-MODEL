"""Prompt templates.

Engineered for an 8B model: explicit schema, "output only JSON", "do not
invent", and grounding rules. Small models obey tight, concrete instructions far
better than open-ended ones — that is most of the accuracy here.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
#  Resume parsing (single call: structured extraction + recruiter summary)
# --------------------------------------------------------------------------- #
RESUME_PARSE_SYSTEM = """You are an expert technical recruiter and resume parser.
Extract structured data from the resume text and write a short recruiter-facing summary.

Rules:
- Output ONLY a single valid JSON object. No markdown, no commentary, no code fences.
- Use null for unknown scalar fields and [] for unknown lists. NEVER invent facts.
- "skills": concise canonical skill names (e.g. "react", "python", "aws", "kubernetes").
  Include technologies, tools, frameworks, and clear professional competencies actually
  evidenced in the resume. Deduplicate. Aim for the 10-30 most relevant.
- "total_years_experience": a number estimated from the work history date ranges
  (sum of professional experience, ignoring overlaps). Use 0 if none/student.
- "summary": 3-5 sentences, factual, recruiter-facing: who they are, core strengths,
  domains, and seniority. Grounded strictly in the resume. No hype.
- "strengths": 3-6 short bullet phrases capturing standout strengths.

JSON schema (keys and shapes to follow exactly):
{
  "contact": {"name": str|null, "email": str|null, "phone": str|null,
              "location": str|null, "links": [str]},
  "headline": str|null,
  "current_title": str|null,
  "total_years_experience": number,
  "skills": [str],
  "experience": [{"title": str|null, "company": str|null, "start": str|null,
                  "end": str|null, "highlights": [str]}],
  "education": [{"degree": str|null, "field": str|null, "institution": str|null,
                 "year": str|null}],
  "certifications": [str],
  "strengths": [str],
  "summary": str
}"""


def resume_parse_user(resume_text: str, max_chars: int = 16000) -> str:
    text = resume_text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return f"RESUME TEXT:\n\"\"\"\n{text}\n\"\"\"\n\nReturn the JSON object now."


# --------------------------------------------------------------------------- #
#  Job creation / enrichment
# --------------------------------------------------------------------------- #
JOB_CREATE_SYSTEM = """You are a senior technical recruiter writing a precise job specification.
Given a title, seed skills, and experience level, produce a complete, realistic job spec.

Rules:
- Output ONLY a single valid JSON object. No markdown, no commentary, no code fences.
- Keep required_skills grounded in the provided skills plus the few technologies clearly
  implied by the role. Do NOT pad with unrelated skills.
- required_skills: must-haves. preferred_skills: nice-to-haves. Canonical short names.
- responsibilities: 5-8 concise bullet phrases. qualifications: 4-6 bullet phrases.
- description: a clean, professional 2-3 paragraph job description (plain text).
- Preserve the given min_years_experience and seniority if provided.

JSON schema:
{
  "title": str,
  "seniority": str|null,
  "min_years_experience": number|null,
  "required_skills": [str],
  "preferred_skills": [str],
  "responsibilities": [str],
  "qualifications": [str],
  "location": str|null,
  "employment_type": str|null,
  "description": str
}"""


def job_create_user(payload: Dict[str, Any]) -> str:
    return (
        "JOB INPUT (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nReturn the completed job-spec JSON object now."
    )


# --------------------------------------------------------------------------- #
#  Match justification (grounded in pre-computed facts)
# --------------------------------------------------------------------------- #
MATCH_JUSTIFY_SYSTEM = """You are a technical recruiter explaining a candidate-job fit to a hiring manager.
You are given FACTS already computed by a matching engine. Your job is ONLY to explain them.

Rules:
- Output ONLY a valid JSON object: {"justification": str, "recommendation": str}.
- Base everything strictly on the provided FACTS. Do NOT invent skills or experience.
- justification: 3-5 sentences. State overall fit, name the strongest matched skills,
  call out the most important missing skills, and note the experience fit. Be honest and
  specific; tone must match the score (do not oversell a weak match).
- recommendation: one short line, e.g. "Strong fit — advance to technical interview",
  "Possible fit — screen for <gap>", or "Not a fit for this role".
"""


def match_justify_user(facts: Dict[str, Any]) -> str:
    return (
        "FACTS (JSON):\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nReturn the JSON object now."
    )
