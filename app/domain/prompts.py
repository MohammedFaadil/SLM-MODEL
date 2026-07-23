"""Prompt templates.

Design principles for reliable 8B output:
  * Domain-generic wording (works for tech, sales, finance, healthcare, ...).
  * STRUCTURED data comes back as small, strict JSON (easy to parse).
  * LONG-FORM prose (job description, candidate summary, fit justification) is
    generated in SEPARATE plain-text calls — never crammed inside a JSON string,
    which is the #1 cause of broken JSON / generic fallbacks on small models.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# =========================================================================== #
#  RESUME
# =========================================================================== #
# 1) Structured extraction (strict JSON, no long prose).
RESUME_PARSE_SYSTEM = """You are an expert recruiter and resume parser for ALL industries
(technology, sales, finance, healthcare, operations, design, etc.).
Extract accurate structured data from the resume text.

Rules:
- Output ONLY a single valid JSON object. No markdown, no commentary, no code fences.
- Use null for unknown scalar fields and [] for unknown lists. NEVER invent facts.
- "skills": concise canonical skill/tool/competency names actually evidenced in the
  resume (e.g. "python", "salesforce", "financial modeling", "project management").
  Deduplicate. Return the 10-30 most relevant.
- "total_years_experience": a number estimated from the work-history date ranges
  (sum of professional experience, ignoring overlaps). 0 if none/student.
- "headline": one short line describing the candidate (their own title/level).
- "strengths": 3-6 short bullet phrases capturing standout, evidence-backed strengths.
- Keep "summary" to ONE factual sentence here (a detailed summary is written separately).

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


def resume_parse_user(resume_text: str, max_chars: int = 18000) -> str:
    text = resume_text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return f'RESUME TEXT:\n"""\n{text}\n"""\n\nReturn the JSON object now.'


# 2) Detailed recruiter summary / justification (plain text, reasoning-friendly).
RESUME_SUMMARY_SYSTEM = """You are a senior recruiter writing a candidate briefing for a hiring manager.
Write a clear, well-structured, DETAILED assessment of the candidate based ONLY on the
provided resume data. Cover:
- who they are and their seniority level,
- core strengths and the domains/functions they are strongest in,
- notable achievements or scope (teams, scale, impact) if evidenced,
- any obvious gaps or caveats a recruiter should note.

Rules: be factual and specific, ground every claim in the resume, do NOT invent
employers, titles, dates, or skills. No hype, no marketing language. 4-8 sentences,
plain prose (you may use short paragraphs). Do not output JSON or bullet lists of skills."""


def resume_summary_user(profile: Dict[str, Any], resume_text: str, max_chars: int = 8000) -> str:
    text = resume_text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    compact = {
        "name": (profile.get("contact") or {}).get("name"),
        "current_title": profile.get("current_title") or profile.get("headline"),
        "total_years_experience": profile.get("total_years_experience"),
        "skills": profile.get("skills", [])[:25],
        "experience": [
            {"title": e.get("title"), "company": e.get("company"),
             "start": e.get("start"), "end": e.get("end")}
            for e in (profile.get("experience") or [])[:8]
        ],
        "education": profile.get("education", [])[:4],
    }
    return (
        "EXTRACTED PROFILE (JSON):\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + f'\n\nRESUME TEXT:\n"""\n{text}\n"""\n\nWrite the candidate assessment now.'
    )


# =========================================================================== #
#  JOB
# =========================================================================== #
# 1) Structured job fields (strict JSON, no long description).
JOB_FIELDS_SYSTEM = """You are a senior recruiter turning a request into a precise job specification.
You are given a title, seed skills, experience level, and (often) a detailed free-text
prompt describing the role. Produce the STRUCTURED fields of the job.

Rules:
- Output ONLY a single valid JSON object. No markdown, no commentary, no code fences.
- Ground required_skills in the provided skills PLUS skills clearly implied by the title
  and the prompt. Do NOT pad with unrelated skills.
- required_skills = must-haves; preferred_skills = nice-to-haves. Canonical short names.
- responsibilities: 6-8 concise, specific bullet phrases tailored to the prompt.
- qualifications: 4-6 concise bullet phrases (education, experience, domain knowledge).
- Honor the given seniority / min_years_experience / location / employment_type if present;
  otherwise infer sensible values from the prompt (or null).
- Do NOT write the long job description here — that is generated separately.

JSON schema:
{
  "seniority": str|null,
  "min_years_experience": number|null,
  "required_skills": [str],
  "preferred_skills": [str],
  "responsibilities": [str],
  "qualifications": [str],
  "location": str|null,
  "employment_type": str|null
}"""


def job_fields_user(payload: Dict[str, Any]) -> str:
    return (
        "JOB REQUEST (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nReturn the structured job-fields JSON object now."
    )


# 2) Rich Markdown job description (plain text, reasoning-friendly).
JOB_DESCRIPTION_SYSTEM = """You are a senior recruiter writing a polished, engaging job description.
Use the provided title, requirements prompt, and structured fields to write a rich,
professional description in Markdown. Heavily customize the content to the specific role
and the user's prompt — do NOT produce a generic template. Use a confident recruiter tone
and strong action verbs. Ground everything in the provided inputs; do not contradict them.

Output ONLY Markdown (no JSON, no code fences) using EXACTLY this structure:

# Job Description: [Job Title]

### Job Title
[Job Title]

### Location
[from inputs, else "Remote / Hybrid"]

### Employment Type
[from inputs, else "Full-Time"]

### Experience
[experience level / years]

## Job Summary
[3-4 engaging sentences: the mission of the role, the impact this person will have, and
why it's a compelling opportunity — grounded in the prompt.]

## Key Responsibilities
[6-8 comprehensive bullets with strong action verbs, tailored to the role.]

## Required Skills
[the must-have skills as bullets]

## Preferred Skills
[nice-to-have skills as bullets]

## Qualifications
[education, years of experience, domain knowledge as bullets]

## Benefits
[5-6 attractive, role-appropriate benefits]"""


def job_description_user(payload: Dict[str, Any], fields: Dict[str, Any]) -> str:
    merged = {
        "title": payload.get("title"),
        "seniority": fields.get("seniority") or payload.get("seniority"),
        "min_years_experience": fields.get("min_years_experience") or payload.get("min_years_experience"),
        "location": fields.get("location") or payload.get("location"),
        "employment_type": fields.get("employment_type") or payload.get("employment_type"),
        "required_skills": fields.get("required_skills") or payload.get("skills"),
        "preferred_skills": fields.get("preferred_skills"),
        "responsibilities": fields.get("responsibilities"),
        "qualifications": fields.get("qualifications"),
        "requirements_prompt": payload.get("prompt") or payload.get("notes"),
    }
    return (
        "JOB INPUTS (JSON):\n"
        + json.dumps(merged, ensure_ascii=False, indent=2)
        + "\n\nWrite the Markdown job description now."
    )


# =========================================================================== #
#  MATCH JUSTIFICATION (plain text, grounded in pre-computed facts)
# =========================================================================== #
MATCH_JUSTIFY_SYSTEM = """You are a recruiter explaining a candidate-vs-job fit to a hiring manager.
You are given FACTS already computed by a matching engine (skill matches/gaps, experience,
scores, candidate strengths). Explain the fit clearly and honestly.

Write a DETAILED, well-reasoned justification that:
- opens with the overall verdict and why (reference the score),
- names the strongest matched requirements and what evidence supports them,
- calls out the most important missing/weak requirements and how much they matter,
- assesses the experience-level fit,
- ends with a clear hiring-manager takeaway: should they proceed, and why / why not.

Rules: base EVERYTHING strictly on the provided FACTS — do NOT invent skills, employers,
or experience. Match the tone to the score (never oversell a weak match). 4-7 sentences,
plain prose. Do NOT output JSON."""


def match_justify_user(facts: Dict[str, Any]) -> str:
    return (
        "FACTS (JSON):\n"
        + json.dumps(facts, ensure_ascii=False, indent=2, default=str)
        + "\n\nWrite the fit justification now."
    )
