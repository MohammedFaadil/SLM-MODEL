"""Recruiting-domain endpoints (in addition to the OpenAI-compatible /v1 API).

  POST /api/ocr/parse         multipart file  -> extracted text (+ per-page method)
  POST /api/resume/parse      multipart file  -> OCR + structured CandidateProfile
  POST /api/resume/parse-text json {text}     -> structured CandidateProfile
  POST /api/jobs              json JobCreate   -> enriched JobSpec
  POST /api/match             json {job,cand}  -> MatchResult (score + AI justification)
  POST /api/match/upload      multipart        -> one-shot: file + job fields -> MatchResult
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..config import settings
from ..domain.job_service import create_job
from ..domain.matching_service import match_candidate
from ..domain.resume_service import parse_resume
from ..domain.schemas import (
    CandidateProfile,
    JobCreateRequest,
    JobSpec,
    MatchResult,
)
from ..logging_conf import get_logger
from ..persistence import repo
from ..security import require_api_key

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["domain"], dependencies=[Depends(require_api_key)])


# --------------------------------------------------------------------------- #
#  request/response models
# --------------------------------------------------------------------------- #
class TextIn(BaseModel):
    text: str


class MatchRequest(BaseModel):
    job: JobSpec
    candidate: CandidateProfile
    justify: bool = True


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (> {settings.max_upload_mb} MB).",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


# --------------------------------------------------------------------------- #
#  OCR
# --------------------------------------------------------------------------- #
@router.post("/ocr/parse")
async def ocr_parse(file: UploadFile = File(...)) -> dict:
    from ..ocr.pdf import extract_any

    data = await _read_upload(file)
    try:
        extraction = extract_any(file.filename or "upload.pdf", data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse document: {exc}")
    return extraction.to_dict()


# --------------------------------------------------------------------------- #
#  Resume -> structured profile
# --------------------------------------------------------------------------- #
@router.post("/resume/parse", response_model=CandidateProfile)
async def resume_parse_file(file: UploadFile = File(...)) -> CandidateProfile:
    from ..ocr.pdf import extract_any

    data = await _read_upload(file)
    extraction = extract_any(file.filename or "upload.pdf", data)
    if not extraction.text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the document (even with OCR).",
        )
    profile = await parse_resume(extraction.text)
    await run_in_threadpool(repo.save_candidate, profile)
    return profile


@router.post("/resume/parse-text", response_model=CandidateProfile)
async def resume_parse_text(body: TextIn) -> CandidateProfile:
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    profile = await parse_resume(body.text)
    await run_in_threadpool(repo.save_candidate, profile)
    return profile


# --------------------------------------------------------------------------- #
#  Job creation
# --------------------------------------------------------------------------- #
@router.post("/jobs", response_model=JobSpec)
async def jobs_create(body: JobCreateRequest) -> JobSpec:
    if not body.title.strip():
        raise HTTPException(status_code=400, detail="Job title is required.")
    spec = await create_job(body)
    await run_in_threadpool(repo.save_job, spec)
    return spec


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
@router.post("/match", response_model=MatchResult)
async def match(body: MatchRequest) -> MatchResult:
    result = await match_candidate(body.job, body.candidate, justify=body.justify)
    await run_in_threadpool(repo.save_match, body.job, body.candidate, result)
    return result


@router.post("/match/upload", response_model=MatchResult)
async def match_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    skills: str = Form(""),
    min_years_experience: Optional[float] = Form(None),
    seniority: Optional[str] = Form(None),
    enrich_job: bool = Form(True),
    justify: bool = Form(True),
) -> MatchResult:
    """One-shot convenience: upload a resume + job fields, get a match back.

    `skills` may be a comma-separated string or a JSON array string.
    """
    from ..ocr.pdf import extract_any

    data = await _read_upload(file)
    extraction = extract_any(file.filename or "upload.pdf", data)
    if not extraction.text.strip():
        raise HTTPException(status_code=422, detail="No text extracted from resume.")

    try:
        skill_list = json.loads(skills) if skills.strip().startswith("[") else [
            s.strip() for s in skills.split(",") if s.strip()
        ]
    except Exception:
        skill_list = [s.strip() for s in skills.split(",") if s.strip()]

    job = await create_job(
        JobCreateRequest(
            title=title,
            skills=skill_list,
            min_years_experience=min_years_experience,
            seniority=seniority,
            enrich=enrich_job,
        )
    )
    candidate = await parse_resume(extraction.text)
    result = await match_candidate(job, candidate, justify=justify)
    await run_in_threadpool(repo.save_candidate, candidate)
    await run_in_threadpool(repo.save_match, job, candidate, result)
    return result


# --------------------------------------------------------------------------- #
#  History (only returns data when MSSQL persistence is enabled)
# --------------------------------------------------------------------------- #
@router.get("/history/jobs")
async def history_jobs(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"items": await run_in_threadpool(repo.list_jobs, limit)}


@router.get("/history/candidates")
async def history_candidates(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"items": await run_in_threadpool(repo.list_candidates, limit)}


@router.get("/history/matches")
async def history_matches(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"items": await run_in_threadpool(repo.list_matches, limit)}
