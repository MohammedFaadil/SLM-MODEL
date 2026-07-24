"""OCR endpoint (in addition to the OpenAI-compatible /v1 API).

The main product owns all prompting/logic; this is the one non-/v1 capability the
gateway exposes so the product can turn a resume PDF/image into clean text.

  POST /api/ocr/parse   multipart file -> {text, num_pages, method_summary, pages[]}
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..config import settings
from ..logging_conf import get_logger
from ..security import require_api_key

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ocr"], dependencies=[Depends(require_api_key)])


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


@router.post("/ocr/parse")
async def ocr_parse(file: UploadFile = File(...)) -> dict:
    """Extract text from a PDF/image. Digital PDFs use exact native text; scanned
    pages fall back to PaddleOCR (PP-OCRv5). Returns text + per-page method."""
    from ..ocr.pdf import extract_any

    data = await _read_upload(file)
    try:
        # OCR/PDF work is blocking + CPU-heavy -> run off the event loop.
        extraction = await run_in_threadpool(extract_any, file.filename or "upload.pdf", data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse document: {exc}")
    return extraction.to_dict()
