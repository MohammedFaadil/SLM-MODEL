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
    """Read the upload in bounded chunks, aborting as soon as it exceeds the limit
    so a huge upload can't force a multi-GB allocation."""
    max_bytes = settings.max_upload_bytes
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(413, detail=f"File too large (> {settings.max_upload_mb} MB).")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, detail=f"File too large (> {settings.max_upload_mb} MB).")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    return b"".join(chunks)


@router.post("/ocr/parse")
async def ocr_parse(file: UploadFile = File(...)) -> dict:
    """Extract text from a PDF/image. Digital PDFs use exact native text; scanned
    pages fall back to PaddleOCR (PP-OCRv5). Returns text + per-page method."""
    from ..ocr.pdf import extract_any

    data = await _read_upload(file)
    try:
        # OCR/PDF work is blocking + CPU-heavy -> run off the event loop.
        extraction = await run_in_threadpool(extract_any, file.filename or "upload.pdf", data)
    except (RuntimeError, MemoryError, ImportError) as exc:  # engine/resource fault
        log.error("OCR engine unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="OCR engine unavailable.")
    except Exception as exc:  # noqa: BLE001 - the client's document is unreadable
        raise HTTPException(status_code=422, detail=f"Could not parse document: {exc}")
    return extraction.to_dict()
