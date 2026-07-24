"""PDF / image → text pipeline.

Strategy:
  1. Try fast, exact native-text extraction (PyMuPDF, then pdfplumber).
  2. For any page that yields little/no text (scanned or image-only), rasterize
     it at OCR_DPI and run PaddleOCR (PP-OCRv5).
Perfect text on digital PDFs, OCR only where needed — best accuracy at lowest cost.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

# A page with fewer than this many chars of native text is treated as "needs OCR".
_NATIVE_MIN_CHARS = 40
# Clamp the rasterized long edge so a huge/high-DPI page can't blow up memory.
_MAX_PIXELS_EDGE = 4000


@dataclass
class PageResult:
    page: int
    method: str  # native | ocr | empty
    char_count: int
    text: str = ""
    confidence: Optional[float] = None  # mean OCR confidence (native pages: null)


@dataclass
class PdfExtraction:
    text: str
    num_pages: int
    status: str = "ok"  # ok | partial | no_text
    overall_confidence: Optional[float] = None
    pages: List[PageResult] = field(default_factory=list)

    @property
    def method_summary(self) -> dict:
        summary: dict = {}
        for p in self.pages:
            summary[p.method] = summary.get(p.method, 0) + 1
        return summary

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "num_pages": self.num_pages,
            "status": self.status,
            "overall_confidence": self.overall_confidence,
            "method_summary": self.method_summary,
            "pages": [asdict(p) for p in self.pages],
        }


def _finalize(pages: List[PageResult]) -> PdfExtraction:
    full = "\n\n".join(p.text for p in pages if p.text.strip()).strip()
    confs = [p.confidence for p in pages if p.confidence is not None]
    overall = round(sum(confs) / len(confs), 4) if confs else None
    if not full:
        status = "no_text"
    elif any(not p.text.strip() for p in pages):
        status = "partial"
    else:
        status = "ok"
    return PdfExtraction(text=full, num_pages=len(pages), status=status,
                         overall_confidence=overall, pages=pages)


def _pdfplumber_page_text(data: bytes, page_index: int) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if page_index < len(pdf.pages):
                return pdf.pages[page_index].extract_text() or ""
    except Exception as exc:  # noqa: BLE001
        log.debug("pdfplumber failed on page %d: %s", page_index, exc)
    return ""


def _ocr_page(page: "fitz.Page", dpi: int) -> tuple[str, float]:
    import numpy as np

    from .paddle import get_ocr_engine

    # Force RGB so grayscale/CMYK pages don't crash the reshape.
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    if max(pix.width, pix.height) > _MAX_PIXELS_EDGE and dpi > 72:
        scale = _MAX_PIXELS_EDGE / max(pix.width, pix.height)
        pix = page.get_pixmap(dpi=max(72, int(dpi * scale)),
                              colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return get_ocr_engine().ocr_image(img)


def extract_pdf(
    data: bytes,
    *,
    prefer_native: Optional[bool] = None,
    use_ocr: Optional[bool] = None,
    dpi: Optional[int] = None,
) -> PdfExtraction:
    prefer_native = settings.ocr_prefer_native_text if prefer_native is None else prefer_native
    use_ocr = settings.ocr_enabled if use_ocr is None else use_ocr
    dpi = settings.ocr_dpi if dpi is None else dpi
    max_pages = settings.ocr_max_pages

    doc = fitz.open(stream=data, filetype="pdf")
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise ValueError("PDF is password-protected / encrypted.")

    pages: List[PageResult] = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                log.info("Stopping at OCR_MAX_PAGES=%d (document has %d).", max_pages, doc.page_count)
                break

            native_text = ""
            if prefer_native:
                native_text = (page.get_text("text") or "").strip()
                if len(native_text) < _NATIVE_MIN_CHARS:
                    alt = _pdfplumber_page_text(data, i).strip()
                    if len(alt) > len(native_text):
                        native_text = alt

            if len(native_text) >= _NATIVE_MIN_CHARS:
                pages.append(PageResult(page=i + 1, method="native",
                                        char_count=len(native_text), text=native_text))
                continue

            if use_ocr:
                try:
                    ocr_text, conf = _ocr_page(page, dpi)
                    ocr_text = ocr_text.strip()
                    if ocr_text:
                        pages.append(PageResult(page=i + 1, method="ocr",
                                                char_count=len(ocr_text), text=ocr_text,
                                                confidence=round(conf, 4)))
                        continue
                except Exception as exc:  # noqa: BLE001
                    log.warning("OCR failed on page %d: %s", i + 1, exc)

            pages.append(PageResult(page=i + 1, method="native" if native_text else "empty",
                                    char_count=len(native_text), text=native_text))
    finally:
        doc.close()

    return _finalize(pages)


def extract_image(data: bytes) -> PdfExtraction:
    """OCR a single image (png/jpg/...)."""
    import numpy as np
    from PIL import Image

    from .paddle import get_ocr_engine

    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img)
    text, conf = get_ocr_engine().ocr_image(arr)
    text = text.strip()
    pr = PageResult(page=1, method="ocr" if text else "empty",
                    char_count=len(text), text=text,
                    confidence=round(conf, 4) if text else None)
    return _finalize([pr])


def _looks_like_pdf(data: bytes) -> bool:
    return data[:5].lstrip().startswith(b"%PDF")


def extract_any(filename: str, data: bytes) -> PdfExtraction:
    name = (filename or "").lower()
    if name.endswith(".pdf") or _looks_like_pdf(data):
        return extract_pdf(data)
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
        return extract_image(data)
    # Unknown extension + not PDF magic: try image, then fall back to text decode.
    try:
        return extract_image(data)
    except Exception:
        text = data.decode("utf-8", "replace").strip()
        return _finalize([PageResult(page=1, method="native" if text else "empty",
                                     char_count=len(text), text=text)])
