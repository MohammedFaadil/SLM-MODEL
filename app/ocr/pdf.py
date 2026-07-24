"""PDF / image → text pipeline.

Strategy:
  1. Try fast, exact native-text extraction (PyMuPDF, then pdfplumber).
  2. For any page that yields little/no text (scanned or image-only), rasterize
     it at a bounded DPI and run PaddleOCR (PP-OCRv5).
Perfect text on digital PDFs, OCR only where needed — best accuracy at lowest cost.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

# A page with fewer than this many chars of native text is treated as "needs OCR".
_NATIVE_MIN_CHARS = 40
# Bound the rasterized long edge (pixels) so a huge/high-DPI page can't OOM.
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
    num_pages: int                        # pages actually processed
    total_pages: int = 0                  # pages in the document (>= num_pages if capped)
    status: str = "ok"                    # ok | partial | no_text
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
            "total_pages": self.total_pages or self.num_pages,
            "status": self.status,
            "overall_confidence": self.overall_confidence,
            "method_summary": self.method_summary,
            "pages": [asdict(p) for p in self.pages],
        }


def _finalize(pages: List[PageResult], total_pages: Optional[int] = None) -> PdfExtraction:
    full = "\n\n".join(p.text for p in pages if p.text.strip()).strip()
    confs = [p.confidence for p in pages if p.confidence is not None]
    overall = round(sum(confs) / len(confs), 4) if confs else None
    if not full:
        status = "no_text"
    elif any(not p.text.strip() for p in pages):
        status = "partial"
    else:
        status = "ok"
    return PdfExtraction(text=full, num_pages=len(pages),
                         total_pages=total_pages if total_pages is not None else len(pages),
                         status=status, overall_confidence=overall, pages=pages)


class _Pdfplumber:
    """Open pdfplumber ONCE and reuse pages (avoids O(n^2) re-parsing per page)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pdf = None
        self._opened = False

    def page_text(self, i: int) -> str:
        if not self._opened:
            self._opened = True
            try:
                # pyrefly: ignore [missing-import]
                import pdfplumber

                self._pdf = pdfplumber.open(io.BytesIO(self._data))
            except Exception as exc:  # noqa: BLE001
                log.debug("pdfplumber open failed: %s", exc)
                self._pdf = None
        if self._pdf is None:
            return ""
        try:
            if i < len(self._pdf.pages):
                return self._pdf.pages[i].extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            log.debug("pdfplumber page %d failed: %s", i, exc)
        return ""

    def close(self) -> None:
        if self._pdf is not None:
            try:
                self._pdf.close()
            except Exception:  # noqa: BLE001
                pass


def _ocr_page(page: "fitz.Page", dpi: int) -> tuple[str, float]:
    import numpy as np

    from .paddle import get_ocr_engine

    # Bound the render BEFORE rasterizing: compute an effective DPI so the long edge
    # is ~<= _MAX_PIXELS_EDGE, so we never allocate a giant pixmap first.
    long_in = max(page.rect.width, page.rect.height) / 72.0  # page long edge in inches
    eff = min(dpi, int(_MAX_PIXELS_EDGE / long_in)) if long_in > 0 else dpi
    eff = max(36, eff)
    # Force RGB so grayscale/CMYK pages don't break the reshape.
    pix = page.get_pixmap(dpi=eff, colorspace=fitz.csRGB, alpha=False)
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

    total_pages = doc.page_count
    plumber = _Pdfplumber(data)
    pages: List[PageResult] = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                log.info("Stopping at OCR_MAX_PAGES=%d (document has %d).", max_pages, total_pages)
                break

            native_text = ""
            if prefer_native:
                native_text = (page.get_text("text") or "").strip()
                if len(native_text) < _NATIVE_MIN_CHARS:
                    alt = plumber.page_text(i).strip()
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
        plumber.close()
        doc.close()

    return _finalize(pages, total_pages=total_pages)


def extract_image(data: bytes) -> PdfExtraction:
    """OCR a single image (png/jpg/...)."""
    import numpy as np
    # pyrefly: ignore [missing-import]
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
    return b"%PDF" in data[:1024]


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
