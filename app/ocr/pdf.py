"""PDF / image → text pipeline for resumes.

Strategy (this is the "parse PDFs wisely" part):
  1. Try fast, exact native-text extraction (PyMuPDF, then pdfplumber).
  2. For any page that yields little/no text (scanned or image-only resume),
     rasterize it and run PaddleOCR.
This gives perfect text on digital resumes and OCR only where it's actually
needed — best accuracy at the lowest cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

# A page with fewer than this many characters of native text is treated as
# "needs OCR" (likely scanned).
_NATIVE_MIN_CHARS = 40


@dataclass
class PageResult:
    page: int
    method: str  # native | ocr | empty
    char_count: int
    text: str = ""
    ocr_confidence: Optional[float] = None


@dataclass
class PdfExtraction:
    text: str
    num_pages: int
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
            "method_summary": self.method_summary,
            "pages": [asdict(p) for p in self.pages],
        }


def _pdfplumber_page_text(data: bytes, page_index: int) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(_bytes_io(data)) as pdf:
            if page_index < len(pdf.pages):
                return pdf.pages[page_index].extract_text() or ""
    except Exception as exc:  # noqa: BLE001
        log.debug("pdfplumber failed on page %d: %s", page_index, exc)
    return ""


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


def _ocr_page(page: "fitz.Page", dpi: int) -> tuple[str, float]:
    import numpy as np

    from .paddle import get_ocr_engine

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    engine = get_ocr_engine()
    return engine.ocr_image(img)


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

    doc = fitz.open(stream=data, filetype="pdf")
    pages: List[PageResult] = []

    for i, page in enumerate(doc):
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
                                            ocr_confidence=round(conf, 4)))
                    continue
            except Exception as exc:  # noqa: BLE001
                log.warning("OCR failed on page %d: %s", i + 1, exc)

        # Nothing worked — keep whatever native text we had (may be empty).
        pages.append(PageResult(page=i + 1, method="native" if native_text else "empty",
                                char_count=len(native_text), text=native_text))

    doc.close()
    full = "\n\n".join(p.text for p in pages if p.text.strip())
    return PdfExtraction(text=full.strip(), num_pages=len(pages), pages=pages)


def extract_image(data: bytes) -> PdfExtraction:
    """OCR a single image (png/jpg) resume."""
    import numpy as np
    from PIL import Image

    from .paddle import get_ocr_engine

    img = Image.open(_bytes_io(data)).convert("RGB")
    arr = np.asarray(img)
    text, conf = get_ocr_engine().ocr_image(arr)
    text = text.strip()
    pr = PageResult(page=1, method="ocr" if text else "empty",
                    char_count=len(text), text=text, ocr_confidence=round(conf, 4))
    return PdfExtraction(text=text, num_pages=1, pages=[pr])


def extract_any(filename: str, data: bytes) -> PdfExtraction:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
        return extract_image(data)
    # Assume PDF bytes; fall back to plain-text decode.
    try:
        return extract_pdf(data)
    except Exception:
        text = data.decode("utf-8", "replace").strip()
        return PdfExtraction(
            text=text,
            num_pages=1,
            pages=[PageResult(page=1, method="native", char_count=len(text), text=text)],
        )
