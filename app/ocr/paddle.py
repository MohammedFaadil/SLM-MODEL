"""PaddleOCR wrapper (PP-OCRv4 / PP-OCRv5).

Written defensively because PaddleOCR's constructor and result format changed
between the 2.x and 3.x lines. We try modern signatures first and fall back,
and we parse both the classic ``[[box,(text,conf)],...]`` result and the newer
``{'rec_texts':[...], 'rec_scores':[...]}`` dict result.

Loaded lazily so importing the app never requires paddle to be installed.
"""
from __future__ import annotations

import threading
from statistics import mean
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_VERSION_MAP = {"v5": "PP-OCRv5", "v4": "PP-OCRv4", "v3": "PP-OCRv3"}


def _poly_topleft(poly: Any) -> Tuple[float, float]:
    """Return (min_y, min_x) of a detection polygon for reading-order sorting."""
    try:
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        return float(pts[:, 1].min()), float(pts[:, 0].min())
    except Exception:
        return 0.0, 0.0


class PaddleOCREngine:
    def __init__(self, lang: str, use_gpu: bool, version: str) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.version = version
        self._engine: Optional[Any] = None
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------ #
    def _build(self) -> Any:
        from paddleocr import PaddleOCR  # imported lazily

        ocr_version = _VERSION_MAP.get(self.version, "PP-OCRv4")
        device = "gpu" if self.use_gpu else "cpu"
        # Ordered from richest to simplest so we degrade gracefully across
        # paddleocr releases (kwargs get renamed / removed between versions):
        #   2.x -> use_gpu=True/False, show_log; 3.x -> device="gpu"/"cpu".
        attempts: List[dict] = []
        if self.use_gpu:
            # Try GPU-explicit signatures first (needs paddlepaddle-gpu / CUDA).
            attempts += [
                dict(lang=self.lang, use_angle_cls=True, ocr_version=ocr_version,
                     use_gpu=True, show_log=False),
                dict(lang=self.lang, ocr_version=ocr_version, device="gpu"),
                dict(lang=self.lang, use_textline_orientation=True, device="gpu"),
                dict(lang=self.lang, device="gpu"),
            ]
        attempts += [
            dict(lang=self.lang, use_angle_cls=True, ocr_version=ocr_version,
                 use_gpu=self.use_gpu, show_log=False),
            dict(lang=self.lang, ocr_version=ocr_version, device=device),
            dict(lang=self.lang, use_textline_orientation=True, ocr_version=ocr_version),
            dict(lang=self.lang, ocr_version=ocr_version),
            dict(lang=self.lang, use_angle_cls=True),
            dict(lang=self.lang),
        ]
        last_err: Optional[Exception] = None
        for kw in attempts:
            try:
                eng = PaddleOCR(**kw)
                log.info("PaddleOCR ready (requested=%s, kwargs=%s).", ocr_version, list(kw))
                return eng
            except Exception as exc:  # noqa: BLE001 - probe next signature
                last_err = exc
                continue
        raise RuntimeError(f"Could not initialize PaddleOCR: {last_err}")

    def _ensure(self) -> None:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._build()

    # -- inference --------------------------------------------------------- #
    def _run(self, img: np.ndarray) -> Any:
        assert self._engine is not None
        # Newer API prefers .predict(); classic uses .ocr(img, cls=True).
        for call in (
            lambda: self._engine.ocr(img),
            lambda: self._engine.ocr(img, cls=True),
            lambda: self._engine.predict(img),
        ):
            try:
                return call()
            except TypeError:
                continue
        # Last resort: let the natural error surface.
        return self._engine.ocr(img)

    @staticmethod
    def _lines_from_item(item: Any) -> List[Tuple[str, float, float, float]]:
        lines: List[Tuple[str, float, float, float]] = []

        # --- 3.x dict / OCRResult form ---
        getter = getattr(item, "get", None)
        if callable(getter):
            texts = getter("rec_texts")
            if texts is not None:
                scores = getter("rec_scores") or [1.0] * len(texts)
                polys = getter("rec_polys") or getter("dt_polys") or [None] * len(texts)
                for t, s, p in zip(texts, scores, polys):
                    y, x = _poly_topleft(p)
                    lines.append((str(t), float(s), y, x))
                return lines

        # --- classic 2.x list form: [[box, (text, conf)], ...] ---
        if isinstance(item, list):
            for entry in item:
                try:
                    box = entry[0]
                    txt = entry[1]
                    if isinstance(txt, (list, tuple)):
                        text, conf = str(txt[0]), float(txt[1])
                    else:
                        text, conf = str(txt), 1.0
                    y, x = _poly_topleft(box)
                    lines.append((text, conf, y, x))
                except Exception:
                    continue
        return lines

    def _extract(self, result: Any) -> Tuple[str, float]:
        if result is None:
            return "", 0.0
        items: Sequence[Any]
        if isinstance(result, dict) or hasattr(result, "get"):
            items = [result]
        elif isinstance(result, list):
            items = result
        else:
            items = [result]

        all_lines: List[Tuple[str, float, float, float]] = []
        for item in items:
            if item is None:
                continue
            all_lines.extend(self._lines_from_item(item))

        if not all_lines:
            return "", 0.0

        # Reading order: bucket by ~row (quantized y) then left-to-right.
        all_lines.sort(key=lambda ln: (round(ln[2] / 12.0), ln[3]))
        text = "\n".join(ln[0] for ln in all_lines if ln[0].strip())
        conf = mean(ln[1] for ln in all_lines) if all_lines else 0.0
        return text, float(conf)

    def ocr_image(self, img: np.ndarray) -> Tuple[str, float]:
        """OCR an RGB/BGR/gray numpy image -> (text, mean_confidence)."""
        self._ensure()
        arr = np.asarray(img)
        if arr.ndim == 2:  # grayscale -> 3ch
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr)
        return self._extract(self._run(arr))


_engine: Optional[PaddleOCREngine] = None
_engine_lock = threading.Lock()


def get_ocr_engine() -> PaddleOCREngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PaddleOCREngine(
                    lang=settings.ocr_lang,
                    use_gpu=settings.ocr_use_gpu,
                    version=settings.ocr_version,
                )
    return _engine
