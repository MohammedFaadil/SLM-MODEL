"""Local embeddings (sentence-transformers) for serving /v1/embeddings.

Kept optional: if sentence-transformers/torch aren't installed, importing this
module raises and the route falls back to the upstream embeddings endpoint.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self, model_name: str) -> None:
        log.info("Loading embedding model '%s' (first run downloads weights)...", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self._dim = self._model.get_sentence_embedding_dimension()
        log.info("Embedding model ready (dim=%d).", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
        vecs = self._model.encode(
            batch,
            normalize_embeddings=True,  # cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        return float(np.dot(va, vb))

    def cross_similarity(
        self, a: Sequence[str], b: Sequence[str]
    ) -> np.ndarray:
        """Return an |a| x |b| cosine matrix (both sides pre-normalized)."""
        if not a or not b:
            return np.zeros((len(a), len(b)), dtype=float)
        va = self.encode(a)
        vb = self.encode(b)
        return va @ vb.T

    def openai_response(
        self,
        input_: Union[str, List[str], None],
        model: Optional[str] = None,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        if input_ is None:
            input_ = ""
        items = [input_] if isinstance(input_, str) else list(input_)
        # Ignore token-id inputs; embed as strings.
        items = [str(x) for x in items]
        vecs = self.encode(items)

        data = []
        for i, vec in enumerate(vecs):
            if dimensions and dimensions < len(vec):
                vec = vec[:dimensions]  # Matryoshka truncation
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            if encoding_format == "base64":
                import base64

                emb: Any = base64.b64encode(
                    np.asarray(vec, dtype="<f4").tobytes()
                ).decode("ascii")
            else:
                emb = vec.tolist()
            data.append({"object": "embedding", "index": i, "embedding": emb})

        approx_tokens = sum(len(x.split()) for x in items)
        return {
            "object": "list",
            "data": data,
            "model": model or self.model_name,
            "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens},
        }


_embedder: Optional[Embedder] = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = Embedder(settings.embedding_model)
    return _embedder


def embeddings_available() -> bool:
    try:
        get_embedder()
        return True
    except Exception as exc:  # noqa: BLE001
        log.info("Embeddings not available: %s", exc)
        return False
