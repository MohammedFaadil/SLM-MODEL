"""Skill normalization, extraction, and semantic matching.

Matching is the accuracy-critical piece, so it is deterministic and grounded:
  * canonicalization collapses obvious variants (js -> javascript, k8s ->
    kubernetes, reactjs -> react, ...);
  * semantic matching uses embeddings when available (bge-small), so "PyTorch"
    can satisfy "deep learning frameworks", falling back to fuzzy token ratio
    when embeddings aren't installed.
The LLM is used for *explanation*, not for computing the match — so the score
and the justification never disagree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

# Canonical forms for common variants/abbreviations.
_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "vuejs": "vue",
    "vue.js": "vue",
    "nextjs": "next.js",
    "py": "python",
    "golang": "go",
    "c sharp": "c#",
    "csharp": "c#",
    "dotnet": ".net",
    ".net core": ".net",
    "postgres": "postgresql",
    "postgre": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "gcp": "google cloud",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ci/cd": "ci-cd",
    "cicd": "ci-cd",
    "rest api": "rest",
    "restful": "rest",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "html5": "html",
    "css3": "css",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "llms": "llm",
    "large language models": "llm",
}

# A modest built-in lexicon for heuristic extraction when the model is offline.
_LEXICON: List[str] = [
    # languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "bash", "perl",
    # web / frameworks
    "react", "angular", "vue", "next.js", "node.js", "express", "django", "flask",
    "fastapi", "spring", "spring boot", ".net", "asp.net", "laravel", "rails",
    "svelte", "graphql", "rest", "html", "css", "tailwind", "bootstrap", "redux",
    # data / ml
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
    "numpy", "spark", "hadoop", "airflow", "kafka", "llm", "generative ai",
    "transformers", "hugging face", "opencv", "data analysis", "statistics",
    "power bi", "tableau", "excel",
    # databases
    "postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle", "sql server",
    "elasticsearch", "dynamodb", "cassandra", "snowflake",
    # cloud / devops
    "aws", "azure", "google cloud", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci-cd", "git", "github", "gitlab", "linux", "nginx", "microservices",
    "serverless", "prometheus", "grafana",
    # general / soft
    "project management", "agile", "scrum", "leadership", "communication",
    "stakeholder management", "problem solving", "team management",
    "product management", "data structures", "algorithms", "system design",
]

_word = re.compile(r"[a-z0-9][a-z0-9+.#\- ]*[a-z0-9+#]")


def normalize_skill(skill: str) -> str:
    s = (skill or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:")
    return _ALIASES.get(s, s)


def normalize_skills(skills: Sequence[str]) -> List[str]:
    seen: Dict[str, str] = {}
    for raw in skills:
        n = normalize_skill(raw)
        if n and n not in seen:
            seen[n] = raw.strip()
    return list(seen.keys())


def extract_skills_heuristic(text: str) -> List[str]:
    """Lexicon scan — a fallback when the model isn't available."""
    if not text:
        return []
    low = text.lower()
    found: List[str] = []
    for skill in _LEXICON:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, low):
            found.append(skill)
    return normalize_skills(found)


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
@dataclass
class _Match:
    required: str
    matched: bool
    evidence: Optional[str]
    score: float


class SkillMatcher:
    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = settings.semantic_match_threshold if threshold is None else threshold
        self._embedder = None
        self._embed_failed = False

    def _get_embedder(self):
        if settings.embeddings_mode == "off" or self._embed_failed:
            return None
        if self._embedder is None:
            try:
                from ..embeddings.embedder import get_embedder

                self._embedder = get_embedder()
            except Exception as exc:  # noqa: BLE001
                log.info("Embeddings unavailable, using fuzzy matching (%s).", exc)
                self._embed_failed = True
                return None
        return self._embedder

    def _fuzzy_score(self, a: str, b: str) -> float:
        try:
            from rapidfuzz import fuzz

            return max(
                fuzz.token_set_ratio(a, b),
                fuzz.partial_ratio(a, b),
            ) / 100.0
        except Exception:
            return 1.0 if a == b else 0.0

    def match(
        self, required: Sequence[str], candidate: Sequence[str]
    ) -> Tuple[List[_Match], List[str], List[str]]:
        req = normalize_skills(required)
        cand = normalize_skills(candidate)
        if not req:
            return [], [], cand

        matches: List[_Match] = []
        used_cand: set = set()

        embedder = self._get_embedder()
        sim_matrix = None
        if embedder is not None and cand:
            try:
                sim_matrix = embedder.cross_similarity(req, cand)
            except Exception as exc:  # noqa: BLE001
                log.info("Embedding similarity failed, using fuzzy (%s).", exc)
                sim_matrix = None

        for i, r in enumerate(req):
            best_j, best_score = -1, 0.0
            for j, c in enumerate(cand):
                if r == c:
                    best_j, best_score = j, 1.0
                    break
                if sim_matrix is not None:
                    score = float(sim_matrix[i][j])
                else:
                    score = self._fuzzy_score(r, c)
                if score > best_score:
                    best_j, best_score = j, score

            threshold = self.threshold if sim_matrix is not None else 0.82
            if best_j >= 0 and best_score >= threshold:
                matches.append(_Match(r, True, cand[best_j], round(best_score, 3)))
                used_cand.add(cand[best_j])
            else:
                ev = cand[best_j] if best_j >= 0 else None
                matches.append(_Match(r, False, ev, round(best_score, 3)))

        missing = [m.required for m in matches if not m.matched]
        additional = [c for c in cand if c not in used_cand]
        return matches, missing, additional
