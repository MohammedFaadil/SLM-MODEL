from app.domain.skills import (
    SkillMatcher,
    extract_skills_heuristic,
    normalize_skill,
    normalize_skills,
)


def test_normalize_aliases():
    assert normalize_skill("JS") == "javascript"
    assert normalize_skill("ReactJS") == "react"
    assert normalize_skill("K8s") == "kubernetes"
    assert normalize_skill("  Python  ") == "python"
    assert normalize_skill("scikit learn") == "scikit-learn"


def test_normalize_dedup_preserves_order():
    out = normalize_skills(["React", "reactjs", "Python", "py"])
    assert out == ["react", "python"]


def test_heuristic_extraction():
    text = "Experienced in Python, PyTorch and AWS. Built Docker/Kubernetes pipelines."
    skills = extract_skills_heuristic(text)
    for expected in ("python", "pytorch", "aws", "docker", "kubernetes"):
        assert expected in skills


def test_matcher_fuzzy_exact_and_missing():
    # EMBEDDINGS_MODE=off -> fuzzy path.
    matcher = SkillMatcher()
    required = ["python", "react", "aws"]
    candidate = ["Python", "ReactJS", "GCP"]  # aws missing (gcp != aws)
    matches, missing, additional = matcher.match(required, candidate)

    matched_names = {m.required for m in matches if m.matched}
    assert "python" in matched_names
    assert "react" in matched_names
    assert "aws" in missing
    assert "google cloud" in additional  # gcp normalized, unused
