"""The resume-extraction reliability fix: recover truncated / messy model JSON."""
from app.domain.llm_client import extract_json


def test_plain_and_fenced():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_think_block_stripped():
    assert extract_json('<think>reasoning here</think>\n{"a": 1}') == {"a": 1}


def test_trailing_commas():
    assert extract_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_truncated_inside_array_string():
    # Cut off mid-skill (token limit) — must still recover name + partial skills.
    out = extract_json('{"name": "Bob", "skills": ["python", "jav')
    assert out.get("name") == "Bob"
    assert "python" in out.get("skills", [])


def test_truncated_inside_nested_object():
    out = extract_json('{"name": "Bob", "experience": [{"title": "Engineer", "start": "2020"')
    assert out.get("name") == "Bob"
    exp = out.get("experience")
    assert isinstance(exp, list) and exp and exp[0]["title"] == "Engineer"


def test_truncated_after_colon():
    out = extract_json('{"name": "Bob", "total_years_experience":')
    assert out.get("name") == "Bob"
