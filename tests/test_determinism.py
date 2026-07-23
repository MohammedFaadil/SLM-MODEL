"""Verify domain LLM calls are configured for reproducible output."""
from app.config import settings
from app.domain.llm_client import _apply_sampling


def test_deterministic_forces_greedy_and_seed():
    assert settings.deterministic is True  # default
    payload: dict = {}
    _apply_sampling(payload, temperature=0.7)  # requested temp ignored when deterministic
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == settings.llm_seed == 42


def test_seed_present_even_when_sampling(monkeypatch):
    monkeypatch.setattr(settings, "deterministic", False)
    payload: dict = {}
    _apply_sampling(payload, temperature=0.5)
    assert payload["temperature"] == 0.5   # honored when not deterministic
    assert payload["seed"] == settings.llm_seed  # seed still pinned
