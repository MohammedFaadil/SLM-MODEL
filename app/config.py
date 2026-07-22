"""Central configuration.

Everything is driven by environment variables (see .env.example). Defaults are
tuned for a LOCAL CPU box running Ollama; moving to cloud GPU is a matter of
pointing UPSTREAM_BASE_URL at vLLM and setting MODEL_NAME to the served id.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Server ----
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ---- Gateway auth ----
    # Comma-separated list of accepted API keys. Empty => open (localhost dev).
    gateway_api_keys: str = ""

    # ---- LLM backend ----
    llm_backend: str = "openai_upstream"  # openai_upstream | mock
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str = "ollama"
    model_name: str = "qwen3:8b"
    served_model_id: str = "slm-qwen3-8b"
    force_model: bool = True
    enable_thinking: bool = False
    request_timeout: float = 600.0

    # ---- Embeddings ----
    embeddings_mode: str = "local"  # local | upstream | off
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    semantic_match_threshold: float = 0.62

    # ---- OCR ----
    ocr_enabled: bool = True
    ocr_version: str = "v5"  # v5 | v4
    ocr_lang: str = "en"
    ocr_use_gpu: bool = False
    ocr_prefer_native_text: bool = True
    ocr_dpi: int = 200

    # ---- Uploads ----
    max_upload_mb: int = 25

    @field_validator("upstream_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def api_key_list(self) -> List[str]:
        return [k.strip() for k in self.gateway_api_keys.split(",") if k.strip()]

    @property
    def auth_required(self) -> bool:
        return len(self.api_key_list) > 0

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
