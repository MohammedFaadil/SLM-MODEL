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
    # Qwen3 reasoning for long-form domain tasks (job description, candidate
    # summary, fit justification). OFF by default = much faster (thinking emits a
    # long hidden reasoning block before every answer). Turn on only if you want
    # extra thoroughness and can accept ~2-3x slower responses.
    domain_reasoning: bool = False
    # Reproducibility: greedy decoding (temperature 0) + fixed seed on every
    # domain call, so the same inputs give the same output each time.
    deterministic: bool = True
    llm_seed: int = 42
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

    # ---- Production / efficiency ----
    # Comma-separated allowed CORS origins ("*" = any). Restrict in production.
    cors_allow_origins: str = "*"
    # Ping the model once on startup so the first real request isn't a cold load.
    warmup_on_start: bool = True
    # Outbound connection pool to the inference engine.
    httpx_max_connections: int = 128
    httpx_max_keepalive: int = 32

    # ---- Persistence (optional MSSQL) ----
    # Preferred: a full SQLAlchemy URL, e.g.
    #   mssql+pyodbc://user:pass@host:1433/DbName?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
    database_url: str = ""
    # Or provide components and we build the URL for you:
    mssql_host: str = ""
    mssql_port: int = 1433
    mssql_database: str = ""
    mssql_user: str = ""
    mssql_password: str = ""
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_trust_cert: bool = True
    mssql_encrypt: bool = True
    # When persistence is on, also log every /v1 request (model, latency, status).
    persist_requests: bool = True

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

    @property
    def cors_origin_list(self) -> List[str]:
        raw = [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]
        return raw or ["*"]

    @property
    def resolved_database_url(self) -> str:
        """Full SQLAlchemy URL, either given directly or built from MSSQL parts."""
        if self.database_url.strip():
            return self.database_url.strip()
        if self.mssql_host and self.mssql_database:
            from urllib.parse import quote_plus

            driver = quote_plus(self.mssql_driver)
            auth = ""
            if self.mssql_user:
                auth = f"{quote_plus(self.mssql_user)}:{quote_plus(self.mssql_password)}@"
            params = f"driver={driver}"
            params += f"&TrustServerCertificate={'yes' if self.mssql_trust_cert else 'no'}"
            params += f"&Encrypt={'yes' if self.mssql_encrypt else 'no'}"
            if not self.mssql_user:
                params += "&Trusted_Connection=yes"  # Windows integrated auth
            return (
                f"mssql+pyodbc://{auth}{self.mssql_host}:{self.mssql_port}/"
                f"{self.mssql_database}?{params}"
            )
        return ""

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.resolved_database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
