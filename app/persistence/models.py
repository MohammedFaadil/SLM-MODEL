"""ORM models for the optional audit/history store.

Imported only after SQLAlchemy is confirmed present (see db.py), so importing
this module without SQLAlchemy installed is never attempted.
"""
from __future__ import annotations

import datetime as _dt

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class RequestLog(Base):
    __tablename__ = "slm_request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    path: Mapped[str] = mapped_column(String(256))
    method: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(128), default="")
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    streamed: Mapped[bool] = mapped_column(Boolean, default=False)
    client: Mapped[str] = mapped_column(String(128), default="")


class JobRecord(Base):
    __tablename__ = "slm_job"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    seniority: Mapped[str] = mapped_column(String(64), default="")
    min_years_experience: Mapped[float] = mapped_column(Float, default=0.0)
    spec_json: Mapped[str] = mapped_column(Text, default="{}")


class CandidateRecord(Base):
    __tablename__ = "slm_candidate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    email: Mapped[str] = mapped_column(String(256), default="")
    total_years_experience: Mapped[float] = mapped_column(Float, default=0.0)
    profile_json: Mapped[str] = mapped_column(Text, default="{}")


class MatchRecord(Base):
    __tablename__ = "slm_match"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    job_title: Mapped[str] = mapped_column(String(256), default="")
    candidate_name: Mapped[str] = mapped_column(String(256), default="")
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    verdict: Mapped[str] = mapped_column(String(64), default="")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
