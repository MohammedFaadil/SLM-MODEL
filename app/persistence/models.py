"""ORM model for the optional request/audit log.

Imported only after SQLAlchemy is confirmed present (see db.py), so importing
this module without SQLAlchemy installed is never attempted.
"""
from __future__ import annotations

import datetime as _dt

from sqlalchemy import DateTime, Integer, String
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
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    client: Mapped[str] = mapped_column(String(128), default="")
