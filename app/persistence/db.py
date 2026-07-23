"""Lazy, optional database engine.

Persistence is OFF unless a connection is configured (DATABASE_URL or MSSQL_*).
If SQLAlchemy / the ODBC driver isn't installed, we log once and stay disabled —
the API keeps working statelessly. Nothing here is on the hot path unless enabled.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..config import settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_state: dict = {"engine": None, "Session": None, "available": False, "checked": False}


def init() -> bool:
    """Initialize the engine + create tables. Safe to call once at startup."""
    if _state["checked"]:
        return _state["available"]
    _state["checked"] = True

    if not settings.persistence_enabled:
        log.info("Persistence disabled (no DATABASE_URL / MSSQL_* configured).")
        return False

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from .models import Base
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Persistence requested but unavailable (%s). "
            "Install requirements-db.txt + ODBC Driver 18. Continuing without DB.",
            exc,
        )
        return False

    try:
        url = settings.resolved_database_url
        kwargs: dict = {"pool_pre_ping": True, "pool_recycle": 1800}
        # fast_executemany is a pyodbc-only optimization; only pass it for MSSQL.
        if url.startswith("mssql+pyodbc"):
            kwargs["fast_executemany"] = True
        engine = create_engine(url, **kwargs)
        Base.metadata.create_all(engine)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not connect to MSSQL (%s). Continuing without DB.", exc)
        return False

    _state["engine"] = engine
    _state["Session"] = sessionmaker(bind=engine, expire_on_commit=False)
    _state["available"] = True
    log.info("Persistence enabled -> database connected (%s), tables ensured.",
             engine.dialect.name)
    return True


def available() -> bool:
    return bool(_state["available"])


@contextmanager
def session() -> Iterator[Any]:
    Session = _state["Session"]
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def dispose() -> None:
    engine = _state.get("engine")
    if engine is not None:
        engine.dispose()
