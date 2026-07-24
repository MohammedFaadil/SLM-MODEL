"""Best-effort request/audit logging.

`save_request_log` is a no-op when the DB is disabled and swallows its own errors,
so a database hiccup can never break an API response. Call it via
`fastapi.concurrency.run_in_threadpool` since the driver is synchronous.
"""
from __future__ import annotations

from typing import Any

from ..logging_conf import get_logger
from . import db

log = get_logger(__name__)


def save_request_log(**kw: Any) -> None:
    if not db.available():
        return
    try:
        from .models import RequestLog

        with db.session() as s:
            s.add(RequestLog(**kw))
    except Exception as exc:  # noqa: BLE001
        log.debug("request-log write skipped: %s", exc)
