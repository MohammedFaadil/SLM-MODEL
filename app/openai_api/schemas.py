"""OpenAI-compatible response models used by the router.

The gateway is a transparent proxy: chat/completion/embedding requests are read as
raw dicts and forwarded verbatim (so no field is ever dropped by a strict schema).
Only the /v1/models listing is modelled here.
"""
from __future__ import annotations

import time
from typing import List

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "slm-gateway"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
