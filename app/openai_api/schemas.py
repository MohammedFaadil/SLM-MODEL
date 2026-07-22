"""OpenAI-compatible request/response models.

These mirror the shapes at api.openai.com/v1 closely enough that an existing
OpenAI client (Python `openai`, LangChain, raw HTTP, etc.) works unchanged.
They are deliberately permissive: unknown fields are accepted and forwarded so
we never reject a request your platform already sends to OpenAI today.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _Permissive(BaseModel):
    # Accept and retain fields we don't explicitly model.
    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- #
#  Chat completions
# --------------------------------------------------------------------------- #
class ChatMessage(_Permissive):
    role: str
    # `content` may be a plain string or the OpenAI "parts" array (text/image).
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(_Permissive):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None


class CompletionRequest(_Permissive):
    """Legacy /v1/completions (some SDKs still use it)."""

    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
#  Embeddings
# --------------------------------------------------------------------------- #
class EmbeddingRequest(_Permissive):
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# --------------------------------------------------------------------------- #
#  Models listing
# --------------------------------------------------------------------------- #
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "slm-gateway"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
