"""Force an offline, model-free configuration for the whole test suite."""
import os

os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDINGS_MODE", "off")
os.environ.setdefault("OCR_ENABLED", "false")
os.environ.setdefault("GATEWAY_API_KEYS", "")
