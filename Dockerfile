# SLM Gateway image (CPU). Runs the OpenAI-compatible façade + PaddleOCR +
# embeddings. The LLM itself is served separately (vLLM on GPU, or Ollama),
# reached via UPSTREAM_BASE_URL.
#
# Build lean (no OCR/embeddings):   docker build --build-arg INSTALL_HEAVY=false -t slm-gateway .
# Build full (default):             docker build -t slm-gateway .
FROM python:3.11-slim

ARG INSTALL_HEAVY=true
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# System libs needed by OpenCV / PaddleOCR and PDF rasterization.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements*.txt ./
RUN pip install -r requirements.txt
RUN if [ "$INSTALL_HEAVY" = "true" ]; then \
        pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements-embeddings.txt && \
        pip install -r requirements-ocr.txt ; \
    fi

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
