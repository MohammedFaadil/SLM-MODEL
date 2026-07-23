#!/usr/bin/env bash
# Start the SLM gateway (Linux/macOS / cloud).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PY="${PY:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

echo "Gateway -> http://${HOST}:${PORT}  (OpenAI base_url = http://<host>:${PORT}/v1)"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
