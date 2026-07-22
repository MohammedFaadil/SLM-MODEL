#!/usr/bin/env bash
# Pull the model into Ollama (Linux/macOS).
set -euo pipefail
MODEL="${1:-qwen3:8b}"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not installed. Get it from https://ollama.com/download" >&2
  exit 1
fi
echo "Pulling $MODEL ..."
ollama pull "$MODEL"
ollama list
