#!/usr/bin/env bash
# Single-container entrypoint for RunPod (or any Linux GPU host):
# starts vLLM (GPU) in the background, waits until it's ready, then starts the
# gateway in the foreground. Expose $PORT (default 8080) -> RunPod gives you a
# public URL like https://<podid>-8080.proxy.runpod.net ; the product uses
#   https://<podid>-8080.proxy.runpod.net/v1
set -euo pipefail

# MODEL = the checkpoint vLLM loads (may be Qwen/Qwen3-8B, ...-FP8, ...-AWQ, 14B, ...).
# SERVED_ALIAS = the STABLE public id; keep it constant so the gateway's MODEL_NAME
# never changes when you swap precision/size. Defaults to a fixed alias, not MODEL.
MODEL="${MODEL_NAME:-Qwen/Qwen3-8B}"
SERVED_ALIAS="${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Accuracy-first extra args (bf16 + prefix caching). For a 4090 set MODEL_NAME to
# Qwen/Qwen3-8B-FP8; for A100 raise --max-num-seqs 128 and MAX_MODEL_LEN 32768.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---dtype bfloat16 --max-num-seqs 32 --kv-cache-dtype auto --enable-prefix-caching}"

echo "[runpod_start] launching vLLM: model=${MODEL} served-as=${SERVED_ALIAS} on :${VLLM_PORT}"
vllm serve "${MODEL}" \
    --served-model-name "${SERVED_ALIAS}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    ${VLLM_EXTRA_ARGS} &
VLLM_PID=$!

# Ensure the gateway talks to the local vLLM using the STABLE alias.
export LLM_BACKEND="${LLM_BACKEND:-openai_upstream}"
export UPSTREAM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export MODEL_NAME="${SERVED_ALIAS}"

echo "[runpod_start] waiting for vLLM to load the model..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[runpod_start] vLLM ready."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[runpod_start] vLLM exited during startup." >&2
        exit 1
    fi
    sleep 5
done

echo "[runpod_start] starting gateway on :${GATEWAY_PORT}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${GATEWAY_PORT}"
