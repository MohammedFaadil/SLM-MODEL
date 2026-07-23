#!/usr/bin/env bash
# Single-container entrypoint for RunPod (or any Linux GPU host):
# starts vLLM (GPU) in the background, waits until it's ready, then starts the
# gateway in the foreground. Expose $PORT (default 8080) -> RunPod gives you a
# public URL like https://<podid>-8080.proxy.runpod.net ; the product uses
#   https://<podid>-8080.proxy.runpod.net/v1
set -euo pipefail

MODEL="${MODEL_NAME:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Speed knobs. FP8 ~doubles throughput and halves VRAM on Ada/Hopper (RTX 4090,
# L40S, H100). Set QUANTIZATION=none for full FP16, or =awq for an Ampere card
# (A100/A40) with a pre-quantized AWQ model id in MODEL_NAME.
QUANTIZATION="${QUANTIZATION:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
QUANT_ARG=""
if [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "none" ]; then
    QUANT_ARG="--quantization ${QUANTIZATION}"
fi

echo "[runpod_start] launching vLLM for ${MODEL} on :${VLLM_PORT} (quant=${QUANTIZATION}, max_seqs=${MAX_NUM_SEQS})"
vllm serve "${MODEL}" \
    --served-model-name "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    ${QUANT_ARG} \
    ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

# Ensure the gateway talks to the local vLLM.
export LLM_BACKEND="${LLM_BACKEND:-openai_upstream}"
export UPSTREAM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export MODEL_NAME="${MODEL}"

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
