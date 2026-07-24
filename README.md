# SLM Gateway — Qwen3, OpenAI-compatible (drop-in model URL + OCR)

A self-hosted **Small Language Model** service your product adopts by changing **one
thing: the OpenAI `base_url`** (and the API key). No code changes on the product side.

It is a **faithful transparent proxy** in front of **Qwen3** (served by vLLM on GPU, or
Ollama). **The product supplies all prompts** — the gateway keeps none. It forwards the
product's prompts and params (streaming, JSON mode, tools, temperature,
`chat_template_kwargs`, …) untouched and never overrides sampling, so accuracy is entirely
the product's choice. It also exposes **PaddleOCR (PP-OCRv5)** for turning resume
PDFs/images into clean text.

```text
Main product
   |   (base_url + api key)
   v
SLM Gateway  -->  vLLM / Ollama (Qwen3)
   /v1/chat/completions      <- the product's prompts
   /v1/completions
   /v1/embeddings
   /v1/models
   /api/ocr/parse            <- PDF / image -> text
```

> **Deploying to production?** Follow **[HOSTING_GUIDE.md](HOSTING_GUIDE.md)** — a step-by-step
> runbook for **Windows host + RunPod GPU + MSSQL**, including how to swap the model size.

---

## 1. Integration — the key swap

Today the product calls OpenAI:

```python
client = OpenAI(api_key=OPENAI_KEY, base_url="https://api.openai.com/v1")
client.chat.completions.create(model="gpt-4o", messages=[...])   # its own prompts
```

Point it at the gateway instead — nothing else changes:

```python
client = OpenAI(api_key=GATEWAY_KEY, base_url="http://YOUR_SLM_HOST:8080/v1")
client.chat.completions.create(model="gpt-4o", messages=[...])   # unchanged
```

- Any `model` name (even `gpt-4o`) is routed to the served Qwen model (`FORCE_MODEL=true`).
- Streaming, `response_format`, tools/function-calling, `seed`, `logprobs`, multimodal image
  parts, and `stream_options` all pass through faithfully.
- Upstream errors return the real HTTP status + an OpenAI-shaped error body.

**The two values you hand the product:**

- `base_url` = `http://<host>:8080/v1`
- `api_key` = one of your `GATEWAY_API_KEYS`

---

## 2. Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/chat/completions` | chat (streaming + non-streaming, tools, JSON mode) |
| POST | `/v1/completions` | legacy text completion (streaming + non-streaming) |
| POST | `/v1/embeddings` | embeddings (local `bge-small`, or upstream) |
| GET | `/v1/models` | advertises the served model id |
| POST | `/api/ocr/parse` | multipart file -> extracted text (+ per-page method) |
| GET | `/health` | backend status + version |

Interactive schema at `/docs`.

---

## 3. Quick start

Instant plumbing test — no model, no GPU (uses the mock backend):

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe scripts/smoke_test.py
```

Real model — run vLLM (GPU) or Ollama, then point the gateway at it:

```powershell
copy .env.example .env
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Minimal test UI (chat + OCR, dev tool only): see [ui/README.md](ui/README.md).

---

## 4. Serving the model (accuracy scales with the GPU)

The gateway is precision-agnostic — swap model / precision purely via vLLM flags and
`UPSTREAM_BASE_URL`, with **no gateway code changes**. Reasoning ("thinking") is the
product's call — it sends `chat_template_kwargs`, which the gateway forwards.

| GPU | Model | Notes |
| --- | --- | --- |
| A100 80GB (default) | `Qwen/Qwen3-8B` bf16 | reference / max accuracy |
| RTX 4090 24GB | `Qwen/Qwen3-8B-FP8` | ~2x faster on Ada, ~same accuracy |
| Windows / Ollama | `qwen3:8b-q8_0` | near-fp16, best practical local |

Example (A100, max accuracy):

```bash
vllm serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 --kv-cache-dtype auto --enable-prefix-caching --port 8000
```

**More accuracy** (bigger @ FP8 beats smaller @ bf16 on hard tasks): step up to
`Qwen/Qwen3-14B-FP8` (~14 GB, a 48 GB card) or `Qwen/Qwen3-32B-FP8` (~33 GB, an 80 GB card).
This is an env change only — see [HOSTING_GUIDE.md](HOSTING_GUIDE.md) § "Change the model".

---

## 5. OCR

`POST /api/ocr/parse` returns exact native text for digital PDFs and falls back to
**PaddleOCR PP-OCRv5** for scanned pages (per-page method + confidence in the response).
Install `requirements-ocr.txt` (CPU) or `requirements-ocr-gpu.txt` + `OCR_USE_GPU=true`
(GPU). Encrypted PDFs return `422`; the page count is capped by `OCR_MAX_PAGES`.

---

## 6. Optional MSSQL (audit log)

Set `DATABASE_URL` (or the `MSSQL_*` parts) to record a request/audit log
(`slm_request_log`). It is a **no-op until configured**, and a DB hiccup never blocks the
API. See `.env.production.example`.

---

## 7. Configuration (key vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `http://localhost:11434/v1` | vLLM / Ollama URL |
| `UPSTREAM_API_KEY` | `ollama` | key the gateway sends upstream (e.g. vLLM `--api-key`) |
| `MODEL_NAME` | `qwen3:8b` | served model id (matches vLLM `--served-model-name`) |
| `SERVED_MODEL_ID` | `slm-qwen3-8b` | id advertised to the product |
| `FORCE_MODEL` | `true` | route any requested model to `MODEL_NAME` |
| `GATEWAY_API_KEYS` | *(empty)* | accepted keys; empty = open (dev only) |
| `EMBEDDINGS_MODE` | `local` | `local` / `upstream` / `off` |
| `OCR_VERSION` / `OCR_DPI` / `OCR_USE_GPU` | `v5` / `300` / `false` | OCR config |
| `DATABASE_URL` / `MSSQL_*` | *(empty)* | optional audit log |

Full annotated list in [.env.example](.env.example).

---

## 8. Project structure

```text
app/
  main.py            FastAPI app (gateway + OCR route + health, error passthrough)
  config.py          env-driven settings
  openai_api/        OpenAI-compatible schemas + router (the drop-in surface)
  backends/          openai_upstream proxy (vLLM/Ollama) + mock + factory
  ocr/               PaddleOCR PP-OCRv5 + PDF pipeline (native-text + OCR fallback)
  embeddings/        sentence-transformers embedder (/v1/embeddings)
  routes/            POST /api/ocr/parse
  persistence/       optional MSSQL request/audit log
scripts/             run / setup / model-pull / runpod_start / smoke_test
ui/                  minimal Streamlit test UI (chat + OCR) - dev tool
tests/               offline pytest (OpenAI compatibility)
```

Deployment: `Dockerfile`, `Dockerfile.runpod`, `docker-compose.yml`, `docker-compose.cpu.yml`.

---

## 9. Testing

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe scripts/smoke_test.py
```
