# SLM Gateway — Qwen3, OpenAI-compatible (drop-in model URL + OCR)

A self-hosted **Small Language Model** service your product adopts by **changing one
thing: the OpenAI `base_url`** (and the API key). No code changes on the product side.

The gateway is a **faithful transparent proxy** in front of **Qwen3** (served by vLLM
on GPU, or Ollama). **The product supplies all prompts** — the gateway keeps none. It
forwards the product's prompts and params (streaming, JSON mode, tools, temperature,
`chat_template_kwargs`, ...) to the model untouched and never overrides sampling, so
accuracy is entirely the product's choice. It also exposes **PaddleOCR (PP-OCRv5)** for
turning resume PDFs/images into clean text.

```
Your product ──(swap base_url + key)──►  SLM Gateway  ──►  vLLM/Ollama (Qwen3)
                                            /v1/chat/completions   (the product's prompts)
                                            /v1/completions
                                            /v1/embeddings
                                            /v1/models
                                            /api/ocr/parse         (PDF/image -> text)
```

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
- Streaming, `response_format` (json_object/json_schema), tools/function-calling, `seed`,
  `logprobs`, multimodal image parts, `stream_options` — all pass through faithfully.
- Upstream errors are returned with the real HTTP status + OpenAI-shaped error body.

**The two values you hand the product:** `base_url = http://<host>:8080/v1` and `api_key = <one of GATEWAY_API_KEYS>`.

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | chat — streaming + non-streaming, tools, JSON mode |
| POST | `/v1/completions` | legacy text completion (streaming + non-streaming) |
| POST | `/v1/embeddings` | embeddings (local `bge-small`, or upstream) |
| GET  | `/v1/models` | advertises `slm-qwen3-8b` |
| POST | `/api/ocr/parse` | multipart file → `{text, num_pages, status, overall_confidence, method_summary, pages[]}` |
| GET  | `/health` | backend + version |

Interactive schema at `/docs`.

---

## 3. Quick start

**Instant plumbing test — no model, no GPU:**
```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe scripts/smoke_test.py      # 9 checks, offline (mock backend)
```

**Real model:** run vLLM (GPU) or Ollama, then point the gateway at it:
```powershell
# gateway
copy .env.example .env
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```
See [DEPLOYMENT.md](DEPLOYMENT.md) for the full GPU/RunPod/Windows runbook.

**Minimal test UI** (chat + OCR, dev tool only): `ui/` — see [ui/README.md](ui/README.md).

---

## 4. Accuracy-first serving (Qwen3-8B)

The gateway is precision-agnostic — swap the model/precision purely via vLLM flags and
`UPSTREAM_BASE_URL`; **no gateway code changes**. Reasoning ("thinking") is the product's
call (it sends `chat_template_kwargs={"enable_thinking":...}`, forwarded verbatim).

| Target | Recommended | Why |
|---|---|---|
| **A100 80GB (max accuracy — default)** | `Qwen/Qwen3-8B` **bf16**, `--max-model-len 32768 --max-num-seqs 128 --enable-prefix-caching` | reference quality |
| **RTX 4090 24GB (accuracy+speed)** | `Qwen/Qwen3-8B-FP8` (served as `Qwen/Qwen3-8B`), `--max-model-len 16384 --max-num-seqs 32` | ~2× on Ada, ~same accuracy, frees VRAM |
| **Windows / Ollama** | `qwen3:8b-q8_0` (near-fp16), `OLLAMA_KEEP_ALIVE=-1`, raise `num_ctx` | best practical local accuracy |

**Step up for more accuracy** (bigger@FP8 beats smaller@bf16 on hard tasks): **Qwen3-14B-FP8**
(~14 GB, a 48 GB L40S/A6000) → **Qwen3-32B-FP8** (~33 GB, single H100/A100 80GB). Keep the
gateway's `MODEL_NAME`/`SERVED_MODEL_ID` stable via `--served-model-name`.

Example (A100 max accuracy):
```bash
vllm serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 --kv-cache-dtype auto --enable-prefix-caching --port 8000
```

---

## 5. OCR

`POST /api/ocr/parse` returns exact native text for digital PDFs and falls back to
**PaddleOCR PP-OCRv5** for scanned pages (per-page method + confidence in the response).
Install `requirements-ocr.txt` (CPU) or `requirements-ocr-gpu.txt` + `OCR_USE_GPU=true`
(GPU). Encrypted PDFs → `422`; the page count is capped by `OCR_MAX_PAGES`.

---

## 6. Optional MSSQL (audit log)

Set `DATABASE_URL` (or `MSSQL_*`) to record a request/audit log (`slm_request_log`). It's a
**no-op until configured** and a DB hiccup never blocks the API. See `.env.production.example`.

---

## 7. Configuration (key vars)

| Var | Default | Meaning |
|---|---|---|
| `UPSTREAM_BASE_URL` | `http://localhost:11434/v1` | vLLM/Ollama URL |
| `MODEL_NAME` | `qwen3:8b` | served model id (== `--served-model-name`) |
| `SERVED_MODEL_ID` | `slm-qwen3-8b` | id advertised to the product |
| `FORCE_MODEL` | `true` | route any requested model to `MODEL_NAME` |
| `GATEWAY_API_KEYS` | *(empty)* | accepted keys; empty = open (dev only) |
| `EMBEDDINGS_MODE` | `local` | `local` / `upstream` / `off` |
| `OCR_VERSION` / `OCR_DPI` / `OCR_USE_GPU` | `v5` / `300` / `false` | OCR config |
| `DATABASE_URL` / `MSSQL_*` | *(empty)* | optional audit log |

Full annotated list in [.env.example](.env.example).

---

## 8. Project structure

```
app/
  main.py            FastAPI app (gateway + OCR route + health, error passthrough)
  config.py          env-driven settings
  openai_api/        OpenAI-compatible schemas + router (the drop-in surface)
  backends/          openai_upstream proxy (vLLM/Ollama) + mock + factory
  ocr/               PaddleOCR PP-OCRv5 + PDF pipeline (native-text + OCR fallback)
  embeddings/        sentence-transformers embedder (/v1/embeddings)
  routes/ocr_routes  POST /api/ocr/parse
  persistence/       optional MSSQL request/audit log
scripts/             run / setup / model-pull / runpod_start / smoke_test
ui/                  minimal Streamlit test UI (chat + OCR) — dev tool
tests/               offline pytest (OpenAI compatibility)
Dockerfile, Dockerfile.runpod, docker-compose*.yml   deployment
```

---

## 9. Testing
```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe scripts/smoke_test.py
```
