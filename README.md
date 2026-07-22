# SLM Gateway — Qwen3-8B, OpenAI-compatible (recruiting / resume matching)

A self-hosted **Small Language Model** service that your existing platform can
adopt by **changing one thing: the OpenAI `base_url`** (and the API key). No code
changes on the platform side.

It wraps **Qwen3-8B** behind the **OpenAI API** (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/models`) and adds a recruiting pipeline:

- **PaddleOCR (PP-OCRv4/v5)** resume parsing (PDF/image → text, native-text fast path)
- **Candidate summaries / justifications** from resume text
- **Job ↔ candidate matching** with matched/missing skills + an AI justification

> Designed to run **now on a CPU-only box** (via Ollama) for accuracy testing, and
> to move to **cloud GPU** (via vLLM) later with **zero application changes** —
> only environment variables differ.

---

## 1. The integration contract (why "just swap the key" works)

Your platform today probably does something like:

```python
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
client.chat.completions.create(model="gpt-4o", messages=[...])
```

Point it at this gateway instead:

```python
client = OpenAI(
    api_key=GATEWAY_API_KEY,             # any key you set in GATEWAY_API_KEYS
    base_url="http://YOUR_SLM_HOST:8000/v1",
)
client.chat.completions.create(model="gpt-4o", messages=[...])   # unchanged
```

- The gateway **accepts any `model` name** (even `"gpt-4o"`) and routes it to
  Qwen3-8B (`FORCE_MODEL=true`). Your hardcoded model string keeps working.
- **Streaming, JSON mode (`response_format`), tools/function-calling** pass
  through to the engine.
- The **`base_url` value is the "url" you replace the key with.** In the cloud,
  `GATEWAY_API_KEYS` is the credential you rotate.

---

## 2. Architecture

```
                        ┌─────────────────────────── SLM Gateway (this repo) ───────────────────────────┐
  Your platform ──────► │  /v1/chat/completions  /v1/embeddings  /v1/models   (OpenAI-compatible)       │
   (swap base_url)      │  /api/ocr  /api/resume  /api/jobs  /api/match         (recruiting domain)      │
                        │      │                 │                    │                                  │
                        │  PaddleOCR         sentence-transformers  deterministic match + LLM justify    │
                        └──────┼─────────────────┼────────────────────┼──────────────────────────────────┘
                               │                 │            proxies chat/embeddings to ▼
                               ▼                 ▼        ┌──────────────────────────────────────┐
                          PDF→text           bge-small    │  Inference engine (OpenAI-compatible) │
                                                          │  LOCAL: Ollama (CPU)  qwen3:8b        │
                                                          │  CLOUD: vLLM  (GPU)   Qwen/Qwen3-8B   │
                                                          └──────────────────────────────────────┘
```

Same gateway code for both; only `UPSTREAM_BASE_URL` + `MODEL_NAME` change.

---

## 3. Quick start

### 3a. Instant plumbing test — no model, no GPU

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe scripts/smoke_test.py        # 13 checks, all offline
```

Or run the server in **mock** mode and browse `http://localhost:8000/docs`:

```powershell
$env:LLM_BACKEND="mock"; ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

### 3b. Real model on your CPU box (Ollama + Qwen3-8B)

```powershell
# 1) Install Ollama:  https://ollama.com/download
# 2) One-shot setup (venv + deps + pull model). Add -WithOCR -WithEmbeddings for the full pipeline:
powershell -ExecutionPolicy Bypass -File scripts\setup_local_windows.ps1 -WithOCR -WithEmbeddings

# 3) Run the gateway
powershell -ExecutionPolicy Bypass -File scripts\run_gateway.ps1
```

Then verify:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" `
  -d '{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words\"}]}'
```

> **Speed note for your i5-6600T / 16 GB / no-GPU box:** `qwen3:8b` (Q4) fits in
> RAM and runs correctly but slowly (a few tokens/sec). That's expected — it's for
> **accuracy validation**. For snappier local iteration use `qwen3:4b`
> (`MODEL_NAME=qwen3:4b`). On your bigger GPU machine / cloud you get full speed
> with no code change.

### 3c. The test UI (disposable)

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-ui.txt
./.venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
```

Three tabs: **AI Job Creation → Add Candidate (resume upload) → Match**. The `ui/`
folder is git-ignored so it never reaches the main product.

---

## 4. Endpoints

### OpenAI-compatible (what the platform uses)
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | streaming + non-streaming, JSON mode, tools |
| POST | `/v1/completions` | legacy text completion |
| POST | `/v1/embeddings` | local `bge-small` (or upstream) |
| GET  | `/v1/models` | advertises `slm-qwen3-8b` |

### Recruiting domain (used by the UI; also callable by the platform)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/ocr/parse` | multipart `file` | extracted text + per-page method |
| POST | `/api/resume/parse` | multipart `file` | `CandidateProfile` (OCR + LLM) |
| POST | `/api/resume/parse-text` | `{"text": "..."}` | `CandidateProfile` |
| POST | `/api/jobs` | `JobCreateRequest` | enriched `JobSpec` |
| POST | `/api/match` | `{job, candidate}` | `MatchResult` (score + justification) |
| POST | `/api/match/upload` | multipart (file + job fields) | one-shot `MatchResult` |
| GET  | `/health` | — | backend + feature status |

Full interactive schema at `/docs`.

---

## 5. The three workflows

1. **AI Job Creation** — `POST /api/jobs` with `{title, skills[], min_years_experience, seniority}`.
   The SLM expands it into required/preferred skills, responsibilities,
   qualifications, and a JD. Your explicit inputs always win over the model.
2. **Add Candidate** — `POST /api/resume/parse` with a resume PDF. Native text is
   extracted first; scanned pages fall back to **PaddleOCR**. The SLM returns a
   structured profile **and** a recruiter-facing summary/justification.
3. **Job ↔ Candidate Match** — `POST /api/match`. Skill overlap is computed
   **deterministically** (semantic via embeddings, fuzzy fallback), producing a
   stable 0–100 score, matched/missing/additional skills, and experience fit. The
   SLM then writes a justification **grounded only in those numbers** — so the text
   never contradicts the score.

---

## 6. Cloud GPU deployment (vLLM)

On a GPU host with Docker + NVIDIA Container Toolkit:

```bash
export GATEWAY_API_KEYS="sk-your-strong-key"     # your platform sends this
docker compose up -d --build                     # vLLM (GPU) + gateway
# platform base_url ->  http://<host>:8080/v1
```

CPU-only server instead? Use Ollama:

```bash
docker compose -f docker-compose.cpu.yml up -d --build
docker compose -f docker-compose.cpu.yml exec ollama ollama pull qwen3:8b
```

Or run vLLM yourself and just point the gateway at it:

```bash
# on the GPU box
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384
# gateway env
UPSTREAM_BASE_URL=http://<gpu-host>:8000/v1   MODEL_NAME=Qwen/Qwen3-8B
```

---

## 7. Configuration (env / `.env`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `openai_upstream` | `openai_upstream` or `mock` |
| `UPSTREAM_BASE_URL` | `http://localhost:11434/v1` | Ollama locally / vLLM in cloud |
| `MODEL_NAME` | `qwen3:8b` | engine model id (`Qwen/Qwen3-8B` for vLLM) |
| `SERVED_MODEL_ID` | `slm-qwen3-8b` | id advertised to the platform |
| `FORCE_MODEL` | `true` | ignore the platform's model name, route to `MODEL_NAME` |
| `ENABLE_THINKING` | `false` | Qwen3 thinking (off = faster, clean JSON on CPU) |
| `GATEWAY_API_KEYS` | *(empty)* | comma-separated accepted keys; empty = open |
| `EMBEDDINGS_MODE` | `local` | `local` / `upstream` / `off` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | local embedding model |
| `SEMANTIC_MATCH_THRESHOLD` | `0.62` | cosine ≥ this = same skill |
| `OCR_ENABLED` | `true` | enable PaddleOCR fallback |
| `OCR_VERSION` | `v5` | `v5` (PP-OCRv5) or `v4` |
| `OCR_PREFER_NATIVE_TEXT` | `true` | try fast native text before OCR |
| `REQUEST_TIMEOUT` | `600` | upstream timeout (s) — keep high for CPU |

See [.env.example](.env.example) for the annotated full list.

---

## 8. Accuracy & tuning

- **Matching is deterministic**, not model-guessed — the score is reproducible and
  the LLM only explains it. This is the single biggest accuracy win.
- **Semantic skills:** install embeddings (`requirements-embeddings.txt`) so
  "PyTorch" can satisfy "deep learning frameworks". Without it, matching falls back
  to fuzzy string matching automatically.
- **Bigger machine = better model, no code change:** bump `MODEL_NAME` to
  `qwen3:14b` (Ollama) or keep `Qwen/Qwen3-8B` at full precision on vLLM.
- **Robust to bad model output:** every LLM step has a heuristic fallback (regex
  contacts, lexicon skills, deterministic scoring), so an endpoint never hard-fails
  on malformed JSON.

---

## 9. Testing

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest          # offline unit + API tests
./.venv/Scripts/python.exe scripts/smoke_test.py
```

---

## 10. Project structure

```
app/
  main.py               FastAPI app (routers, health, errors, CORS)
  config.py             env-driven settings
  openai_api/           OpenAI-compatible schemas + router (the drop-in surface)
  backends/             openai_upstream proxy (Ollama/vLLM) + mock + factory
  ocr/                  PaddleOCR wrapper + PDF pipeline (native-text + OCR)
  embeddings/           sentence-transformers embedder
  domain/               prompts, resume/job/match services, skills engine
  routes/               /api/* recruiting endpoints
scripts/                setup / run / pull-model / smoke_test
ui/                     DISPOSABLE Streamlit test UI (git-ignored)
tests/                  offline pytest suite
Dockerfile, docker-compose*.yml    cloud (vLLM/GPU) + CPU (Ollama)
```

---

## 11. What to hand to the platform team

1. Deploy this gateway (Section 6).
2. Give them **`base_url = http://<host>:8080/v1`** and **one `GATEWAY_API_KEYS` value**.
3. They replace their OpenAI `base_url` + key. Done — no other changes.
