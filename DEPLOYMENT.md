# Deployment runbook — Windows GPU server + Ollama + (optional) MSSQL

Goal: stand up the SLM so it exposes **one URL** you paste into the product in
place of the OpenAI key. The product's existing prompts keep working — it only
talks OpenAI chat-completions, which this serves.

**End state**
- SLM URL:  `http://<server>:8000/v1`  (or `https://slm.yourdomain.com/v1` behind a proxy)
- Key:      the value you put in `GATEWAY_API_KEYS`
- In the product: set OpenAI `base_url` = the SLM URL, `api_key` = that key. Nothing else changes.

---

## 0. Prerequisites (on the GPU server)

| Need | Notes |
|---|---|
| Windows Server / Windows 10-11 | with the NVIDIA GPU |
| NVIDIA driver | recent; `nvidia-smi` must work |
| Python 3.11 (64-bit) | https://www.python.org/downloads/ (tick "Add to PATH") |
| Ollama for Windows | https://ollama.com/download — uses the GPU automatically |
| ODBC Driver 18 for SQL Server | **only if** using MSSQL — https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server |
| NSSM (optional) | to run as a service — https://nssm.cc/download |

---

## 1. Get the code + install dependencies

```powershell
# copy/clone the repo to e.g. C:\slm\SLM-MODEL, then:
cd C:\slm\SLM-MODEL
powershell -ExecutionPolicy Bypass -File scripts\setup_local_windows.ps1 -WithOCR -WithEmbeddings

# add DB support only if you'll use MSSQL:
.\.venv\Scripts\python.exe -m pip install -r requirements-db.txt
```

## 2. Configure Ollama for GPU + pull the model

```powershell
# elevated PowerShell (sets machine env + restarts Ollama)
powershell -ExecutionPolicy Bypass -File scripts\setup_ollama_gpu.ps1 -Model qwen3:8b
nvidia-smi     # confirm an 'ollama' process is on the GPU
```

> **Accuracy vs. speed:** `qwen3:8b` defaults to a Q4 quant (fast). On a GPU with
> spare VRAM, use a higher-precision tag for best accuracy — set `MODEL_NAME` to
> `qwen3:8b-q8_0`, or move up to `qwen3:14b`. Just `ollama pull <tag>` and update `.env`.

## 3. Configure the gateway

```powershell
copy .env.production.example .env
notepad .env
```
Set at minimum:
- `GATEWAY_API_KEYS` → a long random string (this becomes the product's API key)
- `MODEL_NAME` → the tag you pulled
- MSSQL block → only if you want persistence (Section 6)

## 4. Open the firewall + run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\open_firewall.ps1 -Port 8000

# foreground (quick check):
powershell -ExecutionPolicy Bypass -File scripts\run_production.ps1

# OR install as an always-on Windows service (elevated):
powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1 -Port 8000
```

## 5. Verify

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions -H "Authorization: Bearer YOUR_GATEWAY_KEY" `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: ready\"}]}'
```
A normal chat-completions JSON response = you're done.

---

## 6. Integrate with the product (the key swap)

Give the product team **two values**:

| | Value |
|---|---|
| Base URL | `http://<server-ip-or-host>:8000/v1` (or your HTTPS URL from Section 7) |
| API key | your `GATEWAY_API_KEYS` value |

They set these as the OpenAI `base_url` + `api_key`. Their model name (`gpt-4o`,
etc.) is accepted and routed to Qwen automatically (`FORCE_MODEL=true`) — no prompt
or code changes. Streaming, JSON mode, and tool-calls pass through.

---

## 7. (Recommended) Clean HTTPS URL

Serving on a domain with TLS looks more like the old OpenAI endpoint and is safer.

**Easiest — Caddy** (single exe, automatic HTTPS). `Caddyfile`:
```
slm.yourdomain.com {
    reverse_proxy localhost:8000
}
```
Run `caddy run`. URL becomes `https://slm.yourdomain.com/v1`.

**Native — IIS**: install **IIS + Application Request Routing (ARR) + URL Rewrite**,
create a site bound to 443 with your cert, add a reverse-proxy inbound rule to
`http://localhost:8000/`. Keep `GATEWAY_API_KEYS` set so the endpoint stays authed.

---

## 8. MSSQL persistence (optional)

Turns on an audit log + job/candidate/match history. The product does **not** need
this to work — it's for your records.

1. `pip install -r requirements-db.txt` and install **ODBC Driver 18**.
2. In `.env`, set either `DATABASE_URL` or the `MSSQL_*` parts (see
   `.env.production.example`). For a domain service account, leave user/pass blank
   to use Windows Integrated auth.
3. Restart the gateway. On boot it connects and **creates the tables automatically**:
   `slm_request_log`, `slm_job`, `slm_candidate`, `slm_match`.
4. Read history:
   ```
   GET /api/history/jobs        GET /api/history/candidates       GET /api/history/matches
   ```
If the DB is unreachable, the gateway logs a warning and keeps serving statelessly —
persistence never blocks the API.

> When the product team gives you their **actual** MSSQL schema and tells you
> whether the SLM should read from it (vs. keep its own tables), that's a small,
> isolated change in `app/persistence/` — the rest of the service is unaffected.

---

## 9. Efficiency / tuning cheatsheet

| Lever | Where | Effect |
|---|---|---|
| Higher-precision model | `MODEL_NAME=qwen3:8b-q8_0` / `qwen3:14b` | more accuracy (needs VRAM) |
| Keep model warm | `OLLAMA_KEEP_ALIVE=-1` (set by `setup_ollama_gpu.ps1`) | no reload latency |
| Concurrency | `OLLAMA_NUM_PARALLEL` | parallel requests from the product |
| Warmup | `WARMUP_ON_START=true` | first request isn't a cold load |
| **FP8 quantization** (vLLM) | `--quantization fp8` (default in compose/RunPod) | **~2x faster decode + half VRAM** on RTX 4090 / L40S / H100 (Ada/Hopper) |
| **Reasoning off** | `DOMAIN_REASONING=false` (default) | far fewer tokens per answer — the biggest latency win |
| Concurrency (vLLM) | `--max-num-seqs 24` / `MAX_NUM_SEQS` | more candidates processed in parallel |
| Thinking mode | `ENABLE_THINKING=true` / `DOMAIN_REASONING=true` | more thorough but ~2-3x slower |
| GPU OCR | `OCR_USE_GPU=true` + paddlepaddle-gpu | faster OCR for bulk resume volumes |
| Gateway workers | `run_production.ps1 -Workers 2` | more HTTP concurrency (rarely the bottleneck) |

**Fastest config (e.g. RTX 4090):** FP8 + reasoning off are the defaults now, and
the app runs independent LLM calls concurrently (job fields ‖ description; job ‖
resume in the one-shot). Keep `MODEL_NAME=Qwen/Qwen3-8B` and let `--quantization fp8`
do the work; raise `--max-num-seqs` for more concurrent candidates (needs VRAM).

**Fast LOCAL testing (built-in / small GPU via Ollama):** if 8B is slow on a modest
GPU, use the 4B model — `MODEL_NAME=qwen3:4b` (`ollama pull qwen3:4b`) — for quick
iteration, and set `OLLAMA_KEEP_ALIVE=-1` so the model stays in VRAM. Switch back to
`qwen3:8b` / vLLM-FP8 for the real GPU box.

**GPU OCR:** install a CUDA-matched `paddlepaddle-gpu` (see
`requirements-ocr-gpu.txt`), then set `OCR_USE_GPU=true`. Worth it if you process
many resumes; for occasional uploads, CPU OCR (a few seconds each) is simpler and
just as accurate. Bad/missing GPU paddle falls back to CPU automatically.

## 10. Scale up later — vLLM (no app change)

For heavy concurrent load, run **vLLM on a Linux GPU node** and point the gateway
at it — the product URL/key stay identical:
```
# on the Linux GPU box
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384
# in the gateway .env
UPSTREAM_BASE_URL=http://<gpu-host>:8000/v1
MODEL_NAME=Qwen/Qwen3-8B
```
Or use the provided `docker-compose.yml` (vLLM + gateway) on a Linux GPU host.

## 10b. RunPod (GPU cloud hosting)

Two ways; both give you a public `…/v1` URL to hand the product.

### A) One container (simplest) — `Dockerfile.runpod`
This image runs **vLLM + gateway together**. Build once, push to a registry, deploy as a RunPod Pod.
```bash
docker build -f Dockerfile.runpod -t <you>/slm-runpod:latest .
docker push <you>/slm-runpod:latest
```
On RunPod:
1. **Deploy → Pods → GPU** (24 GB is plenty for Qwen3-8B: RTX 4090 / A5000 / L4).
2. **Container image** = `<you>/slm-runpod:latest`.
3. **Expose HTTP Ports** = `8080`.
4. **Environment variables:**
   - `GATEWAY_API_KEYS` = a strong key (the product's API key)
   - `MODEL_NAME` = `Qwen/Qwen3-8B` (or `Qwen/Qwen3-14B` on a bigger GPU)
   - `HUGGING_FACE_HUB_TOKEN` = your HF token (avoids gated/download limits)
   - optional: `OCR_USE_GPU=true` (only if you added paddlepaddle-gpu to the image)
5. (Recommended) attach a **Network Volume** mounted at `/root/.cache/huggingface`
   so the model isn't re-downloaded on every restart.
6. Deploy. The model download + load takes a few minutes (watch the pod logs for
   `[runpod_start] vLLM ready` then `Uvicorn running`).

Your SLM URL = the pod's HTTP proxy for port 8080:
```
https://<POD_ID>-8080.proxy.runpod.net/v1
```
Give the product **that URL** + the `GATEWAY_API_KEYS` value. Done.

### B) Two pods / bring-your-own vLLM
Run RunPod's official **vLLM template** (serves `/v1` on a port) and point a small
gateway (this repo, CPU is fine) at it:
```
UPSTREAM_BASE_URL=https://<vllm-pod-id>-8000.proxy.runpod.net/v1
MODEL_NAME=Qwen/Qwen3-8B
```
Expose the gateway's port; its proxy URL is what the product uses. Keep
`GATEWAY_API_KEYS` set so only the gateway is public-facing with auth.

> **Local testing on your 2nd GPU box** uses the exact same code: either
> `docker compose up -d --build` (vLLM+gateway) on a Linux GPU box, or Ollama-GPU
> (`scripts\setup_ollama_gpu.ps1`) on Windows. Validate there first, then deploy the
> same image/compose to RunPod — no code changes.

## 11. Operations

- **Logs:** `logs\gateway.out.log` / `logs\gateway.err.log` (service), or console.
- **Restart:** `nssm restart SLMGateway`  ·  **Stop:** `nssm stop SLMGateway`
- **Update model:** `ollama pull <tag>`, set `MODEL_NAME`, restart the service.
- **Health probe for a load balancer:** `GET /health` (200 = model reachable).

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `backend_unreachable` in responses | Ollama not running / wrong `UPSTREAM_BASE_URL`. `ollama ps`. |
| First request very slow | Cold model load; ensure `WARMUP_ON_START=true` + `OLLAMA_KEEP_ALIVE=-1`. |
| 401 from the gateway | Product isn't sending a key in `GATEWAY_API_KEYS`. |
| OCR errors | `pip install -r requirements-ocr.txt`; first run downloads OCR models. |
| MSSQL warning at startup | Driver missing or bad connection string; service still runs statelessly. |
| Product rejects responses | Confirm it points at `/v1` (not the bare host) and uses chat-completions. |
