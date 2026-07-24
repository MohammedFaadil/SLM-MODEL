# Hosting Guide — Windows host + RunPod GPU + MSSQL

Follow this to stand up the SLM and get a **stable URL** you paste into the main product in
place of the OpenAI API key. No code changes are needed anywhere — only the steps below.

## What you will end up with

```text
Main product
   |   base_url = https://slm.yourdomain.com/v1     (or http://<windows-ip>:8080/v1)
   |   api_key  = <one of GATEWAY_API_KEYS>
   v
Windows live server:  SLM Gateway  (stable URL, HTTPS, OCR, MSSQL audit log)
   |   UPSTREAM_BASE_URL + UPSTREAM_API_KEY
   v
RunPod GPU pod:  vLLM  (serves Qwen3, GPU)
```

- **RunPod** does the GPU work (vLLM serves the model).
- **The Windows live server** runs the gateway and owns the **public URL** the product uses.
  Because the URL lives on your Windows server, it stays the same even if the RunPod pod is
  recreated (you'd just update one env value on Windows).
- **MSSQL** stores an optional request/audit log.

The product only ever sees the Windows gateway URL + key. Everything else is internal.

---

## Part 1 — RunPod: serve the model on a GPU

1. **Create a GPU Pod.** RunPod → Deploy → Pods → pick a GPU:
   - **A100 80GB** — max accuracy (bf16), or **RTX 4090 24GB** — cheaper, use the FP8 model.
   - Container image: `vllm/vllm-openai:latest`.
2. **Expose HTTP port `8000`.**
3. **Environment variables** on the pod:
   - `HUGGING_FACE_HUB_TOKEN` = your Hugging Face token (avoids download limits).
4. **Container start command / arguments** (this is the vLLM serve line). Pick one:

   **A100 (bf16, max accuracy):**
   ```
   --model Qwen/Qwen3-8B --served-model-name production-slm --api-key CHANGE_ME_UPSTREAM_SECRET --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 128 --kv-cache-dtype auto --enable-prefix-caching --port 8000
   ```
   **RTX 4090 (FP8, faster):**
   ```
   --model Qwen/Qwen3-8B-FP8 --served-model-name production-slm --api-key CHANGE_ME_UPSTREAM_SECRET --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 32 --kv-cache-dtype auto --enable-prefix-caching --port 8000
   ```
   Notes:
   - `--served-model-name production-slm` is a **stable alias**. Keep it constant so you can
     swap the underlying `--model` (8B → 14B → 32B) later without touching Windows.
   - `--api-key CHANGE_ME_UPSTREAM_SECRET` secures the vLLM endpoint (the RunPod URL is public).
     Use a long random value; you'll give the same value to the gateway as `UPSTREAM_API_KEY`.
5. (Recommended) Attach a **Network Volume** mounted at `/root/.cache/huggingface` so the model
   isn't re-downloaded on every restart.
6. Deploy and wait for the model to load (watch the pod logs for `Uvicorn running`).
7. **Copy the pod's HTTP URL for port 8000** — it looks like
   `https://<POD_ID>-8000.proxy.runpod.net`. Your vLLM base URL is that **+ `/v1`**.
8. **Verify** (from anywhere):
   ```bash
   curl https://<POD_ID>-8000.proxy.runpod.net/v1/models -H "Authorization: Bearer CHANGE_ME_UPSTREAM_SECRET"
   ```
   You should get a JSON model list.

> Keep the pod **running** (don't Terminate) so its URL stays stable. Stop/Start keeps the same
> URL; Terminate + recreate gives a new one (then just update one value on Windows — Part 5).

---

## Part 2 — Windows live server: install the gateway

Prerequisites: **Python 3.11 (64-bit)** (tick "Add to PATH"), and **Git**.

```powershell
# get the code
git clone https://github.com/MohammedFaadil/SLM-MODEL.git C:\slm
cd C:\slm

# venv + gateway deps (core + OCR + embeddings + DB driver)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -r requirements-ocr.txt
./.venv/Scripts/python.exe -m pip install -r requirements-embeddings.txt
./.venv/Scripts/python.exe -m pip install -r requirements-db.txt
```

> OCR and embeddings run on the Windows CPU here (fine for normal resume volumes). The heavy
> LLM work is on the RunPod GPU.

---

## Part 3 — MSSQL (audit log)

1. Install the **Microsoft ODBC Driver 18 for SQL Server** on the Windows server.
2. Create a database (e.g. `SLM_DB`). The gateway auto-creates the `slm_request_log` table on
   startup — no schema work needed.
3. You'll put the connection string in `.env` in Part 4.

(The gateway runs fine **without** MSSQL — leave the DB settings blank to skip it.)

---

## Part 4 — Configure and run the gateway

Create `C:\slm\.env`:

```ini
HOST=0.0.0.0
PORT=8080

# The key the PRODUCT will use (this replaces its OpenAI key). Use a long random value.
GATEWAY_API_KEYS=CHANGE_ME_PRODUCT_KEY

# Point at the RunPod vLLM pod from Part 1
LLM_BACKEND=openai_upstream
UPSTREAM_BASE_URL=https://<POD_ID>-8000.proxy.runpod.net/v1
UPSTREAM_API_KEY=CHANGE_ME_UPSTREAM_SECRET
MODEL_NAME=production-slm          # MUST match vLLM --served-model-name
SERVED_MODEL_ID=slm-qwen3           # the id the product sees at /v1/models
FORCE_MODEL=true
REQUEST_TIMEOUT=600
WARMUP_ON_START=true

# Embeddings + OCR (served from the Windows box)
EMBEDDINGS_MODE=local
OCR_ENABLED=true
OCR_VERSION=v5

# MSSQL audit log (leave blank to run without a DB)
DATABASE_URL=mssql+pyodbc://sqluser:sqlpass@DBHOST:1433/SLM_DB?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Encrypt=yes
```

Open the firewall and run it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\open_firewall.ps1 -Port 8080

# quick foreground test:
./.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Run it as an always-on Windows **service** (so it survives reboots) with NSSM
(https://nssm.cc/download):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1 -Port 8080
```

**Verify:**

```powershell
curl http://localhost:8080/health
# -> {"status":"ok","version":"0.4.0-gateway", ... }

curl http://localhost:8080/v1/chat/completions -H "Authorization: Bearer CHANGE_ME_PRODUCT_KEY" `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply: ready\"}]}'
```

A normal chat-completions JSON response = the full chain (product → gateway → RunPod vLLM) works.

---

## Part 5 — Give the product its URL + key (a clean HTTPS URL)

For a stable, professional URL over HTTPS, put a reverse proxy in front of port 8080.

**Easiest — Caddy** (one exe, automatic HTTPS). `Caddyfile`:
```
slm.yourdomain.com {
    reverse_proxy localhost:8080
}
```
Run `caddy run`. The product URL becomes `https://slm.yourdomain.com/v1`.

**Native — IIS:** install **IIS + Application Request Routing (ARR) + URL Rewrite**, bind a site
to 443 with your cert, add a reverse-proxy rule to `http://localhost:8080/`.

**Hand the product team two values:**

| | Value |
| --- | --- |
| Base URL | `https://slm.yourdomain.com/v1` (or `http://<windows-ip>:8080/v1`) |
| API key | your `GATEWAY_API_KEYS` value |

They set these as the OpenAI `base_url` + `api_key`. Their prompts, model name (`gpt-4o`, etc.),
streaming, JSON mode, and tools all keep working — nothing else changes.

**If the RunPod pod URL ever changes** (only on Terminate + recreate): update
`UPSTREAM_BASE_URL` in `C:\slm\.env` and restart the service (`nssm restart SLMGateway`). The
product's URL never changes.

---

## Change the model (8B -> 14B -> 32B)

Because the gateway routes by the **stable alias** `production-slm`, swapping the model is a
**RunPod-only** change — the Windows gateway is untouched. On the RunPod pod, change `--model`
(and pick a GPU with enough VRAM); keep `--served-model-name production-slm`.

| Want | RunPod `--model` | GPU (VRAM) | Windows gateway change |
| --- | --- | --- | --- |
| 8B (default, max accuracy) | `Qwen/Qwen3-8B` (bf16) | A100 80GB | none |
| 8B (faster) | `Qwen/Qwen3-8B-FP8` | RTX 4090 24GB | none |
| **14B (more accuracy)** | `Qwen/Qwen3-14B-FP8` | 48GB (L40S / A6000) | none |
| **32B (top accuracy)** | `Qwen/Qwen3-32B-FP8` | 80GB (H100 / A100) | none |

Steps:
1. On RunPod, edit the pod's start args: change `--model ...` to the new model id; for FP8 the
   `Qwen/...-FP8` checkpoints are already quantized (no extra flag). Keep everything else.
2. Restart the pod; wait for it to load.
3. Nothing to change on Windows — `MODEL_NAME=production-slm` still matches
   `--served-model-name production-slm`.

(If you'd rather NOT use a stable alias and instead set `--served-model-name` to the real
model id, then swapping the model also means updating `MODEL_NAME` in `C:\slm\.env` to match —
still just one env value, no code.)

---

## Verify / operate / troubleshoot

- **Health / build:** `curl http://localhost:8080/health` → must show `"version":"0.4.0-gateway"`.
  If it doesn't after a `git pull`, you didn't restart the service (`nssm restart SLMGateway`).
- **Logs:** `C:\slm\logs\gateway.out.log` / `gateway.err.log` (service), or the console.
- **`backend_unreachable` in responses** → the RunPod pod is down or `UPSTREAM_BASE_URL` is wrong.
  `curl <runpod-url>/v1/models -H "Authorization: Bearer <UPSTREAM_API_KEY>"`.
- **401 from the gateway** → the product isn't sending a key that's in `GATEWAY_API_KEYS`.
- **OCR 503** → OCR deps/model not ready on Windows (`pip install -r requirements-ocr.txt`; the
  first OCR call downloads the PP-OCRv5 model).
- **MSSQL warning at startup** → driver missing or bad connection string; the gateway still runs
  and serves normally (audit logging is optional).

---

## Simpler alternative — everything on RunPod (no Windows gateway)

If you don't need the URL to live on the Windows server, you can run **gateway + vLLM in one
RunPod container** using `Dockerfile.runpod`: build/push it, deploy as a GPU pod, expose port
`8080`, and set `GATEWAY_API_KEYS` + `HUGGING_FACE_HUB_TOKEN`. The product URL is then the pod's
`https://<POD_ID>-8080.proxy.runpod.net/v1`. MSSQL would connect from the pod to your Windows
SQL server. See `Dockerfile.runpod` and `scripts/runpod_start.sh`. The Windows-hosted setup above
is recommended when you want a stable, branded URL and the DB kept local.
