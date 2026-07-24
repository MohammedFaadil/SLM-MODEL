# Test UI (dev tool — not part of the product)

A tiny Streamlit harness to eyeball the gateway on a GPU box. Two testers:

1. **Chat** → `POST /v1/chat/completions` (the OpenAI surface the product uses)
2. **OCR** → `POST /api/ocr/parse` (PDF/image → text)

The gateway itself ships **no prompts** — the product supplies those. This UI just
sends a raw chat request and an OCR upload.

## Run

```powershell
# 1) start the gateway (separate terminal)
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 2) install UI deps + launch
./.venv/Scripts/python.exe -m pip install -r requirements-ui.txt
./.venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
```

Set the **Gateway URL** and (if enabled) **API key** in the sidebar.
