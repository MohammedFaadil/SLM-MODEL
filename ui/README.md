# Test UI (disposable — do not ship)

This folder is a **testing harness only**. It is listed in `.gitignore` so it
never lands in the main product. Delete it any time.

It talks to the gateway over HTTP exactly like your real platform would, so it's
a faithful way to eyeball SLM quality for the three flows:

1. **AI Job Creation** — title + skills + experience → full job spec
2. **Add Candidate** — resume PDF/image → PaddleOCR → structured profile + AI summary
3. **Job ↔ Candidate Match** — matched/missing skills + AI justification

## Run

```powershell
# 1) start the gateway (separate terminal)
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 2) install UI deps + launch
./.venv/Scripts/python.exe -m pip install -r requirements-ui.txt
./.venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
```

Set the **Gateway URL** and (if enabled) **API key** in the sidebar.
