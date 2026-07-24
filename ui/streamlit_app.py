"""Minimal test UI for the SLM gateway (GPU-box dev tool — not part of the product).

Two testers:
  1. Chat  -> POST /v1/chat/completions   (the OpenAI surface the product uses)
  2. OCR   -> POST /api/ocr/parse          (PDF/image -> text)

Run (gateway must be running):
  ./.venv/Scripts/python.exe -m pip install -r requirements-ui.txt
  ./.venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
"""
import json

import requests
import streamlit as st

st.set_page_config(page_title="SLM Gateway — Test UI", page_icon="🧪", layout="wide")

st.sidebar.title("🧪 SLM Gateway test")
base_url = st.sidebar.text_input("Gateway URL", value="http://localhost:8000").rstrip("/")
api_key = st.sidebar.text_input("API key (if gateway auth is on)", value="", type="password")
headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

try:
    h = requests.get(f"{base_url}/health", headers=headers, timeout=10).json()
    st.sidebar.markdown(
        f"**version:** `{h.get('version')}`  \n"
        f"**backend:** `{h.get('backend', {}).get('backend')}` → `{h.get('model')}`  \n"
        f"**status:** {h.get('status')}"
    )
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Gateway unreachable: {exc}")

tab_chat, tab_ocr = st.tabs(["① Chat (/v1/chat/completions)", "② OCR (/api/ocr/parse)"])

# --------------------------------------------------------------------------- #
with tab_chat:
    st.header("Chat completion")
    model = st.text_input("model", value="slm-qwen3-8b",
                          help="Any value — the gateway routes it to the served model.")
    system = st.text_area("system prompt (optional)", height=80)
    user = st.text_area("user message", value="Say hello in one sentence.", height=120)
    stream = st.checkbox("stream", value=False)

    if st.button("Send", type="primary") and user.strip():
        messages = ([{"role": "system", "content": system}] if system.strip() else []) + \
                   [{"role": "user", "content": user}]
        payload = {"model": model, "messages": messages, "stream": stream}
        url = f"{base_url}/v1/chat/completions"
        try:
            if stream:
                out = st.empty()
                acc = ""
                with requests.post(url, json=payload, headers=headers, stream=True, timeout=900) as resp:
                    if resp.status_code != 200:
                        st.error(f"{resp.status_code}: {resp.text}")
                    else:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            s = line.decode("utf-8").removeprefix("data: ").strip()
                            if s == "[DONE]":
                                break
                            try:
                                delta = json.loads(s)["choices"][0]["delta"].get("content", "")
                                acc += delta
                                out.markdown(acc)
                            except Exception:
                                pass
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=900)
                if resp.status_code != 200:
                    st.error(f"{resp.status_code}: {resp.text}")
                else:
                    data = resp.json()
                    st.markdown(data["choices"][0]["message"]["content"])
                    st.caption(f"usage: {data.get('usage')} · model: {data.get('model')}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Connection error: {exc}")

# --------------------------------------------------------------------------- #
with tab_ocr:
    st.header("OCR — PDF / image to text")
    up = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"])
    if st.button("Extract text", type="primary") and up:
        try:
            resp = requests.post(f"{base_url}/api/ocr/parse", headers=headers,
                                 files={"file": (up.name, up.getvalue())}, timeout=900)
            if resp.status_code != 200:
                st.error(f"{resp.status_code}: {resp.text}")
            else:
                d = resp.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Pages", d.get("num_pages"))
                c2.metric("Status", d.get("status"))
                c3.metric("Confidence", d.get("overall_confidence") or "—")
                st.caption(f"method: {d.get('method_summary')}")
                st.text_area("Extracted text", value=d.get("text", ""), height=400)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Connection error: {exc}")
