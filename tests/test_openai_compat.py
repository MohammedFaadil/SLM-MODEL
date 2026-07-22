from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_models_endpoint():
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "slm-qwen3-8b" in ids


def test_chat_completion_shape_and_model_remap():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    # Platform can send any model id; we advertise our served id back.
    assert body["model"] == "slm-qwen3-8b"


def test_chat_completion_streaming():
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as s:
        text = "".join(line for line in s.iter_lines())
    assert "data:" in text and "[DONE]" in text
