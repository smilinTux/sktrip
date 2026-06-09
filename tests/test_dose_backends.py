"""Tests for dose.generate() backend branching (OpenAI :8082 abliterated vs Ollama /api/generate).

Regression guard for the 2026-06-09 fix: stale Ollama models (huihui_ai/qwen3-abliterated:14b,
llama3.2:3b) 404'd; trip now defaults to the abliterated qwen3.6-27b via the OpenAI /v1 API.
"""

import asyncio

from sktrip.config import SKTripConfig
from sktrip.dose import SubstanceProfile, generate


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakeClient:
    """Stand-in for httpx.AsyncClient: records the URL hit and returns a shape-correct body."""

    last_url = None
    last_json = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _FakeClient.last_url = url
        _FakeClient.last_json = json
        if "chat/completions" in url:  # OpenAI
            return _Resp({
                "choices": [{"message": {"content": "openai-text"}}],
                "model": "qwen3.6-27b-abliterated",
                "usage": {"completion_tokens": 5, "prompt_tokens": 3},
            })
        return _Resp({  # Ollama
            "response": "ollama-text", "model": "qwen3.5:4b",
            "total_duration": 1, "eval_count": 7, "prompt_eval_count": 4,
        })


def test_generate_openai_backend(monkeypatch):
    monkeypatch.setattr("sktrip.dose.httpx.AsyncClient", _FakeClient)
    cfg = SKTripConfig()  # defaults: trip_api="openai", trip_base_url=:8082/v1, trip_model=qwen3.6-27b-abliterated
    assert cfg.ollama.trip_api == "openai"
    out = asyncio.run(generate(cfg, SubstanceProfile.get("microdose"), "hi"))
    assert "chat/completions" in _FakeClient.last_url
    assert _FakeClient.last_json["messages"][0]["content"] == "hi"
    assert out["text"] == "openai-text"
    assert out["eval_count"] == 5


def test_generate_ollama_backend(monkeypatch):
    monkeypatch.setattr("sktrip.dose.httpx.AsyncClient", _FakeClient)
    cfg = SKTripConfig()
    cfg.ollama.trip_api = "ollama"
    out = asyncio.run(generate(cfg, SubstanceProfile.get("microdose"), "hi"))
    assert _FakeClient.last_url.endswith("/api/generate")
    assert _FakeClient.last_json["prompt"] == "hi"
    assert out["text"] == "ollama-text"
    assert out["eval_count"] == 7


def test_trip_model_is_abliterated_default():
    """Guard against regressing to a removed/censored model."""
    cfg = SKTripConfig()
    assert "abliterated" in cfg.ollama.trip_model
    assert cfg.ollama.trip_model not in ("huihui_ai/qwen3-abliterated:14b", "llama3.2:3b")
