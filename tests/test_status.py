"""Regression tests for the `sktrip status` command.

Bug (fixed 2026-07-03): `status` probed the Ollama `/api/tags` endpoint for the
TRIP model and instantiated the Qdrant `MemoryFlood` directly. But the trip model
(abliterated qwen3.6-27b) is served via the OpenAI-compatible endpoint at :8082,
and the active memory backend is skmem-pg — so a *healthy* system was misreported
as "Trip ready: ✗ Not found" and "Vectors: 0".

These tests pin the corrected behavior:
  * trip readiness is probed against the OpenAI `/models` endpoint (trip_base_url),
    NOT Ollama `/api/tags`;
  * the memory corpus is read through the ACTIVE backend factory (make_memory_flood),
    not a hardcoded Qdrant handle.
"""

import httpx
import pytest
from click.testing import CliRunner

import sktrip.__main__ as m
import sktrip.memory_flood as mf
from sktrip.__main__ import cli


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _fake_get_factory(hits):
    """Return a fake httpx.get that records URLs and answers by endpoint.

    Crucially, the Ollama /api/tags response does NOT include the trip model —
    so the only way `Trip ready: ✓` can appear is via the OpenAI /models probe.
    """
    def _fake_get(url, timeout=None):
        hits.append(url)
        if url.endswith("/models"):  # OpenAI-compatible trip endpoint
            return _Resp({"data": [{"id": "qwen3.6-27b-abliterated"}]})
        if url.endswith("/api/tags"):  # Ollama: sober + embed only, NO trip model
            return _Resp({"models": [
                {"name": "qwen3.5:4b"},
                {"name": "mxbai-embed-large:latest"},
            ]})
        return _Resp({})
    return _fake_get


class _FakeFlood:
    def get_corpus_size(self):
        return 4242


@pytest.fixture
def patched(monkeypatch):
    hits: list[str] = []
    monkeypatch.setattr(httpx, "get", _fake_get_factory(hits))
    monkeypatch.setattr(mf, "make_memory_flood", lambda config: _FakeFlood())
    # Avoid touching the real session journal on disk.
    monkeypatch.setattr(m, "list_sessions", lambda config: [])
    return hits


def test_status_reports_trip_ready_via_openai_endpoint(patched):
    hits = patched
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    # The OpenAI /models endpoint MUST have been probed for the trip model...
    assert any(u.endswith("/models") for u in hits), hits
    # ...and the trip model is reported ready even though /api/tags lacks it.
    assert "Trip ready:  ✓" in result.output
    # Old bug signature must be gone.
    assert "Trip ready:  ✗ Not found" not in result.output


def test_status_reports_active_skmempg_backend(patched):
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "skmem-pg" in result.output
    # Corpus size comes from the active backend factory (4242), not Qdrant's 0.
    assert "4242" in result.output


def test_status_reports_sober_and_embed_ready(patched):
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "Sober ready: ✓" in result.output
    assert "Embed ready: ✓" in result.output
