"""Tests for configuration loading."""

import tempfile
from pathlib import Path

from sktrip.config import SKTripConfig


class TestConfig:
    def test_defaults(self):
        config = SKTripConfig()
        assert config.ollama.host == "192.168.0.100"
        assert config.ollama.port == 11434
        assert config.ollama.trip_model == "huihui_ai/qwen3-abliterated:14b"
        assert config.ollama.sober_model == "llama3.2:3b"
        assert config.qdrant.collection == "lumina-memory"
        assert config.qdrant.vector_dim == 1024

    def test_base_url(self):
        config = SKTripConfig()
        assert config.ollama.base_url == "http://192.168.0.100:11434"

    def test_load_nonexistent(self):
        config = SKTripConfig.load("/nonexistent/path.toml")
        assert config.ollama.host == "192.168.0.100"

    def test_load_from_file(self, tmp_path):
        toml_content = """
[ollama]
host = "10.0.0.1"
port = 9999
trip_model = "custom-model:7b"

[session]
max_tokens_per_turn = 4096
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        config = SKTripConfig.load(config_file)
        assert config.ollama.host == "10.0.0.1"
        assert config.ollama.port == 9999
        assert config.ollama.trip_model == "custom-model:7b"
        assert config.session.max_tokens_per_turn == 4096
        # Unchanged defaults
        assert config.ollama.sober_model == "llama3.2:3b"
        assert config.qdrant.collection == "lumina-memory"
