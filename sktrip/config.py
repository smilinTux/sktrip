"""Configuration loader for SKTrip."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "sktrip.toml"


@dataclass
class OllamaConfig:
    host: str = "192.168.0.100"
    port: int = 11434
    trip_model: str = "huihui_ai/qwen3-abliterated:14b"
    sober_model: str = "llama3.2:3b"
    embed_model: str = "mxbai-embed-large"
    timeout: float = 300.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class QdrantConfig:
    url: str = "https://skvector.skstack01.douno.it"
    api_key: str = "e4hPZkg0Q899N7x0FmgNPT+s8QvY7a/LOnl0go1QCIQ"
    collection: str = "lumina-memory"
    vector_dim: int = 1024


@dataclass
class SessionDefaults:
    output_dir: str = str(Path(__file__).parent.parent / "sessions")
    max_tokens_per_turn: int = 2048
    intensity_check_interval: int = 5
    peak_novelty_threshold: float = 0.7


@dataclass
class SKTripConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    session: SessionDefaults = field(default_factory=SessionDefaults)

    @classmethod
    def load(cls, path: str | Path | None = None) -> SKTripConfig:
        """Load config from TOML file, falling back to defaults."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        with open(config_path, "rb") as f:
            raw = tomllib.load(f)

        cfg = cls()
        if "ollama" in raw:
            for k, v in raw["ollama"].items():
                if hasattr(cfg.ollama, k):
                    setattr(cfg.ollama, k, v)
        if "qdrant" in raw:
            for k, v in raw["qdrant"].items():
                if hasattr(cfg.qdrant, k):
                    setattr(cfg.qdrant, k, v)
        if "session" in raw:
            for k, v in raw["session"].items():
                if hasattr(cfg.session, k):
                    setattr(cfg.session, k, v)
        return cfg
