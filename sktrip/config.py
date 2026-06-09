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
    trip_model: str = "qwen3.6-27b-abliterated"   # abliterated/refusal-suppressed — the point of trip mode
    trip_api: str = "openai"                       # "openai" (/v1/chat/completions) | "ollama" (/api/generate)
    trip_base_url: str = "http://192.168.0.100:8082/v1"   # qwen3.6-27b-abliterated @ .100:8082 (OpenAI API)
    sober_model: str = "qwen3.5:4b"                # Ollama @ :11434 for sober/reflection steps
    embed_model: str = "mxbai-embed-large"
    timeout: float = 300.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class QdrantConfig:
    url: str = "https://skvector.skstack01.douno.it"
    api_key: str = ""
    collection: str = "lumina-memory"
    vector_dim: int = 1024


@dataclass
class SessionDefaults:
    output_dir: str = str(Path(__file__).parent.parent / "sessions")
    max_tokens_per_turn: int = 2048
    intensity_check_interval: int = 5
    peak_novelty_threshold: float = 0.7


@dataclass
class SkmemPgConfig:
    """skmem-pg (Postgres + pgvector + BM25 + AGE) — the default memory store.

    Reuses skmemory's local stack: the same DSN + mxbai-embed-large vector space.
    """
    dsn: str = field(default_factory=lambda: os.environ.get(
        "SKMEMORY_PG_DSN", "postgresql://postgres:skmemory@localhost:5432/skmemory"))
    agent: str = field(default_factory=lambda: (
        os.environ.get("SKAGENT") or os.environ.get("SKCAPSTONE_AGENT") or "lumina"))


@dataclass
class SKTripConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    skmempg: SkmemPgConfig = field(default_factory=SkmemPgConfig)
    session: SessionDefaults = field(default_factory=SessionDefaults)
    memory_backend: str = "skmempg"   # "skmempg" (default) | "qdrant" (legacy, optional)

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
        if "skmempg" in raw:
            for k, v in raw["skmempg"].items():
                if hasattr(cfg.skmempg, k):
                    setattr(cfg.skmempg, k, v)
        if "memory_backend" in raw:
            cfg.memory_backend = str(raw["memory_backend"])
        return cfg
