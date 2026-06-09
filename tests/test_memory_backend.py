"""Tests for the memory backend selector (skmem-pg default, Qdrant optional)."""

from sktrip.config import SKTripConfig
from sktrip.memory_flood import MemoryFlood, SkmemPgFlood, make_memory_flood


def test_default_backend_is_skmempg():
    cfg = SKTripConfig()
    assert cfg.memory_backend == "skmempg"
    assert type(make_memory_flood(cfg)).__name__ == "SkmemPgFlood"


def test_qdrant_backend_still_available():
    cfg = SKTripConfig()
    cfg.memory_backend = "qdrant"
    assert isinstance(make_memory_flood(cfg), MemoryFlood)


def test_skmempg_config_defaults():
    cfg = SKTripConfig()
    assert "postgresql://" in cfg.skmempg.dsn
    assert cfg.skmempg.agent  # resolves to SKAGENT / lumina


def test_skmempg_flood_has_flood_interface():
    # Drop-in parity with MemoryFlood (no DB connection made here).
    f = SkmemPgFlood(SKTripConfig())
    for m in ("get_corpus_size", "pull_random", "pull_distant", "pull_cross_domain", "flood"):
        assert callable(getattr(f, m))


def test_rows_to_fragments_shape():
    rows = [("id1", "some text", ["health", "x"]), ("id2", None, None)]
    frags = SkmemPgFlood._rows_to_fragments(rows)
    assert frags[0].id == "id1" and frags[0].domain == "health"
    assert frags[1].text == "" and frags[1].tags == []
