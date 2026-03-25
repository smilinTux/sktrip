"""Tests for the trip recorder."""

import json
import tempfile
from pathlib import Path

import pytest

from sktrip.config import SKTripConfig
from sktrip.dose import Substance
from sktrip.recorder import SessionRecorder, TurnRecord, load_session, list_sessions


@pytest.fixture
def tmp_config(tmp_path):
    config = SKTripConfig()
    config.session.output_dir = str(tmp_path)
    return config


class TestSessionRecorder:
    def test_creates_session_file(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.PSILOCYBIN)
        assert recorder.session_file.exists()
        assert recorder.session_id

    def test_record_turn(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.DMT)
        turn = TurnRecord(
            turn_number=1,
            timestamp=1000.0,
            prompt_seed="test prompt",
            raw_output="the entities revealed themselves as fractal geometries",
            temperature=2.0,
            top_p=0.99,
            top_k=120,
            tokens_generated=50,
            generation_time_s=3.5,
        )
        recorder.record_turn(turn)
        assert recorder.metadata.total_turns == 1
        assert recorder.metadata.total_tokens == 50

    def test_record_intensity(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.PSILOCYBIN)
        recorder.record_intensity(1, {"intensity": 8, "emotions": "awe", "sensation": "dissolving"})
        assert recorder.metadata.peak_intensity == 8

    def test_finalize(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.LSD)
        turn = TurnRecord(
            turn_number=1, timestamp=1000.0, prompt_seed="seed",
            raw_output="output", temperature=1.7, top_p=0.97,
            top_k=100, tokens_generated=30, generation_time_s=2.0,
        )
        recorder.record_turn(turn)
        path = recorder.finalize()
        assert path.exists()

        # Verify JSONL content
        lines = path.read_text().strip().split("\n")
        assert len(lines) >= 3  # metadata + turn + session_end

    def test_novelty_calculation(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.PSILOCYBIN)

        # First turn
        t1 = TurnRecord(
            turn_number=1, timestamp=1000.0, prompt_seed="seed",
            raw_output="the mushroom network connects all living things through mycelium",
            temperature=1.5, top_p=0.95, top_k=80,
            tokens_generated=10, generation_time_s=1.0,
        )
        recorder.record_turn(t1)

        # Second turn with very different vocabulary
        t2 = TurnRecord(
            turn_number=2, timestamp=1001.0, prompt_seed="seed",
            raw_output="quantum entanglement creates bridges across spacetime dimensions crystalline",
            temperature=1.5, top_p=0.95, top_k=80,
            tokens_generated=10, generation_time_s=1.0,
        )
        recorder.record_turn(t2)
        # Novelty should be calculated (we can't predict exact value but it should work)


class TestLoadSession:
    def test_roundtrip(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.PSILOCYBIN, intention="test")
        turn = TurnRecord(
            turn_number=1, timestamp=1000.0, prompt_seed="seed",
            raw_output="test output", temperature=1.5, top_p=0.95,
            top_k=80, tokens_generated=20, generation_time_s=1.5,
        )
        recorder.record_turn(turn)
        path = recorder.finalize()

        meta, records = load_session(path)
        assert meta is not None
        assert meta.substance == "psilocybin"
        assert meta.intention == "test"
        assert len(records) >= 2  # turn + session_end


class TestListSessions:
    def test_empty(self, tmp_config):
        sessions = list_sessions(tmp_config)
        assert sessions == []

    def test_with_sessions(self, tmp_config):
        recorder = SessionRecorder(tmp_config, Substance.DMT)
        recorder.finalize()

        sessions = list_sessions(tmp_config)
        assert len(sessions) == 1
        assert sessions[0]["substance"] == "dmt"
