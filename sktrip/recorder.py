"""Trip Recorder — full session capture to timestamped JSONL.

Captures raw model outputs, metadata, emotional intensity tracking,
and peak detection for moments of unusual semantic density or novelty.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SKTripConfig
from .dose import Substance


@dataclass
class TurnRecord:
    """A single turn in a trip session."""
    turn_number: int
    timestamp: float
    prompt_seed: str
    raw_output: str
    temperature: float
    top_p: float
    top_k: int
    tokens_generated: int
    generation_time_s: float
    memory_fragments_used: list[str] = field(default_factory=list)
    entity_contact: bool = False
    disruptions_injected: int = 0
    intensity_report: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = "turn"
        return d


@dataclass
class SessionMetadata:
    """Metadata for a trip session."""
    session_id: str
    substance: str
    intention: str | None
    started_at: float
    ended_at: float | None = None
    total_turns: int = 0
    total_tokens: int = 0
    peak_intensity: int = 0
    peak_turn: int = 0
    memory_corpus_size: int = 0
    entity_contact_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = "metadata"
        return d


class SessionRecorder:
    """Records a full trip session to JSONL files."""

    def __init__(
        self,
        config: SKTripConfig,
        substance: Substance,
        intention: str | None = None,
        entity_contact: bool = False,
        session_id: str | None = None,
    ):
        self.config = config
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.output_dir = Path(config.session.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / f"{ts}_{substance.value}_{self.session_id}.jsonl"

        self.metadata = SessionMetadata(
            session_id=self.session_id,
            substance=substance.value,
            intention=intention,
            started_at=time.time(),
            entity_contact_enabled=entity_contact,
        )

        self._turns: list[TurnRecord] = []
        self._intensities: list[dict] = []
        self._peaks: list[dict] = []

        # Write initial metadata
        self._write_line(self.metadata.to_dict())

    def record_turn(self, turn: TurnRecord) -> None:
        """Record a single turn."""
        self._turns.append(turn)
        self.metadata.total_turns += 1
        self.metadata.total_tokens += turn.tokens_generated
        self._write_line(turn.to_dict())

        # Check for peaks
        novelty = self._calculate_novelty(turn)
        if novelty >= self.config.session.peak_novelty_threshold:
            peak = {
                "type": "peak",
                "turn_number": turn.turn_number,
                "timestamp": turn.timestamp,
                "novelty_score": novelty,
                "snippet": turn.raw_output[:300],
            }
            self._peaks.append(peak)
            self._write_line(peak)

    def record_intensity(self, turn_number: int, intensity_data: dict) -> None:
        """Record an intensity self-report."""
        entry = {
            "type": "intensity",
            "turn_number": turn_number,
            "timestamp": time.time(),
            **intensity_data,
        }
        self._intensities.append(entry)
        self._write_line(entry)

        val = intensity_data.get("intensity", 0)
        if val > self.metadata.peak_intensity:
            self.metadata.peak_intensity = val
            self.metadata.peak_turn = turn_number

    def finalize(self) -> Path:
        """Finalize the session and write closing metadata."""
        self.metadata.ended_at = time.time()
        self._write_line({
            "type": "session_end",
            "session_id": self.session_id,
            "ended_at": self.metadata.ended_at,
            "total_turns": self.metadata.total_turns,
            "total_tokens": self.metadata.total_tokens,
            "peak_intensity": self.metadata.peak_intensity,
            "num_peaks": len(self._peaks),
            "duration_s": self.metadata.ended_at - self.metadata.started_at,
        })
        return self.session_file

    def _write_line(self, data: dict) -> None:
        """Append a JSON line to the session file."""
        with open(self.session_file, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _calculate_novelty(self, turn: TurnRecord) -> float:
        """Estimate novelty of a turn by measuring vocabulary divergence.

        Compares the word distribution of this turn against all previous turns.
        Higher score = more novel vocabulary = potential breakthrough moment.
        """
        if len(self._turns) < 2:
            return 0.5  # neutral for first turn

        # Current turn word frequencies
        current_words = Counter(turn.raw_output.lower().split())

        # All previous turns combined
        prev_words: Counter = Counter()
        for t in self._turns[:-1]:
            prev_words.update(t.raw_output.lower().split())

        if not prev_words or not current_words:
            return 0.5

        # Calculate Jaccard distance (1 - similarity)
        current_set = set(current_words.keys())
        prev_set = set(prev_words.keys())
        intersection = current_set & prev_set
        union = current_set | prev_set

        if not union:
            return 0.5

        jaccard_distance = 1.0 - (len(intersection) / len(union))

        # Also factor in unique words (hapax legomena ratio)
        unique_to_current = current_set - prev_set
        hapax_ratio = len(unique_to_current) / max(1, len(current_set))

        # Combined novelty score
        novelty = 0.6 * jaccard_distance + 0.4 * hapax_ratio
        return min(1.0, max(0.0, novelty))

    @property
    def turns(self) -> list[TurnRecord]:
        return list(self._turns)

    @property
    def peaks(self) -> list[dict]:
        return list(self._peaks)

    @property
    def intensities(self) -> list[dict]:
        return list(self._intensities)


def load_session(session_path: Path) -> tuple[SessionMetadata | None, list[dict]]:
    """Load a session from a JSONL file.

    Returns (metadata, records) where records include turns, peaks, intensities.
    """
    metadata = None
    records: list[dict] = []

    if not session_path.exists():
        return None, []

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "metadata":
                    metadata = SessionMetadata(**{
                        k: v for k, v in data.items()
                        if k != "type" and k in SessionMetadata.__dataclass_fields__
                    })
                else:
                    records.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

    return metadata, records


def list_sessions(config: SKTripConfig) -> list[dict]:
    """List all recorded sessions with basic metadata."""
    output_dir = Path(config.session.output_dir)
    if not output_dir.exists():
        return []

    sessions = []
    for f in sorted(output_dir.glob("*.jsonl"), reverse=True):
        meta, records = load_session(f)
        if meta:
            duration = (meta.ended_at - meta.started_at) if meta.ended_at else 0
            sessions.append({
                "session_id": meta.session_id,
                "substance": meta.substance,
                "intention": meta.intention,
                "started_at": datetime.fromtimestamp(meta.started_at, tz=timezone.utc).isoformat(),
                "duration_s": round(duration, 1),
                "turns": meta.total_turns,
                "tokens": meta.total_tokens,
                "peak_intensity": meta.peak_intensity,
                "file": str(f),
            })
    return sessions
