"""Core daemon-behavior tests for the free-association chain.

The daily microdose daemon (`sktrip dose microdose`) is driven by
`FreeAssociationEngine.run_chain`, which was previously untested. These tests
exercise the orchestration loop with a stubbed LLM so they run offline:

  * the requested number of turns is generated, recorded, and returned;
  * token totals accumulate into session metadata;
  * memory is re-injected on the cadence the engine promises.
"""

import asyncio

import pytest

import sktrip.freeassoc as fa
from sktrip.config import SKTripConfig
from sktrip.dose import Substance, SubstanceProfile
from sktrip.freeassoc import FreeAssociationEngine
from sktrip.memory_flood import MemoryFragment
from sktrip.recorder import SessionRecorder, load_session


@pytest.fixture
def tmp_config(tmp_path):
    config = SKTripConfig()
    config.session.output_dir = str(tmp_path)
    # High interval so the intensity self-check (a separate LLM call) stays out
    # of these turn-count tests.
    config.session.intensity_check_interval = 999
    return config


class _CountingFlood:
    """Fake memory backend that records how often pull_random is called."""

    def __init__(self):
        self.pull_random_calls = 0

    def pull_random(self, count):
        self.pull_random_calls += 1
        return [MemoryFragment(id=f"r{i}", text="reinjected", tags=["x"]) for i in range(count)]


def _stub_generate(monkeypatch):
    calls = {"n": 0, "prompts": []}

    async def fake_generate(config, profile, prompt, max_tokens=2048):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return {"text": f"turn {calls['n']} alpha beta gamma", "eval_count": 3}

    monkeypatch.setattr(fa, "generate", fake_generate)
    return calls


def _run_chain(config, *, num_turns, entity_contact=False, flood=None):
    flood = flood or _CountingFlood()
    recorder = SessionRecorder(config=config, substance=Substance.MICRODOSE, intention="test")
    engine = FreeAssociationEngine(
        config=config,
        profile=SubstanceProfile.get("microdose"),
        recorder=recorder,
        memory_flood=flood,
        entity_contact=entity_contact,
    )
    turns = asyncio.run(engine.run_chain(
        num_turns=num_turns,
        intention="test",
        memory_fragments=[MemoryFragment(id="seed", text="seed mem", tags=["y"])],
    ))
    recorder.finalize()
    return turns, recorder, flood


def test_run_chain_generates_and_records_all_turns(tmp_config, monkeypatch):
    calls = _stub_generate(monkeypatch)
    turns, recorder, _ = _run_chain(tmp_config, num_turns=4)

    assert len(turns) == 4
    assert calls["n"] == 4                      # exactly one LLM call per turn
    assert recorder.metadata.total_turns == 4
    assert recorder.metadata.total_tokens == 12  # 4 turns * 3 tokens
    assert [t.turn_number for t in turns] == [1, 2, 3, 4]


def test_run_chain_persists_turns_to_jsonl(tmp_config, monkeypatch):
    _stub_generate(monkeypatch)
    _, recorder, _ = _run_chain(tmp_config, num_turns=3)

    meta, records = load_session(recorder.session_file)
    assert meta is not None
    turn_records = [r for r in records if r.get("type") == "turn"]
    assert len(turn_records) == 3
    assert all(r["raw_output"] for r in turn_records)


def test_run_chain_reinjects_memory_on_cadence(tmp_config, monkeypatch):
    """Engine promises fresh random memories every 3rd turn (i>0 and i%3==0)."""
    _stub_generate(monkeypatch)
    _, _, flood = _run_chain(tmp_config, num_turns=7)
    # i in 0..6 -> reinjection at i=3 and i=6 -> two pull_random calls.
    assert flood.pull_random_calls == 2


def test_run_chain_entity_contact_injects_entity_prompt(tmp_config, monkeypatch):
    """Regression: entity-contact mode must actually surface an ENTITY prompt.

    The injection guard used to be `if extra_prompt and not self.entity_contact`,
    which can never be true (extra_prompt is only set WHEN entity_contact is on),
    so the labeled [ENTITY CONTACT] section was dead code. This pins that an
    entity turn (i % 4 == 2) now carries entity-contact text into the prompt.
    """
    calls = _stub_generate(monkeypatch)
    _run_chain(tmp_config, num_turns=4, entity_contact=True)
    joined = "\n".join(calls["prompts"])
    # The labeled [ENTITY CONTACT] block must now appear (it was dead code before).
    assert "[ENTITY CONTACT]" in joined, joined
