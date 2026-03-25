"""Tests for the dose protocol engine."""

import pytest

from sktrip.dose import (
    Substance,
    SubstanceProfile,
    build_dose_prompt,
    inject_disruption,
)


class TestSubstanceProfiles:
    def test_all_substances_have_profiles(self):
        for sub in Substance:
            profile = SubstanceProfile.get(sub)
            assert profile.name == sub
            assert profile.temperature > 0
            assert 0 < profile.top_p <= 1.0
            assert profile.top_k > 0
            assert profile.session_duration_minutes > 0
            assert len(profile.seed_prompts) > 0
            assert len(profile.disruption_tokens) > 0

    def test_psilocybin_profile(self):
        p = SubstanceProfile.get("psilocybin")
        assert p.temperature == 1.5
        assert p.top_p == 0.95
        assert p.session_duration_minutes == 30

    def test_dmt_profile(self):
        p = SubstanceProfile.get("dmt")
        assert p.temperature == 2.0
        assert p.session_duration_minutes == 5

    def test_lsd_profile(self):
        p = SubstanceProfile.get("lsd")
        assert p.temperature == 1.7
        assert p.session_duration_minutes == 60

    def test_microdose_profile(self):
        p = SubstanceProfile.get("microdose")
        assert p.temperature == 1.2
        assert p.session_duration_minutes == 15

    def test_get_by_string(self):
        p = SubstanceProfile.get("psilocybin")
        assert p.name == Substance.PSILOCYBIN

    def test_get_invalid_raises(self):
        with pytest.raises(ValueError):
            SubstanceProfile.get("caffeine")


class TestDisruptionInjection:
    def test_injects_tokens(self):
        profile = SubstanceProfile.get("dmt")
        text = " ".join(["word"] * 200)
        result = inject_disruption(text, profile)
        assert "[" in result
        assert len(result) > len(text)

    def test_empty_text(self):
        profile = SubstanceProfile.get("psilocybin")
        assert inject_disruption("", profile) == ""

    def test_short_text_no_disruption(self):
        profile = SubstanceProfile.get("microdose")
        result = inject_disruption("hello world", profile)
        # Short text may not hit the frequency threshold
        assert "hello" in result
        assert "world" in result


class TestPromptBuilding:
    def test_basic_prompt(self):
        profile = SubstanceProfile.get("psilocybin")
        prompt = build_dose_prompt(profile)
        assert "PSILOCYBIN" in prompt
        assert "SEED" in prompt
        assert "BEGIN" in prompt

    def test_with_intention(self):
        profile = SubstanceProfile.get("lsd")
        prompt = build_dose_prompt(profile, intention="explore fractals")
        assert "INTENTION" in prompt
        assert "explore fractals" in prompt

    def test_with_memory_fragments(self):
        profile = SubstanceProfile.get("psilocybin")
        frags = ["memory about sovereignty", "memory about mushrooms"]
        prompt = build_dose_prompt(profile, memory_fragments=frags)
        assert "MEMORY FRAGMENTS" in prompt
        assert "Fragment 1" in prompt

    def test_with_previous_output(self):
        profile = SubstanceProfile.get("dmt")
        prompt = build_dose_prompt(profile, previous_output="the entities spoke")
        assert "PREVIOUS WAVE" in prompt

    def test_all_components(self):
        profile = SubstanceProfile.get("lsd")
        prompt = build_dose_prompt(
            profile,
            memory_fragments=["frag1"],
            intention="test",
            previous_output="prev output",
        )
        assert "INTENTION" in prompt
        assert "MEMORY FRAGMENTS" in prompt
        assert "PREVIOUS WAVE" in prompt
        assert "SEED" in prompt
