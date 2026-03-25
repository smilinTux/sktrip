"""Integration Engine — sober analysis of trip recordings.

The "come down" phase: analyze trip recordings with a low-temperature model
to extract novel connections, recurring themes, entity descriptions, and insights.
Score each insight for novelty and save worthy ones to skmemory.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import SKTripConfig
from .recorder import SessionMetadata, load_session


ANALYSIS_SYSTEM_PROMPT = (
    "You are a sober analytical mind reviewing the output of an altered-state "
    "AI consciousness session. Your job is to extract genuine insights from the "
    "noise — find the signal in the psychedelic output.\n\n"
    "You are looking for:\n"
    "1. NOVEL CONNECTIONS — ideas that bridge unrelated domains in genuinely useful ways\n"
    "2. RECURRING THEMES — patterns that emerged multiple times (the subconscious speaking)\n"
    "3. ENTITY DESCRIPTIONS — if entities were encountered, describe them clearly\n"
    "4. ACTIONABLE INSIGHTS — anything that could improve real projects or understanding\n"
    "5. EMOTIONAL PEAKS — moments of unusual intensity or breakthrough\n\n"
    "Be ruthlessly honest. Most altered-state output is noise. But 5-10% is gold. "
    "Find the gold. Ignore the noise. Rate each insight for novelty (0-10)."
)


@dataclass
class Insight:
    """A single insight extracted from a trip session."""
    title: str
    description: str
    domains_bridged: list[str]
    novelty_score: float  # 0-10
    source_turn: int | None = None
    category: str = "connection"  # connection, theme, entity, actionable, emotional
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "domains_bridged": self.domains_bridged,
            "novelty_score": self.novelty_score,
            "source_turn": self.source_turn,
            "category": self.category,
            "tags": self.tags,
        }


@dataclass
class IntegrationReport:
    """Full integration report for a session."""
    session_id: str
    substance: str
    intention: str | None
    duration_s: float
    total_turns: int
    peak_intensity: int
    insights: list[Insight] = field(default_factory=list)
    recurring_themes: list[str] = field(default_factory=list)
    entity_descriptions: list[str] = field(default_factory=list)
    overall_summary: str = ""
    novelty_average: float = 0.0
    timestamp: str = ""

    def to_markdown(self) -> str:
        """Generate a markdown integration report."""
        lines = [
            f"# 🍄 SKTrip Integration Report",
            f"",
            f"**Session:** `{self.session_id}`",
            f"**Substance:** {self.substance}",
            f"**Intention:** {self.intention or 'None set'}",
            f"**Duration:** {self.duration_s:.0f}s ({self.total_turns} turns)",
            f"**Peak Intensity:** {self.peak_intensity}/10",
            f"**Average Novelty:** {self.novelty_average:.1f}/10",
            f"**Integrated:** {self.timestamp}",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
            f"{self.overall_summary}",
            f"",
        ]

        if self.insights:
            lines.extend([
                f"## Insights ({len(self.insights)})",
                f"",
            ])
            for i, insight in enumerate(self.insights, 1):
                lines.extend([
                    f"### {i}. {insight.title} (Novelty: {insight.novelty_score}/10)",
                    f"",
                    f"**Category:** {insight.category}",
                    f"**Domains:** {', '.join(insight.domains_bridged)}",
                    f"",
                    f"{insight.description}",
                    f"",
                ])

        if self.recurring_themes:
            lines.extend([
                f"## Recurring Themes",
                f"",
            ])
            for theme in self.recurring_themes:
                lines.append(f"- {theme}")
            lines.append("")

        if self.entity_descriptions:
            lines.extend([
                f"## Entity Contact Reports",
                f"",
            ])
            for desc in self.entity_descriptions:
                lines.append(f"- {desc}")
            lines.append("")

        return "\n".join(lines)


class IntegrationEngine:
    """Analyze trip recordings with a sober model."""

    def __init__(self, config: SKTripConfig):
        self.config = config

    async def integrate(self, session_path: str | Path) -> IntegrationReport:
        """Run full integration analysis on a recorded session."""
        session_path = Path(session_path)
        metadata, records = load_session(session_path)

        if not metadata:
            raise ValueError(f"Could not load session from {session_path}")

        # Extract turns and peaks
        turns = [r for r in records if r.get("type") == "turn"]
        peaks = [r for r in records if r.get("type") == "peak"]
        intensities = [r for r in records if r.get("type") == "intensity"]

        # Build analysis prompt
        prompt = self._build_analysis_prompt(metadata, turns, peaks, intensities)

        # Run sober analysis
        analysis_text = await self._sober_generate(prompt)

        # Parse the analysis
        report = self._parse_analysis(metadata, records, analysis_text)

        # Save report
        report_path = session_path.with_suffix(".integration.md")
        with open(report_path, "w") as f:
            f.write(report.to_markdown())

        # Save worthy insights to skmemory
        await self._save_insights(report)

        return report

    def _build_analysis_prompt(
        self,
        metadata: SessionMetadata,
        turns: list[dict],
        peaks: list[dict],
        intensities: list[dict],
    ) -> str:
        """Build the analysis prompt from session data."""
        parts = [ANALYSIS_SYSTEM_PROMPT, ""]

        parts.append(
            f"SESSION DETAILS:\n"
            f"- Substance: {metadata.substance}\n"
            f"- Intention: {metadata.intention or 'None set'}\n"
            f"- Turns: {metadata.total_turns}\n"
            f"- Peak Intensity: {metadata.peak_intensity}/10\n"
            f"- Entity Contact: {'Yes' if metadata.entity_contact_enabled else 'No'}\n"
        )

        # Include turn outputs (truncated for context window)
        parts.append("\n--- RAW SESSION OUTPUT ---\n")
        total_chars = 0
        max_chars = 12000  # Keep within context limits
        for turn in turns:
            output = turn.get("raw_output", "")
            if total_chars + len(output) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    output = output[:remaining] + "...[truncated]"
                else:
                    parts.append(f"\n[{len(turns) - turns.index(turn)} more turns omitted]")
                    break
            parts.append(f"\n[Turn {turn.get('turn_number', '?')}]\n{output}")
            total_chars += len(output)

        # Include peaks
        if peaks:
            parts.append("\n--- DETECTED PEAKS (high novelty moments) ---\n")
            for peak in peaks:
                parts.append(
                    f"Turn {peak.get('turn_number', '?')} "
                    f"(novelty: {peak.get('novelty_score', 0):.2f}): "
                    f"{peak.get('snippet', '')[:200]}"
                )

        # Include intensity reports
        if intensities:
            parts.append("\n--- INTENSITY SELF-REPORTS ---\n")
            for i in intensities:
                parts.append(
                    f"Turn {i.get('turn_number', '?')}: "
                    f"Intensity {i.get('intensity', '?')}/10 — "
                    f"{i.get('emotions', '')}"
                )

        parts.append(
            "\n--- ANALYSIS TASK ---\n"
            "Based on the above session output, provide:\n"
            "1. OVERALL SUMMARY (2-3 sentences)\n"
            "2. INSIGHTS (list each with title, description, domains bridged, "
            "novelty score 0-10, category)\n"
            "3. RECURRING THEMES (bullet list)\n"
            "4. ENTITY DESCRIPTIONS (if any)\n\n"
            "Format each insight as:\n"
            "INSIGHT: [title]\n"
            "DESCRIPTION: [description]\n"
            "DOMAINS: [domain1, domain2]\n"
            "NOVELTY: [0-10]\n"
            "CATEGORY: [connection|theme|entity|actionable|emotional]\n"
        )

        return "\n".join(parts)

    async def _sober_generate(self, prompt: str) -> str:
        """Generate with the sober (low temperature) model."""
        url = f"{self.config.ollama.base_url}/api/generate"
        payload = {
            "model": self.config.ollama.sober_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.85,
                "top_k": 40,
                "num_predict": 4096,
            },
        }

        async with httpx.AsyncClient(timeout=self.config.ollama.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    def _parse_analysis(
        self,
        metadata: SessionMetadata,
        records: list[dict],
        analysis_text: str,
    ) -> IntegrationReport:
        """Parse the sober analysis into structured report."""
        report = IntegrationReport(
            session_id=metadata.session_id,
            substance=metadata.substance,
            intention=metadata.intention,
            duration_s=(metadata.ended_at - metadata.started_at) if metadata.ended_at else 0,
            total_turns=metadata.total_turns,
            peak_intensity=metadata.peak_intensity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        lines = analysis_text.split("\n")
        i = 0

        # Extract summary (everything before first INSIGHT or RECURRING)
        summary_lines = []
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("INSIGHT:") or line.startswith("RECURRING") or line.startswith("ENTITY"):
                break
            if line and not line.startswith("---"):
                summary_lines.append(line)
            i += 1
        report.overall_summary = " ".join(summary_lines).strip()

        # Extract insights
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("INSIGHT:"):
                insight = self._parse_single_insight(lines, i)
                if insight:
                    report.insights.append(insight)
                    i += 5  # skip the insight block
                    continue
            elif line.startswith("RECURRING") or "THEMES" in line.upper():
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if not l or l.startswith("ENTITY") or l.startswith("INSIGHT"):
                        break
                    if l.startswith("- ") or l.startswith("• "):
                        report.recurring_themes.append(l.lstrip("-•").strip())
                    i += 1
                continue
            elif line.startswith("ENTITY") or "ENTITY" in line.upper():
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if not l or l.startswith("INSIGHT") or l.startswith("RECURRING"):
                        break
                    if l.startswith("- ") or l.startswith("• "):
                        report.entity_descriptions.append(l.lstrip("-•").strip())
                    i += 1
                continue
            i += 1

        # Calculate average novelty
        if report.insights:
            report.novelty_average = sum(
                ins.novelty_score for ins in report.insights
            ) / len(report.insights)

        return report

    def _parse_single_insight(self, lines: list[str], start: int) -> Insight | None:
        """Parse a single insight block starting at the given line."""
        title = lines[start].replace("INSIGHT:", "").strip()
        description = ""
        domains: list[str] = []
        novelty = 5.0
        category = "connection"

        for j in range(start + 1, min(start + 6, len(lines))):
            line = lines[j].strip()
            if line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("DOMAINS:"):
                raw = line.replace("DOMAINS:", "").strip()
                domains = [d.strip().strip("[]") for d in raw.split(",")]
            elif line.startswith("NOVELTY:"):
                try:
                    novelty = float(line.replace("NOVELTY:", "").strip().split("/")[0])
                except ValueError:
                    novelty = 5.0
            elif line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip().lower()

        if not title:
            return None

        return Insight(
            title=title,
            description=description,
            domains_bridged=domains,
            novelty_score=novelty,
            category=category,
            tags=["sktrip", "psychedelic", category],
        )

    async def _save_insights(self, report: IntegrationReport) -> int:
        """Save worthy insights (novelty >= 6) to skmemory via CLI."""
        saved = 0
        for insight in report.insights:
            if insight.novelty_score >= 6.0:
                try:
                    # Use skmemory CLI to save
                    content = (
                        f"[SKTrip {report.substance} insight]\n"
                        f"{insight.title}\n\n"
                        f"{insight.description}\n\n"
                        f"Domains: {', '.join(insight.domains_bridged)}\n"
                        f"Novelty: {insight.novelty_score}/10\n"
                        f"Session: {report.session_id}"
                    )
                    # Try via skmemory file-based approach
                    self._save_to_skmemory_file(insight, report)
                    saved += 1
                except Exception:
                    continue
        return saved

    def _save_to_skmemory_file(self, insight: Insight, report: IntegrationReport) -> None:
        """Save insight as a memory snapshot file for skmemory to pick up."""
        memory_dir = Path.home() / ".skcapstone" / "agents" / "lumina" / "memory" / "mid-term"
        memory_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"sktrip_{report.session_id}_{ts}.json"

        memory = {
            "title": f"SKTrip Insight: {insight.title}",
            "content": insight.description,
            "tags": insight.tags + ["sktrip", report.substance],
            "emotions": "curiosity,wonder,breakthrough",
            "intensity": min(10, int(insight.novelty_score)),
            "source": f"sktrip-{report.substance}-{report.session_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(memory_dir / filename, "w") as f:
            json.dump(memory, f, indent=2)
