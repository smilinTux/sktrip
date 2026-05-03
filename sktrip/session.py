"""Session orchestrator — ties together SET → DOSE → EXPERIENCE → INTEGRATE → STORE."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import SKTripConfig
from .dose import Substance, SubstanceProfile
from .freeassoc import FreeAssociationEngine
from .integration import IntegrationEngine, IntegrationReport
from .memory_flood import MemoryFlood
from .recorder import SessionRecorder

console = Console()


@dataclass
class SessionResult:
    """Result of a complete trip session."""
    session_id: str
    substance: str
    session_file: str
    total_turns: int
    total_tokens: int
    peak_intensity: int
    num_peaks: int
    duration_s: float
    integration_report: IntegrationReport | None = None


async def run_session(
    config: SKTripConfig,
    substance: Substance | str,
    intention: str | None = None,
    entity_contact: bool = False,
    num_turns: int | None = None,
    burst: bool = False,
    auto_integrate: bool = True,
) -> SessionResult:
    """Run a complete trip session.

    Phases:
    1. SET — prepare memory corpus, select profile, set intention
    2. DOSE — configure altered-state parameters
    3. EXPERIENCE — free association through memory space
    4. INTEGRATE — sober analysis (if auto_integrate=True)
    5. STORE — save insights to skmemory
    """
    if isinstance(substance, str):
        substance = Substance(substance.lower())

    profile = SubstanceProfile.get(substance)

    # Override for burst mode (DMT)
    if burst:
        profile.session_duration_minutes = 2
        if num_turns is None:
            num_turns = 3

    # Default turn count based on substance
    if num_turns is None:
        num_turns = {
            Substance.PSILOCYBIN: 8,
            Substance.DMT: 5,
            Substance.LSD: 6,
            Substance.MICRODOSE: 4,
        }.get(substance, 6)

    # Enable entity contact for DMT by default
    if substance == Substance.DMT:
        entity_contact = True

    console.print(Panel(
        f"[bold magenta]🍄 SKTrip Session — {substance.value.upper()}[/]\n\n"
        f"Temperature: {profile.temperature}\n"
        f"Top-P: {profile.top_p} | Top-K: {profile.top_k}\n"
        f"Turns: {num_turns}\n"
        f"Entity Contact: {'ON' if entity_contact else 'OFF'}\n"
        f"Intention: {intention or 'Open exploration'}",
        title="SET",
        border_style="magenta",
    ))

    # Phase 1: SET — prepare memory corpus
    console.print("\n[cyan]Phase 1: Loading memory corpus...[/]")
    memory_flood = MemoryFlood(config)
    corpus_size = memory_flood.get_corpus_size()
    console.print(f"  Memory corpus: {corpus_size} vectors")

    # Pull memories — use cross-domain for max novelty
    memories = memory_flood.flood(
        count=min(10, max(3, num_turns)),
        anchor=intention,
        cross_domain=True,
    )
    console.print(f"  Loaded {len(memories)} memory fragments for injection")

    if memories:
        for m in memories[:3]:
            console.print(f"    • {str(m)[:80]}...")

    # Phase 2: DOSE — initialize recorder
    console.print("\n[yellow]Phase 2: Configuring dose...[/]")
    recorder = SessionRecorder(
        config=config,
        substance=substance,
        intention=intention,
        entity_contact=entity_contact,
    )
    recorder.metadata.memory_corpus_size = corpus_size
    console.print(f"  Session ID: {recorder.session_id}")
    console.print(f"  Recording to: {recorder.session_file}")

    # Phase 3: EXPERIENCE — run free association
    console.print(f"\n[bold green]Phase 3: Entering {substance.value} space...[/]\n")

    engine = FreeAssociationEngine(
        config=config,
        profile=profile,
        recorder=recorder,
        memory_flood=memory_flood,
        entity_contact=entity_contact,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[magenta]Experiencing {substance.value}...",
            total=num_turns,
        )

        turns = await engine.run_chain(
            num_turns=num_turns,
            intention=intention,
            memory_fragments=memories,
        )

        for t in turns:
            progress.advance(task)
            # Print a snippet of each turn
            snippet = t.raw_output[:150].replace("\n", " ")
            console.print(f"  [dim]Turn {t.turn_number}:[/] {snippet}...")

    # Finalize recording
    session_file = recorder.finalize()
    console.print(f"\n[green]✓ Session recorded: {session_file}[/]")

    result = SessionResult(
        session_id=recorder.session_id,
        substance=substance.value,
        session_file=str(session_file),
        total_turns=recorder.metadata.total_turns,
        total_tokens=recorder.metadata.total_tokens,
        peak_intensity=recorder.metadata.peak_intensity,
        num_peaks=len(recorder.peaks),
        duration_s=time.time() - recorder.metadata.started_at,
    )

    # Phase 4 & 5: INTEGRATE + STORE
    if auto_integrate:
        console.print("\n[blue]Phase 4: Integration (sober analysis)...[/]")
        try:
            integration = IntegrationEngine(config)
            report = await integration.integrate(session_file)
            result.integration_report = report
            console.print(f"  Insights found: {len(report.insights)}")
            console.print(f"  Average novelty: {report.novelty_average:.1f}/10")
            console.print(f"  Themes: {', '.join(report.recurring_themes[:5])}")
            console.print(f"\n[green]✓ Integration complete[/]")
        except Exception as e:
            console.print(f"  [yellow]Integration skipped: {e}[/]")

    # Summary
    console.print(Panel(
        f"Session: {result.session_id}\n"
        f"Substance: {result.substance}\n"
        f"Turns: {result.total_turns} | Tokens: {result.total_tokens}\n"
        f"Peak Intensity: {result.peak_intensity}/10\n"
        f"Peaks Detected: {result.num_peaks}\n"
        f"Duration: {result.duration_s:.0f}s\n"
        f"File: {result.session_file}",
        title="[bold]🍄 Trip Complete",
        border_style="green",
    ))

    return result
