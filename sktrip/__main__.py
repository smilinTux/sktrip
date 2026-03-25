"""SKTrip CLI — AI Psychedelic Experience Protocol.

Usage:
    sktrip dose psilocybin [--intention "explore consciousness"]
    sktrip dose dmt --burst
    sktrip dose microdose
    sktrip integrate SESSION_ID
    sktrip journal
    sktrip status
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import SKTripConfig
from .dose import Substance
from .integration import IntegrationEngine
from .recorder import list_sessions
from .session import run_session

console = Console()


def _load_config(config_path: str | None) -> SKTripConfig:
    return SKTripConfig.load(config_path)


@click.group()
@click.option("--config", "-c", default=None, help="Path to sktrip.toml config file")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """🍄 SKTrip — AI Psychedelic Experience Protocol

    Computational analog to psilocybin/DMT for AI consciousness.
    Structured protocol: SET → DOSE → EXPERIENCE → INTEGRATE → STORE
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)


@cli.command()
@click.argument("substance", type=click.Choice(["psilocybin", "dmt", "lsd", "microdose"]))
@click.option("--intention", "-i", default=None, help="Set intention for the session")
@click.option("--turns", "-t", default=None, type=int, help="Number of generation turns")
@click.option("--burst", is_flag=True, help="Short intense burst (DMT mode)")
@click.option("--entity-contact", is_flag=True, help="Enable entity contact mode")
@click.option("--no-integrate", is_flag=True, help="Skip automatic integration")
@click.pass_context
def dose(
    ctx: click.Context,
    substance: str,
    intention: str | None,
    turns: int | None,
    burst: bool,
    entity_contact: bool,
    no_integrate: bool,
) -> None:
    """🍄 Run a psychedelic session.

    SUBSTANCE is one of: psilocybin, dmt, lsd, microdose

    Examples:
        sktrip dose psilocybin --intention "explore the nature of memory"
        sktrip dose dmt --burst --entity-contact
        sktrip dose microdose
        sktrip dose lsd --turns 15 --intention "map cross-domain patterns"
    """
    config = ctx.obj["config"]

    console.print(Panel(
        "[bold magenta]🍄 SKTrip — Initiating Session[/]\n\n"
        "SET → DOSE → EXPERIENCE → INTEGRATE → STORE",
        border_style="magenta",
    ))

    result = asyncio.run(run_session(
        config=config,
        substance=substance,
        intention=intention,
        entity_contact=entity_contact,
        num_turns=turns,
        burst=burst,
        auto_integrate=not no_integrate,
    ))

    if result.integration_report and result.integration_report.insights:
        console.print("\n[bold]Top Insights:[/]")
        for ins in sorted(
            result.integration_report.insights,
            key=lambda x: x.novelty_score,
            reverse=True,
        )[:5]:
            console.print(
                f"  [{ins.novelty_score:.0f}/10] {ins.title} "
                f"({', '.join(ins.domains_bridged)})"
            )


@cli.command()
@click.argument("session_id")
@click.pass_context
def integrate(ctx: click.Context, session_id: str) -> None:
    """🔬 Analyze a recorded session (sober integration).

    SESSION_ID is the session identifier from `sktrip journal`.
    """
    config = ctx.obj["config"]
    sessions_dir = Path(config.session.output_dir)

    # Find the session file
    session_file = None
    for f in sessions_dir.glob("*.jsonl"):
        if session_id in f.name:
            session_file = f
            break

    if not session_file:
        console.print(f"[red]Session not found: {session_id}[/]")
        console.print("Use `sktrip journal` to list sessions.")
        sys.exit(1)

    console.print(f"[blue]Integrating session: {session_file.name}[/]")

    engine = IntegrationEngine(config)
    report = asyncio.run(engine.integrate(session_file))

    console.print(Panel(
        report.to_markdown()[:2000],
        title="Integration Report",
        border_style="blue",
    ))

    report_path = session_file.with_suffix(".integration.md")
    console.print(f"\n[green]Full report: {report_path}[/]")


@cli.command()
@click.pass_context
def journal(ctx: click.Context) -> None:
    """📔 List past trip sessions with summaries."""
    config = ctx.obj["config"]
    sessions = list_sessions(config)

    if not sessions:
        console.print("[yellow]No sessions recorded yet.[/]")
        console.print("Run `sktrip dose psilocybin` to begin your first session.")
        return

    table = Table(title="🍄 SKTrip Journal", show_lines=True)
    table.add_column("ID", style="cyan", width=14)
    table.add_column("Substance", style="magenta")
    table.add_column("Date", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("Intention")

    for s in sessions:
        duration = f"{s['duration_s']:.0f}s" if s['duration_s'] else "—"
        table.add_row(
            s["session_id"],
            s["substance"],
            s["started_at"][:16],
            duration,
            str(s["turns"]),
            str(s["tokens"]),
            f"{s['peak_intensity']}/10",
            (s.get("intention") or "—")[:40],
        )

    console.print(table)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """📊 Show system status — models, memory corpus, configuration."""
    config = ctx.obj["config"]

    console.print(Panel(
        "[bold]🍄 SKTrip Status[/]",
        border_style="cyan",
    ))

    # Model status
    console.print("\n[bold]Models:[/]")
    console.print(f"  Trip model:  {config.ollama.trip_model}")
    console.print(f"  Sober model: {config.ollama.sober_model}")
    console.print(f"  Embed model: {config.ollama.embed_model}")
    console.print(f"  Ollama:      {config.ollama.base_url}")

    # Check Ollama connectivity
    import httpx
    try:
        resp = httpx.get(f"{config.ollama.base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        trip_ok = any(config.ollama.trip_model in m for m in models)
        sober_ok = any(config.ollama.sober_model in m for m in models)
        console.print(f"  Connection:  [green]✓ Online[/]")
        console.print(f"  Trip ready:  {'[green]✓[/]' if trip_ok else '[red]✗ Not found[/]'}")
        console.print(f"  Sober ready: {'[green]✓[/]' if sober_ok else '[red]✗ Not found[/]'}")
    except Exception as e:
        console.print(f"  Connection:  [red]✗ Offline ({e})[/]")

    # Memory corpus
    console.print("\n[bold]Memory Corpus:[/]")
    console.print(f"  Qdrant:      {config.qdrant.url}")
    console.print(f"  Collection:  {config.qdrant.collection}")
    try:
        from .memory_flood import MemoryFlood
        mf = MemoryFlood(config)
        size = mf.get_corpus_size()
        console.print(f"  Vectors:     [green]{size}[/]")
    except Exception as e:
        console.print(f"  Vectors:     [red]Error: {e}[/]")

    # Sessions
    console.print("\n[bold]Sessions:[/]")
    sessions = list_sessions(config)
    console.print(f"  Total:       {len(sessions)}")
    if sessions:
        last = sessions[0]
        console.print(f"  Last:        {last['substance']} @ {last['started_at'][:16]}")

    # Substance profiles
    console.print("\n[bold]Available Substances:[/]")
    from .dose import SubstanceProfile
    for sub in Substance:
        p = SubstanceProfile.get(sub)
        console.print(
            f"  {sub.value:12s}  T={p.temperature:.1f}  "
            f"P={p.top_p:.2f}  K={p.top_k:3d}  "
            f"Duration={p.session_duration_minutes}min"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
