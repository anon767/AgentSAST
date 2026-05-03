"""Orchestrator: runs the full planner → analyzer(s) → verifier pipeline.

The planner creates hypotheses, the orchestrator spawns analyzer agents for
each one (sharing the same tool handler / semantic index), and the verifier
produces the final deduplicated report.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sast_agent.agents.analyzer import run_analyzer
from sast_agent.agents.planner import run_planner
from sast_agent.agents.verifier import run_verifier
from sast_agent.providers import create_llm_from_config
from sast_agent.config import ScanConfig
from sast_agent.models import (
    AnalyzerOutput,
    HypothesisStatus,
    PlannerOutput,
    ScanReport,
)
from sast_agent.tool_registry import ToolHandler

logger = logging.getLogger(__name__)
console = Console()


def run_scan(config: ScanConfig) -> ScanReport:
    """Execute the full SAST scan pipeline."""
    bedrock = create_llm_from_config(config.llm)

    shared_tool_handler = ToolHandler(config)

    # ── Phase 1: Planner ────────────────────────────────────────────────
    console.print(Panel("[bold cyan]Phase 1: Planner[/bold cyan] — mapping attack surface"))
    t0 = time.time()
    planner_output = run_planner(config, bedrock)
    planner_time = time.time() - t0

    _print_planner_summary(planner_output, planner_time)

    if not planner_output.hypotheses:
        console.print("[yellow]Planner produced no hypotheses. Nothing to analyze.[/yellow]")
        return ScanReport(
            repo=config.repo_path,
            summary="No hypotheses generated.",
            hypotheses_total=0,
        )

    max_workers = min(config.max_concurrent_analyzers, len(planner_output.hypotheses))
    console.print(Panel(
        f"[bold cyan]Phase 2: Analyzers[/bold cyan] — "
        f"investigating {len(planner_output.hypotheses)} hypotheses "
        f"({max_workers} concurrent)"
    ))
    analyzer_outputs: list[AnalyzerOutput] = [None] * len(planner_output.hypotheses)  # type: ignore[list-item]

    def _run_one(idx: int, hypothesis):
        """Run a single analyzer in a thread. Each gets its own LLM client."""
        thread_llm = create_llm_from_config(config.llm)
        t0 = time.time()
        try:
            result = run_analyzer(
                hypothesis=hypothesis,
                config=config,
                bedrock=thread_llm,
                tool_handler=shared_tool_handler,
            )
            elapsed = time.time() - t0
            return idx, result, elapsed, None
        except Exception as exc:
            elapsed = time.time() - t0
            return idx, None, elapsed, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, i, h): (i, h)
            for i, h in enumerate(planner_output.hypotheses)
        }

        for future in as_completed(futures):
            i, hypothesis = futures[future]
            idx, result, elapsed, error = future.result()
            tag = f"[{idx+1}/{len(planner_output.hypotheses)}]"
            prio = hypothesis.priority.value.upper()
            short = hypothesis.hypothesis[:70]

            if error:
                logger.error("Analyzer failed for %s: %s", hypothesis.id, error)
                console.print(f"  {tag} [bold]{prio}[/bold] — {short} ... ❌ ERROR [{elapsed:.0f}s]")
                analyzer_outputs[idx] = AnalyzerOutput(
                    hypothesis_id=hypothesis.id,
                    status=HypothesisStatus.UNCERTAIN,
                    confidence=0.0,
                    reasoning=f"Analyzer error: {error}",
                )
            else:
                hypothesis.status = result.status
                analyzer_outputs[idx] = result
                status_icon = {
                    HypothesisStatus.CONFIRMED: "🔴",
                    HypothesisStatus.DISPROVEN: "✅",
                    HypothesisStatus.UNCERTAIN: "🟡",
                }.get(result.status, "❓")
                console.print(
                    f"  {tag} [bold]{prio}[/bold] — {short} ... "
                    f"{status_icon} {result.status.value} ({result.confidence:.0%}) [{elapsed:.0f}s]"
                )

    _print_analyzer_summary(analyzer_outputs)

    confirmed_or_uncertain = [
        ao for ao in analyzer_outputs
        if ao.status in (HypothesisStatus.CONFIRMED, HypothesisStatus.UNCERTAIN)
    ]

    if not confirmed_or_uncertain:
        console.print("[green]All hypotheses disproven. No findings to verify.[/green]")
        return ScanReport(
            repo=config.repo_path,
            summary="All hypotheses were disproven. No security findings.",
            hypotheses_total=len(planner_output.hypotheses),
            hypotheses_confirmed=0,
            hypotheses_disproven=len(analyzer_outputs),
            hypotheses_uncertain=0,
        )

    console.print(Panel("[bold cyan]Phase 3: Verifier[/bold cyan] — deduplicating and ranking"))
    report = run_verifier(planner_output, analyzer_outputs, config, bedrock)

    _print_final_report(report)
    return report



def _print_planner_summary(output: PlannerOutput, elapsed: float) -> None:
    console.print(f"\n[dim]Planner completed in {elapsed:.1f}s[/dim]")
    console.print(f"[bold]Repo:[/bold] {output.repo_summary[:120]}")
    console.print(f"[bold]Attack surface areas:[/bold] {len(output.attack_surface)}")
    console.print(f"[bold]Hypotheses generated:[/bold] {len(output.hypotheses)}")

    if output.hypotheses:
        table = Table(title="Hypotheses", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Priority", width=8)
        table.add_column("Risk Area", width=16)
        table.add_column("Hypothesis", min_width=40)
        table.add_column("Files", width=30)

        for i, h in enumerate(output.hypotheses, 1):
            color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}.get(
                h.priority.value, "white"
            )
            table.add_row(
                str(i),
                f"[{color}]{h.priority.value}[/{color}]",
                h.risk_area,
                h.hypothesis[:80],
                ", ".join(h.files[:3]) + ("..." if len(h.files) > 3 else ""),
            )
        console.print(table)
    console.print()


def _print_analyzer_summary(outputs: list[AnalyzerOutput]) -> None:
    confirmed = sum(1 for o in outputs if o.status == HypothesisStatus.CONFIRMED)
    disproven = sum(1 for o in outputs if o.status == HypothesisStatus.DISPROVEN)
    uncertain = sum(1 for o in outputs if o.status == HypothesisStatus.UNCERTAIN)

    console.print(
        f"\n[bold]Analyzer results:[/bold] "
        f"🔴 {confirmed} confirmed  ✅ {disproven} disproven  🟡 {uncertain} uncertain"
    )
    console.print()


def _print_final_report(report: ScanReport) -> None:
    console.print(Panel("[bold green]Scan Complete[/bold green]"))
    console.print(f"[bold]Summary:[/bold] {report.summary}")
    console.print(
        f"[bold]Hypotheses:[/bold] {report.hypotheses_total} total, "
        f"{report.hypotheses_confirmed} confirmed, "
        f"{report.hypotheses_disproven} disproven, "
        f"{report.hypotheses_uncertain} uncertain"
    )
    console.print(f"[bold]Findings:[/bold] {len(report.findings)}")

    if report.findings:
        table = Table(title="Security Findings", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Severity", width=10)
        table.add_column("Confidence", width=10)
        table.add_column("Title", min_width=40)
        table.add_column("Files", width=30)

        for i, f in enumerate(report.findings, 1):
            color = {
                "critical": "bold red",
                "high": "red",
                "medium": "yellow",
                "low": "dim",
            }.get(f.severity.value, "white")
            table.add_row(
                str(i),
                f"[{color}]{f.severity.value.upper()}[/{color}]",
                f"{f.confidence:.0%}",
                f.title,
                ", ".join(f.affected_files[:3]),
            )
        console.print(table)

        # Print details for each finding.
        for i, f in enumerate(report.findings, 1):
            console.print(f"\n[bold]Finding {i}: {f.title}[/bold]")
            console.print(f"  Severity: {f.severity.value} | Confidence: {f.confidence:.0%}")
            console.print(f"  {f.description}")
            if f.exploitability:
                console.print(f"  [red]Exploitability:[/red] {f.exploitability}")
            if f.minimal_fix:
                console.print(f"  [green]Fix:[/green] {f.minimal_fix}")
            if f.test_or_poc:
                console.print(f"  [cyan]PoC:[/cyan] {f.test_or_poc[:200]}")
    console.print()
