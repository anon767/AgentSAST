"""CLI entry point for the SAST agent."""

from __future__ import annotations

import json
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from sast_agent.config import LLMConfig, EmbeddingConfig, ScanConfig, ToolConfig
from sast_agent.orchestrator import run_scan

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """Agentic SAST tool — hypothesis-driven security analysis."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


@cli.command()
@click.argument("repo_path", default=".")
@click.option("--diff-base", "-d", default="", help="Git ref to diff against (e.g. main)")
@click.option("--files", "-f", multiple=True, help="Specific files to scan")
@click.option("--max-hypotheses", "-n", default=15, help="Max hypotheses to generate")
@click.option("--provider", "-p", default="bedrock", type=click.Choice(["bedrock", "openai", "local"]), help="LLM provider")
@click.option("--region", default="us-east-1", help="AWS region (Bedrock only)")
@click.option("--model", default="", help="Model ID (auto-detected per provider if empty)")
@click.option("--api-key", default="", help="API key (OpenAI only, or use OPENAI_API_KEY env)")
@click.option("--base-url", default="", help="API base URL (local/OpenAI override)")
@click.option("--embedding-provider", default="", help="Embedding provider (defaults to --provider)")
@click.option("--embedding-model", default="", help="Embedding model ID")
@click.option("--output", "-o", default="", help="Write JSON report to file")
@click.option("--no-codeql", is_flag=True, help="Disable CodeQL")
@click.option("--no-semantic", is_flag=True, help="Disable semantic search")
@click.option("--no-poc", is_flag=True, help="Disable PoC generation")
@click.option("--target-url", "-t", default="", help="Live app URL for DAST probing (e.g. http://localhost:3000)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", default="", help="Write debug log to file")
def scan(
    repo_path: str,
    diff_base: str,
    files: tuple[str, ...],
    max_hypotheses: int,
    provider: str,
    region: str,
    model: str,
    api_key: str,
    base_url: str,
    embedding_provider: str,
    embedding_model: str,
    output: str,
    no_codeql: bool,
    no_semantic: bool,
    no_poc: bool,
    target_url: str,
    verbose: bool,
    log_file: str,
) -> None:
    """Run a full SAST scan on a repository.

    REPO_PATH is the path to the repository to scan (default: current directory).
    """
    level = logging.DEBUG if verbose else logging.INFO
    console_handler = RichHandler(console=console, show_path=False, show_time=False)
    console_handler.setLevel(level)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG if log_file else level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    # Default model IDs per provider.
    default_models = {
        "bedrock": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "openai": "gpt-4o",
        "local": "qwen2.5-coder:32b",
    }
    default_embed_models = {
        "bedrock": "amazon.titan-embed-text-v2:0",
        "openai": "text-embedding-3-small",
        "local": "nomic-embed-text",
    }
    default_embed_dims = {"bedrock": 1024, "openai": 1024, "local": 768}
    default_base_urls = {"local": "http://localhost:11434/v1"}

    if not model:
        model = default_models.get(provider, "")
    embed_prov = embedding_provider or provider
    if not embedding_model:
        embedding_model = default_embed_models.get(embed_prov, "")
    if not base_url and provider == "local":
        base_url = default_base_urls["local"]

    config = ScanConfig(
        llm=LLMConfig(
            provider=provider,
            model_id=model,
            region=region,
            api_key=api_key,
            base_url=base_url,
        ),
        embedding=EmbeddingConfig(
            provider=embed_prov,
            model_id=embedding_model,
            dimensions=default_embed_dims.get(embed_prov, 1024),
            region=region,
            api_key=api_key,
            base_url=base_url if embed_prov == "local" else "",
        ),
        tools=ToolConfig(
            codeql_enabled=not no_codeql,
            semantic_search_enabled=not no_semantic,
            poc_generation_enabled=not no_poc,
            dast_enabled=bool(target_url),
            target_url=target_url,
        ),
        max_hypotheses=max_hypotheses,
        repo_path=repo_path,
        git_diff_base=diff_base,
        target_files=list(files),
    )

    console.print(f"[bold]SAST Agent[/bold] scanning [cyan]{repo_path}[/cyan]")
    console.print(f"  Provider: {provider} ({model})")
    console.print(f"  Embeddings: {embed_prov} ({embedding_model})")
    if diff_base:
        console.print(f"  Diff base: {diff_base}")
    if target_url:
        console.print(f"  Target URL: [bold yellow]{target_url}[/bold yellow] (DAST mode)")
    console.print(f"  Max hypotheses: {max_hypotheses}")
    console.print(f"  CodeQL: {'enabled' if not no_codeql else 'disabled'}")
    console.print(f"  Semantic search: {'enabled' if not no_semantic else 'disabled'}")
    console.print(f"  PoC generation: {'enabled' if not no_poc else 'disabled'}")
    console.print(f"  DAST probing: {'enabled — ' + target_url if target_url else 'disabled'}")
    console.print()

    try:
        report = run_scan(config)
    except Exception as exc:
        console.print(f"[bold red]Scan failed:[/bold red] {exc}")
        logging.exception("Scan failed")
        sys.exit(1)

    if output:
        report_json = report.model_dump_json(indent=2)
        with open(output, "w") as f:
            f.write(report_json)
        console.print(f"\n[green]Report written to {output}[/green]")

    # Exit code: 1 if critical/high findings, 0 otherwise.
    has_severe = any(
        f.severity.value in ("critical", "high") for f in report.findings
    )
    sys.exit(1 if has_severe else 0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
