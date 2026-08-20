"""
run.py
──────
Command-line interface for the Research Agent System.
Usage: python run.py "Your research query here"
"""

import sys
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.streaming import run_sync
from core.evaluation import evaluate_run
from tools.rag_tool import ingest_directory

console = Console()


def print_evaluation(eval_result) -> None:
    """Print evaluation results as a Rich table."""
    table = Table(
        title="📊 Research Quality Evaluation",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Score", justify="center")
    table.add_column("Grade", justify="center")

    table.add_row("Overall Score",     f"{eval_result.overall_score}/10", eval_result.grade)
    table.add_row("Quality",           f"{eval_result.quality_score}/10", "")
    table.add_row("Coverage",          f"{eval_result.coverage_score}/10", "")
    table.add_row("Accuracy",          f"{eval_result.accuracy_score}/10", "")
    table.add_row("Depth",             f"{eval_result.depth_score}/10", "")
    table.add_row("Source Diversity",  f"{eval_result.source_diversity}/10", "")
    table.add_row("Fact Verified",     f"{eval_result.fact_verification_rate:.0%}", "")
    table.add_row("Sources Used",      str(eval_result.source_count), "")
    table.add_row("Critique Loops",    str(eval_result.critique_loops), "")
    table.add_row("Report Words",      str(eval_result.word_count), "")

    console.print(table)
    console.print(f"\n[dim]{eval_result.summary}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research & Report Generation System"
    )
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents from ./data/documents/ before researching"
    )
    parser.add_argument(
        "--docs-dir",
        default="./data/documents",
        help="Directory containing documents to ingest"
    )

    args = parser.parse_args()

    # ── Optional document ingestion ───────────────────────────────────────────
    if args.ingest:
        console.print(Panel(
            f"[bold]📄 Ingesting documents from {args.docs_dir}...[/bold]",
            border_style="cyan"
        ))
        count = ingest_directory(args.docs_dir)
        console.print(f"[green]✅ Ingested {count} document chunks[/green]")
        if not args.query:
            return

    # ── Get query ─────────────────────────────────────────────────────────────
    query = args.query
    if not query:
        console.print("[bold cyan]🔬 Multi-Agent Research System[/bold cyan]")
        query = console.input("[bold]Enter your research query: [/bold]").strip()
        if not query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    # ── Run research ──────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]🚀 Starting research:[/bold]\n{query}",
        border_style="blue"
    ))

    try:
        markdown, final_state = run_sync(query)

        # ── Print report ──────────────────────────────────────────────────────
        console.print("\n")
        console.print(Panel("[bold green]📄 FINAL REPORT[/bold green]", border_style="green"))
        console.print(Markdown(markdown))

        # ── Print evaluation ──────────────────────────────────────────────────
        console.print("\n")
        eval_result = evaluate_run(final_state)
        print_evaluation(eval_result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()