import json
from datetime import datetime
from typing import List
from jinja2 import Template
from pathlib import Path

from rich.console import Console
from rich.table import Table
from benchmark.models import BenchmarkResult

console = Console()

def print_results(results: list[BenchmarkResult]):
    table = Table(title="Token Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Total (ms)", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "✅" if r.success else "❌"
        table.add_row(
            r.provider, r.model,
            f"{r.ttft_ms:.0f}" if r.success else "-",
            f"{r.tokens_per_second:.1f}" if r.success else "-",
            f"{r.total_latency_ms:.0f}" if r.success else "-",
            str(r.total_tokens) if r.success else "-",
            status,
        )

    console.print(table)

def generate_html_report(results: list[BenchmarkResult], output_path: str = "benchmark_report.html"):
    template_path = Path(__file__).parent.parent / "templates" / "report.html"
    template = Template(template_path.read_text())

    html = template.render(
        results=results,
        results_json=json.dumps([
            {
                "provider": r.provider,
                "model": r.model,
                "ttft_ms": r.ttft_ms,
                "tokens_per_second": r.tokens_per_second,
                "total_latency_ms": r.total_latency_ms,
                "total_tokens": r.total_tokens,
                "success": r.success,
            } for r in results
        ]),
    )
    Path(output_path).write_text(html)
    print(f"📊 HTML report saved to: {output_path}")
