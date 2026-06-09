#!/usr/bin/env python3
"""Token Benchmark CLI.

Default behaviour (no flags) walks the user through an interactive
on-boarding flow: pick a provider, enter key + URL, auto-detect the
endpoint style, fetch available models, multi-select, and benchmark.

The legacy ``-p`` flag still works for users who already have a
``config.yaml`` set up.
"""
from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel

from benchmark.endpoint_detector import EndpointStyle
from benchmark.onboarding import EndpointConfig, print_endpoint_summary, run_onboarding
from benchmark.reporter import generate_html_report, print_results
from benchmark.runner import export_results, run_from_config, run_from_endpoints

console = Console()


def _parse_add_endpoint(arg: str) -> EndpointConfig:
    """Parse ``--add-endpoint owner=openai|key=sk-xxx|url=...|style=openai|model=gpt-4o``."""
    cfg = EndpointConfig(owner="custom", api_key="", base_url="",
                         style=EndpointStyle.OPENAI)
    models: list[str] = []
    for part in arg.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "owner":
            cfg.owner = v
        elif k == "key":
            cfg.api_key = v
        elif k == "url":
            cfg.base_url = v
        elif k == "style":
            cfg.style = EndpointStyle.ANTHROPIC if v.lower() == "anthropic" else EndpointStyle.OPENAI
        elif k == "model":
            models.extend([m.strip() for m in v.split(",") if m.strip()])
    cfg.model_ids = models
    if not cfg.model_ids:
        console.print("[red]--add-endpoint requires a model=... field[/red]")
        sys.exit(2)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Token-speed benchmark for OpenAI / Anthropic / OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-p", "--providers", nargs="+",
                        help="Use legacy config.yaml path with the named providers")
    parser.add_argument("-r", "--rounds", type=int, default=3,
                        help="Rounds per model (default: 3)")
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Config file path (legacy mode)")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("-o", "--output", default="benchmark_report.html",
                        help="HTML output path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show live streaming output, token counts, and calculation breakdown")
    parser.add_argument("--prompt", default=None,
                        help="Override the benchmark prompt")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-request timeout in seconds")
    parser.add_argument("--add-endpoint", action="append", default=[],
                        metavar="owner=k|key=k|url=u|style=s|model=m",
                        help="Pre-add an endpoint (repeatable, skips on-boarding)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Use plain input() even when a TTY is attached")
    parser.add_argument("--list-models", action="store_true",
                        help="For -p mode: print configured models and exit")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup rounds to discard before measurement (default: 1)")
    parser.add_argument("--export", metavar="PATH", default=None,
                        help="Export results to JSON or CSV (inferred from extension)")
    args = parser.parse_args()

    # ----------------------------------------------------------- header
    console.print(Panel.fit(
        "[bold]Token Benchmark[/bold]\n"
        "OpenAI  ·  Anthropic  ·  any OpenAI-compatible endpoint",
        border_style="magenta",
    ))

    # ----------------------------------------------------- legacy path
    if args.providers:
        console.print(f"[dim]Legacy mode[/dim] providers={args.providers} "
                      f"rounds={args.rounds} verbose={args.verbose}")
        if args.list_models:
            from benchmark.runner import load_config
            cfg = load_config(args.config)
            for name in args.providers:
                p = cfg.get("providers", {}).get(name, {}) or {}
                console.print(f"  {name}: model={p.get('model', '?')!r} "
                              f"url={p.get('base_url') or '(default)'}")
            return
        results = run_from_config(
            providers=args.providers, rounds=args.rounds,
            config_path=args.config, verbose=args.verbose,
            warmup=args.warmup,
        )

    # --------------------------------------------- on-boarding / preset
    else:
        endpoints: list[EndpointConfig] = []
        for arg in args.add_endpoint:
            endpoints.append(_parse_add_endpoint(arg))
        if not endpoints:
            endpoints = run_onboarding()
        if not endpoints:
            console.print("[yellow]No endpoints configured. Exiting.[/yellow]")
            return
        print_endpoint_summary(endpoints)

        prompt = args.prompt or os.environ.get(
            "BENCHMARK_PROMPT",
            "请用100字介绍一下人工智能的发展历史。",
        )
        results = run_from_endpoints(
            endpoints=endpoints,
            rounds=args.rounds,
            prompt=prompt,
            timeout=args.timeout,
            verbose=args.verbose,
            warmup=args.warmup,
        )

    # ------------------------------------------------------ report
    console.print()
    print_results(results)
    if args.html:
        generate_html_report(results, args.output)
    if args.export:
        export_results(results, args.export)


if __name__ == "__main__":
    main()
