"""First-run / interactive on-boarding flow.

Walks the user through:

  1. Pick a starting point (one of the known providers or custom URL)
  2. Enter the API key (masked) and base URL
  3. Auto-detect the endpoint dialect
  4. Fetch the catalog of chat-capable models
  5. Multi-select which models to benchmark
  6. Optionally loop to add another endpoint
  7. Hand the chosen ``EndpointConfig`` list off to the runner
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from benchmark.endpoint_detector import (
    EndpointStyle,
    detect_endpoint_style,
    normalize_base_url,
)
from benchmark.menu import (
    ask_confirm,
    ask_text,
    select_models,
)
from benchmark.model_fetcher import ModelInfo, fallback_models, fetch_models
from benchmark.providers import DEFAULT_BASE_URLS

console = Console()


# ---------------------------------------------------------------------------
# Configuration object the runner consumes
# ---------------------------------------------------------------------------

@dataclass
class EndpointConfig:
    """Everything the runner needs to hit a model."""

    owner: str                              # short label for the result table
    api_key: str
    base_url: str
    style: EndpointStyle
    model_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step banners
# ---------------------------------------------------------------------------

def _banner(step: int, total: int, text: str) -> None:
    console.print(
        f"\n[bold magenta]Step {step}/{total}: {text}[/bold magenta]"
    )


def _ok(msg: str) -> None:
    console.print(f"  [green]OK[/green] {msg}")


def _warn(msg: str) -> None:
    console.print(f"  [yellow]WARN[/yellow] {msg}")


def _err(msg: str) -> None:
    console.print(f"  [red]ERR[/red] {msg}")


# ---------------------------------------------------------------------------
# Step 1 — pick the starting point
# ---------------------------------------------------------------------------

def _choose_start() -> tuple[str, str, EndpointStyle]:
    """Return (owner, default_url, hint_style)."""
    console.print(
        Panel.fit(
            "[bold]Token Benchmark — Onboarding[/bold]\n"
            "Configure one or more endpoints, then pick which models to test.\n"
            "All inputs are stored only in this session — nothing is written to disk.",
            border_style="magenta",
        )
    )

    if not sys.stdin.isatty():
        console.print(
            "  [dim]Non-interactive mode: enter 'owner|url|key' or "
            "hit enter for OpenAI defaults[/dim]"
        )
        line = input("  > ").strip()
        if not line:
            return ("openai", DEFAULT_BASE_URLS["openai"], EndpointStyle.OPENAI)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            raise SystemExit("Need owner|url|key in non-interactive mode")
        style = EndpointStyle.ANTHROPIC if "anthropic" in parts[0].lower() else EndpointStyle.OPENAI
        return (parts[0], parts[1], style)

    console.print("\n  [bold]Choose a starting point:[/bold]")
    console.print("   [cyan]1[/cyan]. OpenAI          (api.openai.com)")
    console.print("   [cyan]2[/cyan]. Anthropic        (api.anthropic.com)")
    console.print("   [cyan]3[/cyan]. DeepSeek         (api.deepseek.com)")
    console.print("   [cyan]4[/cyan]. Kimi / Moonshot  (api.moonshot.cn)")
    console.print("   [cyan]5[/cyan]. Custom URL       (Ollama / vLLM / OpenRouter / anything else)")

    choice = ask_text("  Selection", default="1").strip() or "1"
    mapping = {
        "1": ("openai",    DEFAULT_BASE_URLS["openai"],    EndpointStyle.OPENAI),
        "2": ("anthropic", DEFAULT_BASE_URLS["anthropic"], EndpointStyle.ANTHROPIC),
        "3": ("deepseek",  DEFAULT_BASE_URLS["deepseek"],  EndpointStyle.OPENAI),
        "4": ("kimi",      DEFAULT_BASE_URLS["kimi"],      EndpointStyle.OPENAI),
        "5": ("custom",    "",                             EndpointStyle.OPENAI),
    }
    return mapping.get(choice, mapping["1"])


# ---------------------------------------------------------------------------
# Step 2 — credentials
# ---------------------------------------------------------------------------

def _gather_credentials(owner: str, default_url: str) -> tuple[str, str]:
    base_url = ask_text(
        f"  Base URL for {owner}",
        default=default_url or "",
    ).strip()
    if base_url:
        base_url = normalize_base_url(base_url)
    api_key = ask_text("  API Key", default="", password=True).strip()
    if not api_key:
        _warn("No API key entered — listing models may fail, but a "
              "fallback catalog will be used.")
    return (api_key, base_url)


# ---------------------------------------------------------------------------
# Step 3 — detect + 4 — list models
# ---------------------------------------------------------------------------

def _detect_and_list(
    owner: str,
    base_url: str,
    api_key: str,
    hint: EndpointStyle,
) -> tuple[EndpointStyle, List[ModelInfo]]:
    if not base_url:
        return hint, fallback_models(hint, owner)

    _banner(3, 4, f"Detecting endpoint style for [cyan]{base_url}[/cyan]")
    probe = detect_endpoint_style(base_url, api_key, hint=hint)
    if probe.style == EndpointStyle.UNKNOWN:
        _warn(f"Could not auto-detect style ({probe.note}). "
              f"Falling back to hint={hint.value}.")
        style = hint
    else:
        _ok(f"Detected [bold]{probe.style.label}[/bold] "
            f"(HTTP {probe.status_code}, {probe.elapsed_ms:.0f} ms) — {probe.note}")
        style = probe.style

    _banner(4, 4, "Fetching available models")
    try:
        models = fetch_models(style, base_url, api_key)
    except Exception as e:
        _err(f"Model fetch failed: {e}")
        models = []

    if not models:
        _warn("Endpoint returned no models (or /v1/models is disabled). "
              "Using a hard-coded catalog instead.")
        models = fallback_models(style, owner)

    for m in models:
        if not m.owned_by:
            m.owned_by = owner
    _ok(f"Found {len(models)} candidate model(s).")
    return style, models


# ---------------------------------------------------------------------------
# Step 5 — multi-select
# ---------------------------------------------------------------------------

def _select(models: List[ModelInfo], owner: str) -> List[ModelInfo]:
    if not models:
        return []
    title = f"Pick models on {owner}  (Space=toggle, a=all, /=filter, Enter=confirm)"
    result = select_models(models, title=title)
    if result.cancelled:
        _warn("Selection cancelled.")
        return []
    _ok(f"Selected {len(result.selected)} model(s).")
    return result.selected


# ---------------------------------------------------------------------------
# Top-level on-boarding driver
# ---------------------------------------------------------------------------

def run_onboarding(
    *,
    preset: Optional[EndpointConfig] = None,
) -> List[EndpointConfig]:
    """Drive the interactive flow and return the list of chosen endpoint configs.

    If ``preset`` is provided we skip straight to the model picker.
    """
    if preset is not None:
        return [preset]

    endpoints: List[EndpointConfig] = []
    while True:
        owner, default_url, hint = _choose_start()
        api_key, base_url = _gather_credentials(owner, default_url)
        if not base_url and not api_key:
            _err("Need at least a base URL to proceed.")
            break

        style, models = _detect_and_list(owner, base_url, api_key, hint)
        chosen = _select(models, owner)
        if chosen:
            endpoints.append(EndpointConfig(
                owner=owner,
                api_key=api_key,
                base_url=base_url,
                style=style,
                model_ids=[m.id for m in chosen],
            ))

        if not ask_confirm("\n  Add another endpoint?", default=False):
            break
    return endpoints


def print_endpoint_summary(endpoints: List[EndpointConfig]) -> None:
    """Pretty table of what we're about to benchmark."""
    if not endpoints:
        return
    table = Table(
        title="Endpoints queued for benchmarking",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Owner", style="cyan")
    table.add_column("Style")
    table.add_column("Base URL", style="dim")
    table.add_column("Models", justify="right")
    for ep in endpoints:
        table.add_row(
            ep.owner,
            ep.style.label,
            ep.base_url or "(default)",
            str(len(ep.model_ids)),
        )
    console.print(table)
