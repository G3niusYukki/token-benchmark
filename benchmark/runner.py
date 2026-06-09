"""Benchmark runner.

Two entry points:

* :func:`run_from_config` — legacy path: read ``config.yaml`` and use the
  hard-coded :data:`benchmark.providers.PROVIDERS` registry. Used by the
  ``-p openai anthropic ...`` CLI flag.

* :func:`run_from_endpoints` — modern path: take a list of
  :class:`benchmark.onboarding.EndpointConfig` produced by the on-boarding
  flow (or ``--add-endpoint`` on the CLI) and run them through the
  :class:`UniversalProvider`.

Both honor ``warmup_rounds``: those rounds run first and are discarded
before measurement. The first measured round is reported separately as
``cold_start_*`` and the median of the rest as ``steady_state_tps``.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from typing import List, Optional

import yaml

from benchmark.models import BenchmarkResult
from benchmark.onboarding import EndpointConfig
from benchmark.pricing import annotate as annotate_cost
from benchmark.providers import PROVIDERS, UniversalProvider


# ---------------------------------------------------------------------------
# YAML config loader (legacy)
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return {"providers": {}, "benchmark": {}}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    def replace_env(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], "")
        return v

    def walk(d):
        if isinstance(d, dict):
            return {k: walk(replace_env(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [walk(i) for i in d]
        return replace_env(d)

    return walk(raw)


# ---------------------------------------------------------------------------
# Universal helper — drive any (api_key, model, base_url, style) combo
# ---------------------------------------------------------------------------

def _build_universal(ep: EndpointConfig, model_id: str, verbose: bool) -> UniversalProvider:
    return UniversalProvider(
        api_key=ep.api_key,
        model=model_id,
        base_url=ep.base_url or None,
        style=ep.style,
        owner=ep.owner,
        verbose=verbose,
    )


def _run_rounds(
    *,
    label: str,
    provider: UniversalProvider,
    prompt: str,
    rounds: int,
    timeout: int,
    warmup: int = 0,
) -> List[BenchmarkResult]:
    """Run ``warmup + rounds`` requests; return all of them in order."""
    total = warmup + rounds
    out: List[BenchmarkResult] = []
    for i in range(total):
        phase = "warmup" if i < warmup else "round"
        idx = i - warmup + 1 if phase == "round" else i + 1
        tag = f"{phase} {idx}/{warmup if phase == 'warmup' else rounds}"
        print(f"\n  [{tag}] {label}...", flush=True)
        r = provider.run(prompt, timeout=timeout)
        annotate_cost(r)
        out.append(r)
        if r.success:
            print(f"  done - TTFT {r.ttft_ms:.0f}ms | TPS {r.tokens_per_second:.1f} | "
                  f"P50 ITL {r.p50_itl_ms:.0f}ms | cost ${r.cost_usd:.6f}")
        else:
            print(f"  FAIL: {r.error}")
    return out


def _summarize_rounds(
    round_results: List[BenchmarkResult],
    warmup: int = 0,
) -> BenchmarkResult:
    """Pick the median-latency successful round; attach cold/hot/steady stats.

    If ``warmup > 0`` the first ``warmup`` entries are dropped from the
    measurement set. The first *measured* round is reported as
    ``cold_start_*``; the rest's median TPS is reported as
    ``steady_state_tps`` (stored on the result's private ``_all`` attr).
    """
    measured = round_results[warmup:] if warmup else round_results
    valid = [r for r in measured if r.success]
    if valid:
        valid_sorted = sorted(valid, key=lambda r: r.total_latency_ms)
        med = valid_sorted[len(valid_sorted) // 2]
        med._all = round_results     # type: ignore[attr-defined]
        med._warmup = warmup         # type: ignore[attr-defined]

        # cold vs steady-state
        if valid:
            cold = valid[0]
            med.cold_start_ms = cold.total_latency_ms          # type: ignore[attr-defined]
            med.cold_ttft_ms = cold.ttft_ms                     # type: ignore[attr-defined]
            if len(valid) >= 2:
                steady = valid[1:]
                med.steady_state_tps = statistics.median(      # type: ignore[attr-defined]
                    s.tokens_per_second for s in steady
                )
                med.steady_p95_itl_ms = statistics.median(      # type: ignore[attr-defined]
                    s.p95_itl_ms for s in steady
                )
            else:
                med.steady_state_tps = med.tokens_per_second    # type: ignore[attr-defined]
                med.steady_p95_itl_ms = med.p95_itl_ms          # type: ignore[attr-defined]
        return med
    return round_results[0] if round_results else BenchmarkResult(
        provider="?", model="?", success=False,
        error="no rounds executed",
    )


# ---------------------------------------------------------------------------
# Public entry — new on-boarding path
# ---------------------------------------------------------------------------

def run_from_endpoints(
    endpoints: List[EndpointConfig],
    *,
    rounds: int = 3,
    prompt: str = "Hello, world!",
    timeout: int = 60,
    verbose: bool = False,
    warmup: int = 1,
) -> List[BenchmarkResult]:
    """Run the benchmark against each model in each endpoint config."""
    results: List[BenchmarkResult] = []
    for ep in endpoints:
        for model_id in ep.model_ids:
            provider = _build_universal(ep, model_id, verbose)
            label = f"{ep.owner}/{model_id}"
            rounds_out = _run_rounds(
                label=label, provider=provider,
                prompt=prompt, rounds=rounds, timeout=timeout,
                warmup=warmup,
            )
            results.append(_summarize_rounds(rounds_out, warmup=warmup))
    return results


# ---------------------------------------------------------------------------
# Public entry — legacy config.yaml path
# ---------------------------------------------------------------------------

def run_from_config(
    providers: List[str],
    *,
    rounds: int = 3,
    config_path: str = "config.yaml",
    verbose: bool = False,
    warmup: int = 1,
) -> List[BenchmarkResult]:
    config = load_config(config_path)
    benchmark_cfg = config.get("benchmark", {})
    prompt = benchmark_cfg.get("prompt", "Hello, world!")
    timeout = benchmark_cfg.get("timeout", 60)
    provider_cfgs = config.get("providers", {})

    results: List[BenchmarkResult] = []
    for target in providers:
        cfg = provider_cfgs.get(target, {}) or {}
        cls = PROVIDERS.get(target)
        if not cls:
            print(f"WARN: Provider '{target}' not implemented, skipping")
            continue
        api_key = cfg.get("api_key", "")
        if not api_key:
            print(f"WARN: No API key for '{target}', skipping")
            continue
        base_url = cfg.get("base_url") or None
        provider = cls(api_key=api_key, model=cfg.get("model", ""),
                       verbose=verbose, base_url=base_url)
        rounds_out = _run_rounds(
            label=f"{target}/{provider.model}", provider=provider,
            prompt=prompt, rounds=rounds, timeout=timeout, warmup=warmup,
        )
        results.append(_summarize_rounds(rounds_out, warmup=warmup))
    return results


# ---------------------------------------------------------------------------
# Backwards-compat shim — keep ``run_benchmark`` working for old callers
# ---------------------------------------------------------------------------

def run_benchmark(
    providers: Optional[List[str]] = None,
    rounds: int = 3,
    config_path: str = "config.yaml",
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """Legacy entry — same signature as the original runner."""
    return run_from_config(
        providers=providers or [],
        rounds=rounds,
        config_path=config_path,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Export — JSON / CSV for downstream analysis
# ---------------------------------------------------------------------------

def export_results(
    results: List[BenchmarkResult],
    output_path: str,
    fmt: Optional[str] = None,
) -> None:
    """Write benchmark results to disk. ``fmt`` inferred from path extension.

    JSON output includes the full per-round history (raw ITL samples and
    throughput curves are kept lightweight — only summary statistics).
    """
    if fmt is None:
        ext = os.path.splitext(output_path)[1].lower().lstrip(".")
        fmt = ext if ext in ("json", "csv") else "json"

    rows = []
    for r in results:
        row = r.to_dict()
        # attach cold/hot extras if the runner populated them
        for attr in ("cold_start_ms", "cold_ttft_ms",
                     "steady_state_tps", "steady_p95_itl_ms"):
            if hasattr(r, attr):
                row[attr] = getattr(r, attr)
        # include ITL summary stats (not raw samples — too noisy for CSV)
        row["itl_p50_ms"] = round(r.p50_itl_ms, 2)
        row["itl_p95_ms"] = round(r.p95_itl_ms, 2)
        row["itl_p99_ms"] = round(r.p99_itl_ms, 2)
        row["itl_max_ms"] = round(r.max_itl_ms, 2)
        row["tpot_ms"] = round(r.tpot_ms, 2)
        rows.append(row)

    if fmt == "json":
        with open(output_path, "w") as f:
            json.dump({
                "schema": "token-benchmark/v1",
                "results": rows,
            }, f, indent=2, default=str)
    else:  # csv
        if not rows:
            open(output_path, "w").close()
            return
        with open(output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
    print(f"  exported {len(rows)} result(s) -> {output_path}")
