"""Per-model token pricing for cost calculation.

All prices are USD per 1,000,000 tokens (the unit both OpenAI and Anthropic
publish in).  Numbers are best-effort snapshots — verify against the
provider's pricing page before using for billing decisions.

Override at runtime with the ``$TOKEN_BENCHMARK_PRICING_OVERRIDE`` env
var, which should be JSON: ``{"<model_id>": {"input": X, "output": Y}}``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple


# (input_per_1m, output_per_1m) in USD
_PRICING: dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (2.50, 10.00),
    "gpt-4o-mini":          (0.15, 0.60),
    "gpt-4-turbo":          (10.00, 30.00),
    "gpt-4":                (30.00, 60.00),
    "gpt-3.5-turbo":        (0.50, 1.50),
    "o1":                   (15.00, 60.00),
    "o1-mini":              (3.00, 12.00),
    "o1-preview":           (15.00, 60.00),
    "o3-mini":              (1.10, 4.40),
    # Anthropic
    "claude-opus-4-1-latest":    (15.00, 75.00),
    "claude-opus-4-latest":      (15.00, 75.00),
    "claude-sonnet-4-5-latest":   (3.00, 15.00),
    "claude-sonnet-4-latest":     (3.00, 15.00),
    "claude-3-7-sonnet-latest":   (3.00, 15.00),
    "claude-3-5-sonnet-latest":   (3.00, 15.00),
    "claude-3-5-haiku-latest":    (0.80, 4.00),
    "claude-3-opus-latest":       (15.00, 75.00),
    "claude-3-haiku-20240307":    (0.25, 1.25),
    # DeepSeek (input is cache-miss; cached input is much cheaper but we ignore that)
    "deepseek-chat":         (0.27, 1.10),
    "deepseek-coder":        (0.27, 1.10),
    "deepseek-reasoner":     (0.55, 2.19),
    # Moonshot / Kimi
    "moonshot-v1-8k":        (2.00, 2.00),
    "moonshot-v1-32k":       (2.00, 2.00),
    "moonshot-v1-128k":      (2.00, 2.00),
}


def _load_override() -> dict[str, Tuple[float, float]]:
    raw = os.environ.get("TOKEN_BENCHMARK_PRICING_OVERRIDE", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[pricing] could not parse TOKEN_BENCHMARK_PRICING_OVERRIDE: {e}")
        return {}
    out: dict[str, Tuple[float, float]] = {}
    for k, v in data.items():
        try:
            out[k] = (float(v["input"]), float(v["output"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


_OVERRIDE: dict[str, Tuple[float, float]] = _load_override()


def lookup(model_id: str) -> Optional[Tuple[float, float]]:
    """Return (input_per_1m, output_per_1m) for a model, or None if unknown.

    Strips date/version suffixes (``-2024-08-06``, ``-20241022``,
    ``-latest``) for fuzzy match against the static table.
    """
    if not model_id:
        return None
    mid = model_id.strip().lower()
    if mid in _OVERRIDE:
        return _OVERRIDE[mid]
    if mid in _PRICING:
        return _PRICING[mid]
    # try fuzzy match — progressively strip date/version suffixes
    candidates = [mid]
    for pattern in (
        r"-\d{4}-\d{2}-\d{2}$",   # -2024-08-06
        r"-\d{8}$",                # -20241022
        r"-latest$",               # -latest
        r"-\d{4,}$",               # bare 4+ digit suffix
    ):
        stripped = re.sub(pattern, "", mid)
        if stripped != mid:
            candidates.append(stripped)
    for c in candidates:
        if c in _OVERRIDE:
            return _OVERRIDE[c]
        if c in _PRICING:
            return _PRICING[c]
    # last resort: substring match — pick the longest registered key
    # that's a PREFIX of any candidate. This handles "claude-3-5-sonnet"
    # matching the "claude-3-5-sonnet-latest" entry.
    best_key = None
    for c in candidates:
        for k in list(_PRICING) + list(_OVERRIDE):
            if k.startswith(c) and (best_key is None or len(k) > len(best_key)):
                best_key = k
    if best_key:
        return _OVERRIDE.get(best_key) or _PRICING[best_key]
    return None


def estimate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return USD cost for the given token counts. 0 if pricing unknown."""
    price = lookup(model_id)
    if not price:
        return 0.0
    in_per_m, out_per_m = price
    return (prompt_tokens / 1_000_000.0) * in_per_m + (completion_tokens / 1_000_000.0) * out_per_m


def annotate(result, model_id: Optional[str] = None) -> float:
    """Compute and attach cost_usd to a BenchmarkResult. Returns the cost."""
    from benchmark.models import BenchmarkResult  # avoid circular at import time
    if not isinstance(result, BenchmarkResult):
        return 0.0
    mid = model_id or result.model
    cost = estimate_cost(mid, result.prompt_tokens, result.completion_tokens)
    result.cost_usd = cost
    return cost
