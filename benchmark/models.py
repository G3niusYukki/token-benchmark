"""Result data class for a single benchmark round.

One :class:`BenchmarkResult` corresponds to one streaming request — not
one model. The runner produces N of these per model (one per round) and
then picks a representative. Anything that wants per-round raw data can
attach it via the ``_all`` private attr (set by the runner).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile. Returns 0 for an empty list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class BenchmarkResult:
    """All metrics captured for one streaming request.

    Fields are organized into three groups:

    Identity
        provider, model, success, error, full_text, timestamp
    Core timing
        ttft_ms, total_latency_ms, tokens_per_second, completion_tokens
    Production-grade
        itl_ms (inter-token latency list)
        p50_itl_ms, p95_itl_ms, p99_itl_ms, max_itl_ms
        tpot_ms (time per output token)
        prompt_tokens, completion_tokens (from API usage)
        throughput_curve (TPS samples over the response)
        cost_usd (computed from pricing table)
    """
    # identity
    provider: str
    model: str
    success: bool
    # core timing
    total_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    # production-grade timing
    itl_ms: List[float] = field(default_factory=list)
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    max_itl_ms: float = 0.0
    tpot_ms: float = 0.0
    # token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # throughput curve — list of (elapsed_ms, tps) snapshots
    throughput_curve: List[List[float]] = field(default_factory=list)
    # cost (computed externally from pricing table)
    cost_usd: float = 0.0
    # error / raw
    error: Optional[str] = None
    full_text: Optional[str] = None
    # metadata
    timestamp: Optional[str] = None  # ISO-8601 UTC

    def _calc_tps(self, token_count: int, elapsed: float) -> float:
        return token_count / elapsed if elapsed and elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # Post-processing — call once after the streaming loop finishes
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Compute derived metrics (P50/P95/ITL, TPOT, throughput curve)."""
        if self.itl_ms:
            self.p50_itl_ms = _percentile(self.itl_ms, 50)
            self.p95_itl_ms = _percentile(self.itl_ms, 95)
            self.p99_itl_ms = _percentile(self.itl_ms, 99)
            self.max_itl_ms = max(self.itl_ms)

        # TPOT = (total_latency - TTFT) / (completion_tokens - 1)
        # i.e. average time spent producing each token AFTER the first.
        if self.completion_tokens > 1 and self.ttft_ms > 0:
            generation_ms = max(self.total_latency_ms - self.ttft_ms, 1e-6)
            self.tpot_ms = generation_ms / (self.completion_tokens - 1)
        elif self.total_tokens > 1 and self.ttft_ms > 0:
            generation_ms = max(self.total_latency_ms - self.ttft_ms, 1e-6)
            self.tpot_ms = generation_ms / (self.total_tokens - 1)

        # Build throughput curve from ITL samples: cumulative TPS at each step
        if self.itl_ms:
            curve: list[list[float]] = []
            cumulative_tokens = 0
            cumulative_ms = 0.0
            for i, itl in enumerate(self.itl_ms, start=1):
                cumulative_ms += itl
                cumulative_tokens = i
                tps = (cumulative_tokens * 1000.0 / cumulative_ms) if cumulative_ms > 0 else 0.0
                curve.append([round(cumulative_ms, 2), round(tps, 2)])
            self.throughput_curve = curve

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        if not self.success:
            return f"{status} {self.provider}/{self.model}: {self.error}"
        ttft = f"TTFT: {self.ttft_ms:.0f}ms" if self.ttft_ms is not None else "TTFT: -"
        return (
            f"{status} {self.provider}/{self.model} — "
            f"{ttft} | "
            f"Tokens/s: {self.tokens_per_second:.1f} | "
            f"P50 ITL: {self.p50_itl_ms:.0f}ms | "
            f"P95 ITL: {self.p95_itl_ms:.0f}ms | "
            f"Total: {self.total_latency_ms:.0f}ms | "
            f"Tokens: {self.total_tokens} | "
            f"Cost: ${self.cost_usd:.5f}"
        )

    def calc_breakdown(self) -> str:
        """Multi-line human-readable breakdown of how each metric was derived."""
        t = self.total_latency_ms / 1000
        return (
            f"  计算: {self.total_tokens} tokens / {t:.2f}s = {self.tokens_per_second:.1f} tokens/s\n"
            f"  TTFT: {self.ttft_ms:.0f}ms (首 token 延迟)\n"
            f"  P50/P95/P99 ITL: {self.p50_itl_ms:.0f}/{self.p95_itl_ms:.0f}/{self.p99_itl_ms:.0f}ms\n"
            f"  TPOT: {self.tpot_ms:.1f}ms (avg per-token after first)\n"
            f"  Prompt tokens: {self.prompt_tokens} | Completion: {self.completion_tokens}\n"
            f"  Cost: ${self.cost_usd:.6f}\n"
            f"  总耗时: {self.total_latency_ms:.0f}ms\n"
            f"  响应长度: {len(self.full_text or '')} 字符"
        )

    def to_dict(self) -> dict:
        """Flat dict suitable for JSON / CSV export."""
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "error": self.error,
            "total_tokens": self.total_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "p50_itl_ms": round(self.p50_itl_ms, 2),
            "p95_itl_ms": round(self.p95_itl_ms, 2),
            "p99_itl_ms": round(self.p99_itl_ms, 2),
            "max_itl_ms": round(self.max_itl_ms, 2),
            "tpot_ms": round(self.tpot_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "itl_samples": len(self.itl_ms),
            "timestamp": self.timestamp,
        }
