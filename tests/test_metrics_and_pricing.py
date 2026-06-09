"""Tests for the new production-grade metrics + pricing + export."""
import csv
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/Users/peterzhang/token-benchmark")

from benchmark.models import BenchmarkResult, _percentile
from benchmark.pricing import annotate, estimate_cost, lookup
from benchmark.runner import export_results


# ----- percentile helper ---------------------------------------------------

def test_percentile_empty():
    assert _percentile([], 50) == 0.0
    assert _percentile([], 95) == 0.0


def test_percentile_single():
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 95) == 42.0


def test_percentile_basic():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert abs(_percentile(vals, 50) - 30.0) < 0.01
    assert abs(_percentile(vals, 95) - 48.0) < 0.5
    assert abs(_percentile(vals, 99) - 49.6) < 0.5


# ----- BenchmarkResult.finalize() ------------------------------------------

def test_finalize_computes_itl_percentiles():
    r = BenchmarkResult(
        provider="x", model="y", success=True,
        itl_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        total_tokens=5, ttft_ms=100, total_latency_ms=300,
    )
    r.finalize()
    assert r.p50_itl_ms == 30.0
    assert r.max_itl_ms == 50.0
    # 100ms TTFT + 200ms generation over 4 inter-token gaps = 50ms TPOT
    assert abs(r.tpot_ms - 50.0) < 0.01


def test_finalize_populates_throughput_curve():
    r = BenchmarkResult(
        provider="x", model="y", success=True,
        itl_ms=[10.0, 10.0, 10.0, 10.0],
        total_tokens=5, ttft_ms=0, total_latency_ms=40,
    )
    r.finalize()
    assert len(r.throughput_curve) == 4
    # Each point: (cumulative_ms, cumulative_tps)
    # Step 1: 10ms / 1 token = 100 tps
    assert r.throughput_curve[0][1] == 100.0


def test_to_dict_excludes_heavy_fields():
    r = BenchmarkResult(
        provider="x", model="y", success=True,
        total_tokens=100, ttft_ms=50, total_latency_ms=1000,
        tokens_per_second=100, prompt_tokens=20, completion_tokens=100,
        cost_usd=0.001234, itl_ms=[10, 20, 30],
    )
    r.finalize()
    d = r.to_dict()
    assert d["provider"] == "x"
    assert d["total_tokens"] == 100
    assert d["p50_itl_ms"] == 20.0
    assert d["prompt_tokens"] == 20
    assert d["completion_tokens"] == 100
    assert d["cost_usd"] == 0.001234
    # ITL raw samples NOT in dict (too noisy for CSV)
    assert "itl_ms" not in d
    assert "itl_samples" in d


# ----- pricing ------------------------------------------------------------

def test_lookup_known_models():
    assert lookup("gpt-4o") == (2.50, 10.00)
    assert lookup("claude-sonnet-4-5-latest") == (3.00, 15.00)
    assert lookup("deepseek-chat") == (0.27, 1.10)


def test_lookup_strips_date_suffix():
    # OpenAI sometimes returns dated model IDs
    assert lookup("gpt-4o-2024-08-06") == (2.50, 10.00)
    assert lookup("claude-3-5-sonnet-20241022") == (3.00, 15.00)


def test_lookup_unknown_returns_none():
    assert lookup("totally-made-up-model") is None
    assert lookup("") is None


def test_estimate_cost_basic():
    # 1M input at $2.50/M + 1M output at $10/M = $12.50
    cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert abs(cost - 12.50) < 0.01


def test_estimate_cost_zero_for_unknown():
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_annotate_attaches_cost_to_result():
    r = BenchmarkResult(
        provider="openai", model="gpt-4o", success=True,
        prompt_tokens=1000, completion_tokens=2000,
    )
    cost = annotate(r)
    # 1000 * 2.50/1M + 2000 * 10/1M = 0.0025 + 0.02 = 0.0225
    assert abs(cost - 0.0225) < 0.0001
    assert abs(r.cost_usd - 0.0225) < 0.0001


def test_pricing_env_override(monkeypatch):
    monkeypatch.setenv(
        "TOKEN_BENCHMARK_PRICING_OVERRIDE",
        json.dumps({"custom-model": {"input": 1.0, "output": 2.0}}),
    )
    import importlib
    import benchmark.pricing
    importlib.reload(benchmark.pricing)
    assert benchmark.pricing.lookup("custom-model") == (1.0, 2.0)
    monkeypatch.delenv("TOKEN_BENCHMARK_PRICING_OVERRIDE", raising=False)
    importlib.reload(benchmark.pricing)


# ----- export_results -----------------------------------------------------

def _make_results():
    r1 = BenchmarkResult(
        provider="openai", model="gpt-4o", success=True,
        total_tokens=100, ttft_ms=200, total_latency_ms=2000,
        tokens_per_second=50, p50_itl_ms=15, p95_itl_ms=40,
        prompt_tokens=20, completion_tokens=100, cost_usd=0.001,
    )
    r1.finalize()
    r2 = BenchmarkResult(
        provider="anthropic", model="claude-sonnet-4-5-latest", success=True,
        total_tokens=80, ttft_ms=300, total_latency_ms=2500,
        tokens_per_second=32, p50_itl_ms=20, p95_itl_ms=55,
        prompt_tokens=20, completion_tokens=80, cost_usd=0.002,
    )
    r2.finalize()
    return [r1, r2]


def test_export_json(tmp_path):
    results = _make_results()
    out = str(tmp_path / "results.json")
    export_results(results, out)
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert data["schema"] == "token-benchmark/v1"
    assert len(data["results"]) == 2
    assert data["results"][0]["provider"] == "openai"
    assert data["results"][0]["p50_itl_ms"] == 15.0
    assert "itl_p50_ms" in data["results"][0]


def test_export_csv(tmp_path):
    results = _make_results()
    out = str(tmp_path / "results.csv")
    export_results(results, out)
    assert os.path.exists(out)
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["provider"] == "openai"
    assert float(rows[0]["p50_itl_ms"]) == 15.0
    assert float(rows[0]["cost_usd"]) == 0.001


def test_export_handles_empty():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        export_results([], tf.name)
        with open(tf.name) as f:
            data = json.load(f)
        assert data["results"] == []
        os.unlink(tf.name)
