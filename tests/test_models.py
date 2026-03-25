import sys
sys.path.insert(0, "/Users/peterzhang/token-benchmark")

from benchmark.models import BenchmarkResult

def test_benchmark_result_summary_success():
    r = BenchmarkResult(
        provider="openai", model="gpt-4o",
        total_tokens=100, ttft_ms=500, total_latency_ms=2000,
        tokens_per_second=50, success=True,
    )
    assert "openai" in r.summary()
    assert "50.0" in r.summary()
    assert "✅" in r.summary()

def test_benchmark_result_failure():
    r = BenchmarkResult(
        provider="openai", model="gpt-4o",
        total_tokens=0, ttft_ms=0, total_latency_ms=0,
        tokens_per_second=0, success=False, error="timeout",
    )
    assert "❌" in r.summary()
    assert "timeout" in r.summary()

def test_calc_tps():
    r = BenchmarkResult(
        provider="test", model="test",
        total_tokens=100, ttft_ms=100, total_latency_ms=2000,
        tokens_per_second=0, success=True,
    )
    assert r._calc_tps(100, 2.0) == 50.0
    assert r._calc_tps(0, 0.0) == 0.0
