from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkResult:
    provider: str
    model: str
    total_tokens: int
    ttft_ms: float          # Time to First Token (ms)
    total_latency_ms: float # Total response time (ms)
    tokens_per_second: float
    success: bool
    error: Optional[str] = None

    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        if not self.success:
            return f"{status} {self.provider}/{self.model}: {self.error}"
        return (
            f"{status} {self.provider}/{self.model} — "
            f"TTFT: {self.ttft_ms:.0f}ms | "
            f"Tokens/s: {self.tokens_per_second:.1f} | "
            f"Total: {self.total_latency_ms:.0f}ms | "
            f"Tokens: {self.total_tokens}"
        )

    def _calc_tps(self, tokens: int, time_s: float) -> float:
        if time_s <= 0:
            return 0.0
        return tokens / time_s
