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
    full_text: Optional[str] = None   # 完整响应文本 (verbose 模式)

    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        if not self.success:
            return f"{status} {self.provider}/{self.model}: {self.error}"
        return (
            f"{status} {self.provider}/{self.model} — "
            f"TTFT: {self.ttft_ms:.0f}ms | " if self.ttft_ms is not None else "TTFT: - | "
            f"Tokens/s: {self.tokens_per_second:.1f} | "
            f"Total: {self.total_latency_ms:.0f}ms | "
            f"Tokens: {self.total_tokens}"
        )

    def calc_breakdown(self) -> str:
        """返回计算过程说明"""
        t = self.total_latency_ms / 1000
        return (
            f"  计算: {self.total_tokens} tokens / {t:.2f}s = {self.tokens_per_second:.1f} tokens/s\n"
            f"  TTFT: {self.ttft_ms:.0f}ms (首 token 延迟)\n"
            f"  总耗时: {self.total_latency_ms:.0f}ms\n"
            f"  响应长度: {len(self.full_text)} 字符"
        )
