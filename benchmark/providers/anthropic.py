import time
from anthropic import Anthropic
from benchmark.providers.base import BaseProvider
from benchmark.models import BenchmarkResult

class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        try:
            client = Anthropic(api_key=self.api_key)
            t0 = time.perf_counter()
            ttft = None
            total_tokens = 0

            with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    total_tokens += 1

            elapsed = time.perf_counter() - t0
            return BenchmarkResult(
                provider=self.name,
                model=self.model,
                total_tokens=total_tokens,
                ttft_ms=ttft or 0,
                total_latency_ms=elapsed * 1000,
                tokens_per_second=self._calc_tps(total_tokens, elapsed),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                provider=self.name, model=self.model,
                total_tokens=0, ttft_ms=0, total_latency_ms=0,
                tokens_per_second=0, success=False, error=str(e)
            )
