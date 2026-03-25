import time
from openai import OpenAI
from benchmark.providers.base import BaseProvider
from benchmark.models import BenchmarkResult

class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        try:
            t0 = time.perf_counter()
            ttft = None
            total_tokens = 0

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=timeout,
            )

            for chunk in stream:
                if ttft is None and chunk.choices[0].delta.content:
                    ttft = (time.perf_counter() - t0) * 1000
                if chunk.choices[0].delta.content:
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
                provider=self.name,
                model=self.model,
                total_tokens=0, ttft_ms=0, total_latency_ms=0,
                tokens_per_second=0, success=False, error=str(e)
            )
