import time
import tiktoken
from anthropic import Anthropic
from benchmark.providers.base import BaseProvider
from benchmark.models import BenchmarkResult

def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        try:
            client = Anthropic(api_key=self.api_key)
            t0 = time.perf_counter()
            ttft = None
            t_last = t0
            full_text = ""

            with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    t_last = time.perf_counter()
                    if ttft is None:
                        ttft = (t_last - t0) * 1000
                    full_text += text
                    t_last = time.perf_counter()

            total_tokens = _count_tokens(full_text)
            streaming_time = t_last - t0
            return BenchmarkResult(
                provider=self.name,
                model=self.model,
                total_tokens=total_tokens,
                ttft_ms=ttft or 0,
                total_latency_ms=streaming_time * 1000,
                tokens_per_second=self._calc_tps(total_tokens, streaming_time),
                success=True,
            )
        except Exception as e:
            return BenchmarkResult(
                provider=self.name, model=self.model,
                total_tokens=0, ttft_ms=0, total_latency_ms=0,
                tokens_per_second=0, success=False, error=str(e)
            )
