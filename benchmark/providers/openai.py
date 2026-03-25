import time
import tiktoken
from openai import OpenAI
from benchmark.providers.base import BaseProvider
from benchmark.models import BenchmarkResult

def _count_tokens(text: str) -> int:
    """使用 tiktoken 计算真实 token 数"""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # 回退: 按字符估算 (1 token ≈ 4 字符)
        return max(1, len(text) // 4)

class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)
        base_url = kwargs.get("base_url") or self.extra.get("base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        try:
            t0 = time.perf_counter()
            ttft = None
            t_last = t0
            full_text = ""

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=timeout,
            )

            for chunk in stream:
                t_last = time.perf_counter()
                content = chunk.choices[0].delta.content or ""
                if content:
                    if ttft is None:
                        ttft = (t_last - t0) * 1000
                    full_text += content

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
                provider=self.name,
                model=self.model,
                total_tokens=0, ttft_ms=0, total_latency_ms=0,
                tokens_per_second=0, success=False, error=str(e)
            )
