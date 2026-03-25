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

    def __init__(self, api_key: str, model: str, verbose: bool = False, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.verbose = verbose

    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        try:
            from rich.live import Live
            from rich.panel import Panel
            from rich.console import Console

            client = Anthropic(api_key=self.api_key)
            t0 = time.perf_counter()
            ttft = None
            t_last = t0
            full_text = ""
            live, console = None, None

            def make_status():
                elapsed = time.perf_counter() - t0
                tps = _count_tokens(full_text) / elapsed if elapsed > 0 else 0
                return Panel(
                    f"[bold cyan]TTFT:[/bold cyan] {(ttft or 0):.0f}ms  "
                    f"[bold green]Tokens:[/bold green] {_count_tokens(full_text):>6}  "
                    f"[bold yellow]TPS:[/bold yellow] {tps:.1f}\n"
                    f"[dim]{full_text}[/dim]",
                    title=f"[bold]{self.name}/{self.model}[/bold] Streaming",
                    border_style="cyan",
                    width=console.width if console and hasattr(console, 'width') else 100,
                )

            if self.verbose:
                console = Console()
                live = Live(make_status(), refresh_per_second=10, transient=False)
                live.start()

            try:
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
                        if live and full_text:
                            live.update(make_status())
            finally:
                if live:
                    live.stop()

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
                full_text=full_text,
            )
        except Exception as e:
            return BenchmarkResult(
                provider=self.name, model=self.model,
                total_tokens=0, ttft_ms=0, total_latency_ms=0,
                tokens_per_second=0, success=False, error=str(e)
            )
