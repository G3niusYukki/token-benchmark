"""Universal streaming provider — handles both OpenAI and Anthropic dialects.

All the differences between SDKs are concentrated in the two iterators
(:py:meth:`_iter_openai` and :py:meth:`_iter_anthropic`). They yield
``StreamEvent`` objects that carry text, per-event timestamps, and any
usage metadata the API exposed. The rest of the benchmark loop is
dialect-agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

from benchmark.endpoint_detector import EndpointStyle
from benchmark.models import BenchmarkResult
from benchmark.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Token counting — single source of truth
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Use tiktoken cl100k_base; fall back to a 4-char-per-token estimate."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Stream event — what the iterators yield
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One chunk off the wire."""
    text: str = ""
    received_at: float = 0.0          # time.perf_counter() at reception
    is_final: bool = False            # True only for the terminal sentinel
    # Optional usage info (typically only on the last chunk before is_final)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Raw chunk for callers that need extra fields
    raw: Any = None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class UniversalProvider(BaseProvider):
    """A single provider class that speaks both OpenAI and Anthropic.

    Construction parameters
    -----------------------
    api_key        : str
    model          : str
    base_url       : str | None
    style          : EndpointStyle (OPENAI or ANTHROPIC)
    owner          : str
    verbose        : bool
    max_tokens     : int
    extra_headers  : dict
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        style: EndpointStyle = EndpointStyle.OPENAI,
        owner: str = "custom",
        verbose: bool = False,
        max_tokens: int = 1024,
        extra_headers: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url
        self.style = style
        self.owner = owner or "custom"
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.name = self.owner
        self._openai_client = None
        self._anthropic_client = None

    # ------------------------------------------------------------------
    # SDK construction (lazy)
    # ------------------------------------------------------------------
    def _openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.api_key or "sk-no-key",
                base_url=self.base_url,
                default_headers=self.extra_headers or None,
            )
        return self._openai_client

    def _anthropic(self):
        if self._anthropic_client is None:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.extra_headers or None,
            )
        return self._anthropic_client

    # ------------------------------------------------------------------
    # Streaming primitives
    # ------------------------------------------------------------------
    def _iter_openai(self, prompt: str, timeout: int) -> Iterator[StreamEvent]:
        """Yield StreamEvents from an OpenAI Chat Completions stream.

        Captures usage data when ``stream_options.include_usage=True`` is
        honored by the upstream provider (OpenAI, DeepSeek, vLLM, etc.).
        """
        client = self._openai()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=timeout,
                stream_options={"include_usage": True},
            )
        except TypeError:
            # Older SDKs / proxies that don't accept stream_options
            stream = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=timeout,
            )

        prompt_tokens = 0
        completion_tokens = 0
        for chunk in stream:
            received_at = time.perf_counter()
            text = ""
            try:
                if chunk.choices:
                    text = chunk.choices[0].delta.content or ""
            except Exception:
                text = ""
            # usage shows up on the final chunk (choices=[]) when the server
            # supports include_usage
            try:
                usage = getattr(chunk, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                    completion_tokens = getattr(usage, "completion_tokens", completion_tokens) or completion_tokens
            except Exception:
                pass
            yield StreamEvent(
                text=text,
                received_at=received_at,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw=chunk,
            )
        # terminal sentinel carries final usage
        yield StreamEvent(
            received_at=time.perf_counter(),
            is_final=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _iter_anthropic(self, prompt: str) -> Iterator[StreamEvent]:
        """Yield StreamEvents from an Anthropic Messages stream.

        Anthropic's SDK exposes ``stream.usage`` after the context manager
        exits, so we accumulate the input/output token counts and emit them
        on the final sentinel.
        """
        manager = self._anthropic().messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        prompt_tokens = 0
        completion_tokens = 0
        with manager as stream:
            for text in stream.text_stream:
                yield StreamEvent(text=text or "", received_at=time.perf_counter())
            try:
                usage = getattr(stream, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                    completion_tokens = getattr(usage, "output_tokens", 0) or 0
            except Exception:
                pass
        yield StreamEvent(
            received_at=time.perf_counter(),
            is_final=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------------
    # Public benchmark method
    # ------------------------------------------------------------------
    def run(self, prompt: str, timeout: int = 60) -> BenchmarkResult:
        live = None
        try:
            from rich.live import Live
            from rich.panel import Panel

            t0 = time.perf_counter()
            ttft: Optional[float] = None
            t_last = t0
            prev_t = t0
            full_text = ""
            itl_ms: List[float] = []
            prompt_tokens = 0
            completion_tokens = 0
            console_width = 100

            def make_status():
                el = time.perf_counter() - t0
                tokens = _count_tokens(full_text)
                tps = tokens / el if el > 0 else 0
                p50 = _safe_pctl(itl_ms, 50)
                p95 = _safe_pctl(itl_ms, 95)
                preview = full_text
                if len(preview) > 400:
                    preview = preview[:200] + " … " + preview[-200:]
                return Panel(
                    f"[bold cyan]TTFT:[/bold cyan] {(ttft or 0):.0f}ms  "
                    f"[bold green]Tokens:[/bold green] {tokens:>6}  "
                    f"[bold yellow]TPS:[/bold yellow] {tps:.1f}  "
                    f"[bold magenta]P50 ITL:[/bold magenta] {p50:.0f}ms  "
                    f"[bold magenta]P95 ITL:[/bold magenta] {p95:.0f}ms\n"
                    f"[dim]{preview}[/dim]",
                    title=f"[bold]{self.name}/{self.model}[/bold] Streaming",
                    border_style="cyan",
                    width=console_width,
                )

            if self.verbose:
                try:
                    from rich.console import Console as _C
                    console_width = max(80, _C().width or 100)
                except Exception:
                    pass
                live = Live(make_status(), refresh_per_second=10, transient=False)
                live.start()

            iterator = (
                self._iter_anthropic(prompt)
                if self.style == EndpointStyle.ANTHROPIC
                else self._iter_openai(prompt, timeout)
            )

            try:
                for ev in iterator:
                    if not ev.is_final:
                        t_last = ev.received_at or time.perf_counter()
                        if ev.text:
                            if ttft is None:
                                ttft = (t_last - t0) * 1000
                            else:
                                # inter-token latency: gap since previous token
                                # previous = t0 + sum(prior itl + ttft) — easier:
                                # gap from the most recent prior event
                                gap_ms = (t_last - prev_t) * 1000
                                if gap_ms >= 0:
                                    itl_ms.append(gap_ms)
                            full_text += ev.text
                            prev_t = t_last
                        # carry forward last-known usage if any
                        if ev.prompt_tokens:
                            prompt_tokens = ev.prompt_tokens
                        if ev.completion_tokens:
                            completion_tokens = ev.completion_tokens
                    else:
                        # terminal event — capture final usage
                        if ev.prompt_tokens:
                            prompt_tokens = ev.prompt_tokens
                        if ev.completion_tokens:
                            completion_tokens = ev.completion_tokens
                    if live:
                        live.update(make_status())
            finally:
                if live:
                    live.stop()

            # Establish prev_t reference: if we never entered the text branch
            # we still want ttft to be set on first text (already done above).
            if ttft is None:
                ttft = (t_last - t0) * 1000

            total_tokens = _count_tokens(full_text)
            streaming_time = max(t_last - t0, 1e-6)

            result = BenchmarkResult(
                provider=self.name,
                model=self.model,
                success=True,
                total_tokens=total_tokens,
                ttft_ms=ttft,
                total_latency_ms=streaming_time * 1000,
                tokens_per_second=self._calc_tps(total_tokens, streaming_time),
                itl_ms=itl_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens or total_tokens,
                full_text=full_text,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            result.finalize()
            return result
        except Exception as e:
            if live:
                try:
                    live.stop()
                except Exception:
                    pass
            return BenchmarkResult(
                provider=self.name,
                model=self.model,
                success=False,
                error=f"{type(e).__name__}: {e}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )


def _safe_pctl(values: List[float], pct: float) -> float:
    """Inline percentile used in the live display panel."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)
