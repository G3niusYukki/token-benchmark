import os
import time
import yaml
from typing import List, Optional
from benchmark.models import BenchmarkResult
from benchmark.providers import PROVIDERS

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return {"providers": {}, "benchmark": {}}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    # 替换环境变量
    def replace_env(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], "")
        return v
    def walk(d):
        if isinstance(d, dict):
            return {k: walk(replace_env(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [walk(i) for i in d]
        return replace_env(d)
    return walk(raw)

def interactive_config(provider_name: str, default_model: str) -> Optional[dict]:
    """交互式配置 provider，返回 None 表示跳过"""
    print(f"\n📝 Configuring {provider_name}")
    api_key = input(f"   API Key: ").strip()
    if not api_key:
        print(f"   ⏭️  Skipped")
        return None

    model = input(f"   Model [{default_model}]: ").strip() or default_model

    # OpenAI兼容的 providers 支持自定义 endpoint
    base_url = None
    if provider_name in ("openai", "deepseek", "kimi", "custom"):
        url = input(f"   Base URL (optional, press Enter for default): ").strip()
        base_url = url or None

    return {"api_key": api_key, "model": model, "base_url": base_url}

def run_benchmark(
    providers: List[str] = None,
    rounds: int = 3,
    config_path: str = "config.yaml",
    verbose: bool = False,
) -> List[BenchmarkResult]:
    config = load_config(config_path)
    benchmark_cfg = config.get("benchmark", {})
    prompt = benchmark_cfg.get("prompt", "Hello, world!")
    timeout = benchmark_cfg.get("timeout", 60)
    provider_cfgs = config.get("providers", {})

    targets = providers or list(provider_cfgs.keys())
    results = []

    for target in targets:
        cfg = provider_cfgs.get(target)
        cls = PROVIDERS.get(target)
        base_url = None

        # 没有配置，尝试交互式输入
        if not cfg or not cfg.get("api_key"):
            default_model = cfg.get("model", "gpt-4o") if cfg else "gpt-4o"
            interactive = interactive_config(target, default_model)
            if not interactive:
                continue
            cfg = interactive
            base_url = cfg.get("base_url")
            if base_url and target not in ("openai", "deepseek", "kimi"):
                # 通用 OpenAI-compatible provider
                from benchmark.providers.base import BaseProvider
                def _ct(text):
                    try:
                        import tiktoken
                        return len(tiktoken.get_encoding("cl100k_base").encode(text))
                    except Exception:
                        return max(1, len(text) // 4)

                class CustomProvider(BaseProvider):
                    name = target
                    def __init__(self, api_key, model, verbose=False, **kw):
                        super().__init__(api_key, model, **kw)
                        self.verbose = verbose
                    def run(self, prompt, timeout=60):
                        from openai import OpenAI
                        from rich.live import Live
                        from rich.panel import Panel
                        from rich.console import Console

                        client = OpenAI(api_key=self.api_key, base_url=self.extra.get("base_url"))
                        t0 = time.perf_counter()
                        ttft = None
                        t_last = t0
                        full_text = ""
                        live = None

                        def make_status():
                            el = time.perf_counter() - t0
                            tps = _ct(full_text) / el if el > 0 else 0
                            return Panel(
                                f"[bold cyan]TTFT:[/bold cyan] {ttft:.0f}ms  "
                                f"[bold green]Tokens:[/bold green] {_ct(full_text):>6}  "
                                f"[bold yellow]TPS:[/bold yellow] {tps:.1f}\n"
                                f"[dim]{full_text}[/dim]",
                                title=f"[bold]{self.name}/{self.model}[/bold] Streaming",
                                border_style="cyan", width=100,
                            )

                        if self.verbose:
                            console = Console()
                            live = Live(make_status(), refresh_per_second=10, transient=False)
                            live.start()
                        try:
                            stream = client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                stream=True, timeout=timeout,
                            )
                            for chunk in stream:
                                t_last = time.perf_counter()
                                content = chunk.choices[0].delta.content or ""
                                if content:
                                    if ttft is None:
                                        ttft = (t_last - t0) * 1000
                                    full_text += content
                                if live and full_text:
                                    live.update(make_status())
                        finally:
                            if live:
                                live.stop()

                        total = _ct(full_text)
                        st = t_last - t0
                        return BenchmarkResult(
                            provider=self.name, model=self.model,
                            total_tokens=total, ttft_ms=ttft or 0,
                            total_latency_ms=st * 1000,
                            tokens_per_second=total / st if st > 0 else 0,
                            success=True, full_text=full_text,
                        )
                cls = CustomProvider
        elif not cls:
            print(f"⚠️  Provider '{target}' not implemented, skipping")
            continue
        else:
            base_url = cfg.get("base_url") if cfg else None

        api_key = cfg.get("api_key", "")
        if not api_key:
            print(f"⚠️  No API key for '{target}', skipping")
            continue

        provider = cls(api_key=api_key, model=cfg.get("model", ""), verbose=verbose, base_url=base_url)
        round_results = []

        for i in range(rounds):
            print(f"\n  [{i+1}/{rounds}] {target}...", flush=True)
            r = provider.run(prompt, timeout=timeout)
            round_results.append(r)

            if r.success:
                if verbose:
                    print(f"  ✅ 完成  TTFT: {r.ttft_ms:.0f}ms | Tokens: {r.total_tokens} | "
                          f"TPS: {r.tokens_per_second:.1f} | 耗时: {r.total_latency_ms:.0f}ms")
                    print(f"  📝 响应预览: {r.full_text[:200]}{'...' if len(r.full_text or '') > 200 else ''}")
                    print(f"  🔢 计算: {r.total_tokens} tokens / {r.total_latency_ms/1000:.2f}s = {r.tokens_per_second:.1f} tokens/s")
                else:
                    print(f"  done")
            else:
                print(f"  FAIL: {r.error}")

        valid = [r for r in round_results if r.success]
        if valid:
            med = valid[len(valid)//2]
            med._all = round_results
            results.append(med)
        else:
            results.append(round_results[0])

    return results
