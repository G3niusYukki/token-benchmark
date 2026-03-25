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
                def _count_tokens(text):
                    try:
                        import tiktoken
                        enc = tiktoken.get_encoding("cl100k_base")
                        return len(enc.encode(text))
                    except Exception:
                        return max(1, len(text) // 4)

                class CustomProvider(BaseProvider):
                    name = target
                    def run(self, prompt, timeout=60):
                        from openai import OpenAI
                        client = OpenAI(api_key=self.api_key, base_url=self.extra.get("base_url"))
                        t0 = time.perf_counter()
                        ttft = None
                        t_last = t0
                        full_text = ""
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
                            total_tokens = _count_tokens(full_text)
                            streaming_time = t_last - t0
                            return BenchmarkResult(
                                provider=self.name, model=self.model,
                                total_tokens=total_tokens, ttft_ms=ttft or 0,
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
                cls = CustomProvider
        elif not cls:
            print(f"⚠️  Provider '{target}' not implemented, skipping")
            continue
        else:
            # 从已配置的 cfg 中取 base_url（config.yaml 预配置场景）
            base_url = cfg.get("base_url") if cfg else None

        api_key = cfg.get("api_key", "")
        if not api_key:
            print(f"⚠️  No API key for '{target}', skipping")
            continue

        provider = cls(api_key=api_key, model=cfg.get("model", ""), base_url=base_url)

        round_results = []
        for i in range(rounds):
            print(f"  [{i+1}/{rounds}] {target}...", end=" ", flush=True)
            r = provider.run(prompt, timeout=timeout)
            round_results.append(r)
            print("done" if r.success else f"FAIL: {r.error}")

        valid = [r for r in round_results if r.success]
        if valid:
            med = valid[len(valid)//2]
            med._all = round_results
            results.append(med)
        else:
            results.append(round_results[0])

    return results
