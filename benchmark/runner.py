import os
import yaml
from typing import List
from benchmark.models import BenchmarkResult
from benchmark.providers import PROVIDERS

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
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
        if not cfg:
            print(f"⚠️  Provider '{target}' not found in config")
            continue
        cls = PROVIDERS.get(target)
        if not cls:
            print(f"⚠️  Provider '{target}' not implemented")
            continue

        api_key = cfg.get("api_key", "")
        if not api_key:
            print(f"⚠️  No API key for '{target}', skipping")
            continue

        provider = cls(api_key=api_key, model=cfg.get("model", ""))
        round_results = []

        for i in range(rounds):
            print(f"  [{i+1}/{rounds}] {target}...", end=" ", flush=True)
            r = provider.run(prompt, timeout=timeout)
            round_results.append(r)
            print("done" if r.success else f"FAIL: {r.error}")

        # 取中位数
        valid = [r for r in round_results if r.success]
        if valid:
            med = valid[len(valid)//2]
            med._all = round_results
            results.append(med)
        else:
            results.append(round_results[0])

    return results
