#!/usr/bin/env python3
import argparse
from benchmark.runner import run_benchmark
from benchmark.reporter import print_results, generate_html_report

def main():
    parser = argparse.ArgumentParser(description="AI Token Speed Benchmark")
    parser.add_argument("-p", "--providers", nargs="+",
                        help="Providers to test (default: all)")
    parser.add_argument("-r", "--rounds", type=int, default=3,
                        help="Rounds per provider (default: 3)")
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Config file path")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("-o", "--output", default="benchmark_report.html",
                        help="HTML output path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show live streaming output, token counts, and calculation breakdown")
    args = parser.parse_args()

    available = ["openai", "anthropic", "deepseek", "kimi"]
    targets = args.providers or available

    print("=" * 60)
    print("       🚀  AI Token Speed Benchmark")
    print("=" * 60)
    print(f"  Providers : {', '.join(targets)}")
    print(f"  Rounds    : {args.rounds}")
    print(f"  Verbose   : {'ON' if args.verbose else 'OFF'}")
    print(f"  HTML Out  : {args.output if args.html else 'disabled'}")
    print("=" * 60)
    print()

    results = run_benchmark(
        providers=targets,
        rounds=args.rounds,
        config_path=args.config,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print_results(results)

    if args.html:
        generate_html_report(results, args.output)

if __name__ == "__main__":
    main()
