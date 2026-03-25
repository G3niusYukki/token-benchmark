#!/usr/bin/env python3
import argparse
from benchmark.runner import run_benchmark
from benchmark.reporter import print_results, generate_html_report

def main():
    parser = argparse.ArgumentParser(description="AI Token Speed Benchmark")
    parser.add_argument("-p", "--providers", nargs="+", help="Providers to test (default: all)")
    parser.add_argument("-r", "--rounds", type=int, default=3, help="Rounds per provider (default: 3)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("-o", "--output", default="benchmark_report.html", help="HTML output path")
    args = parser.parse_args()

    print("🚀 Starting Token Benchmark...\n")
    results = run_benchmark(
        providers=args.providers,
        rounds=args.rounds,
        config_path=args.config,
    )

    print("\n" + "="*60)
    print_results(results)

    if args.html:
        generate_html_report(results, args.output)

if __name__ == "__main__":
    main()
