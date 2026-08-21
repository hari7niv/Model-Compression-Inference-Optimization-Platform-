import argparse
import os

from src.benchmark import run_benchmark
from src.ConfigLoader import ConfigLoader
from src.ParetoFrontier import (
    build_comparison_table,
    generate_recommendations,
    pareto_frontier,
)
from src.plot import plot_pareto


def main():
    parser = argparse.ArgumentParser(prog="optimize")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark for a given config")
    bench_parser.add_argument("--config", required=True)

    report_parser = subparsers.add_parser("report", help="Build comparison table, Pareto analysis, and recommendations")
    report_parser.add_argument("--configs-dir", default="configs")

    args = parser.parse_args()

    if args.command == "benchmark":
        config = ConfigLoader.load(args.config)
        results = run_benchmark(config)
        

    elif args.command == "report":
        # Gather (technique, json_path) pairs from every config in configs_dir
        result_files = []
        for fname in os.listdir(args.configs_dir):
            if not fname.endswith(".yaml"):
                continue
            cfg = ConfigLoader.load(os.path.join(args.configs_dir, fname))
            technique = cfg["model"].get("quantization") or "fp16"
            json_path = os.path.join(cfg["output"]["directory"], cfg["output"]["filename"])
            result_files.append((technique, json_path))

        rows = build_comparison_table(result_files)
        metrics = ["p50_latency_sec", "peak_memory_mb", "perplexity"]
        frontier = pareto_frontier(rows, metrics)

        print("Pareto-optimal configs:")
        for r in frontier:
            print(f"  {r['technique']}")

        print(generate_recommendations(rows))

        frontier_techniques = [r["technique"] for r in frontier]
        plot_pareto(rows, frontier_techniques)


if __name__ == "__main__":
    main()