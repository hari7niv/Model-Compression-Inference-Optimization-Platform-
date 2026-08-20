import csv
import json

def load_results(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # cast numeric fields
    for r in rows:
        for k in r:
            if k != "technique":
                r[k] = float(r[k])
    return rows

def dominates(a, b, metrics):
    """True if config a dominates config b (a is <= b on all metrics, < on at least one)."""
    at_least_as_good = all(a[m] <= b[m] for m in metrics)
    strictly_better = any(a[m] < b[m] for m in metrics)
    return at_least_as_good and strictly_better

def pareto_frontier(rows, metrics):
    frontier = []
    for candidate in rows:
        dominated = any(
            dominates(other, candidate, metrics)
            for other in rows
            if other["technique"] != candidate["technique"]
        )
        if not dominated:
            frontier.append(candidate)
    return frontier

if __name__ == "__main__":
    rows = load_results("results/comparison_table.csv")

    # metrics where LOWER is better — pick the ones that matter for the tradeoff story
    metrics = ["p50_latency_sec", "peak_memory_mb", "perplexity"]

    frontier = pareto_frontier(rows, metrics)
    dominated = [r for r in rows if r not in frontier]

    print("Pareto-optimal configs:")
    for r in frontier:
        print(f"  {r['technique']}: latency={r['p50_latency_sec']:.2f}s, "
              f"memory={r['peak_memory_mb']:.0f}MB, perplexity={r['perplexity']:.2f}")

    print("\nDominated configs (strictly worse on every relevant axis vs. some other config):")
    for r in dominated:
        print(f"  {r['technique']}: latency={r['p50_latency_sec']:.2f}s, "
              f"memory={r['peak_memory_mb']:.0f}MB, perplexity={r['perplexity']:.2f}")