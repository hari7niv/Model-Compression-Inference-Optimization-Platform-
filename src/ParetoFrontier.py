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


def generate_recommendations(rows):
    best_latency = min(rows, key=lambda r: r["p50_latency_sec"])
    best_memory = min(rows, key=lambda r: r["peak_memory_mb"])
    best_accuracy = min(rows, key=lambda r: r["perplexity"])

    lines = []
    lines.append("=== Deployment Recommendations ===\n")

    lines.append(
        f"Latency-critical (e.g. real-time / interactive serving):\n"
        f"  → {best_latency['technique']} "
        f"(p50={best_latency['p50_latency_sec']:.2f}s, "
        f"{best_latency['throughput_tok_per_sec']:.1f} tok/s)\n"
    )

    lines.append(
        f"Memory-constrained (e.g. edge devices, limited VRAM):\n"
        f"  → {best_memory['technique']} "
        f"(peak_memory={best_memory['peak_memory_mb']:.0f}MB, "
        f"perplexity={best_memory['perplexity']:.2f})\n"
    )

    lines.append(
        f"Accuracy-critical, memory flexible:\n"
        f"  → {best_accuracy['technique']} "
        f"(perplexity={best_accuracy['perplexity']:.2f}, "
        f"lowest measured degradation)\n"
    )

    # flag any technique that doesn't win any category outright
    winners = {best_latency["technique"], best_memory["technique"], best_accuracy["technique"]}
    non_winners = [r["technique"] for r in rows if r["technique"] not in winners]
    if non_winners:
        lines.append(
            f"Note: {', '.join(non_winners)} did not lead in any single category in this "
            f"benchmark run, but remains on the Pareto frontier — meaning it still represents "
            f"a valid tradeoff point on at least one axis relative to every other config tested. "
            f"Worth re-evaluating if benchmarking conditions or hardware change.\n"
        )

    return "\n".join(lines)


def build_comparison_table(result_files, output_csv="results/comparison_table.csv"):
    """
    result_files: list of (technique, json_path) tuples, e.g.
        [("fp16", "results/baseline_fp16.json"),
         ("int8", "results/int8_llm.json"),
         ("int4", "results/int4_llm.json")]
    Reads each file's "summary" block and writes a flat comparison CSV.
    """
    rows = []
    for technique, path in result_files:
        with open(path) as f:
            data = json.load(f)["summary"]
        rows.append({
            "technique": technique,
            "p50_latency_sec": data["p50_latency_sec"],
            "p95_latency_sec": data["p95_latency_sec"],
            "p99_latency_sec": data["p99_latency_sec"],
            "mean_ttft_sec": data["mean_ttft_sec"],
            "throughput_tok_per_sec": data["mean_throughput_tok_per_sec"],
            "peak_memory_mb": data["peak_memory_mb"],
            "perplexity": data["perplexity"],
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}")
    return rows


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

    recommendation_text = generate_recommendations(rows)
    
    print(recommendation_text)

    with open("results/recommendations.txt", "w", encoding="utf-8") as f:
        f.write(recommendation_text)
    print("Saved: results/recommendations.txt") 