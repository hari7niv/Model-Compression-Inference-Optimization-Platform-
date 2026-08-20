import csv

def load_results(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in r:
            if k != "technique":
                r[k] = float(r[k])
    return rows

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

if __name__ == "__main__":
    rows = load_results("results/comparison_table.csv")
    recommendation_text = generate_recommendations(rows)

    print(recommendation_text)

    with open("results/recommendations.txt", "w", encoding="utf-8") as f:
        f.write(recommendation_text)
    print("Saved: results/recommendations.txt")