import csv
import matplotlib.pyplot as plt

def load_results(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in r:
            if k != "technique":
                r[k] = float(r[k])
    return rows

def plot_pareto(rows, frontier_techniques, save_path="results/pareto_frontier.png"):
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"fp16": "#2ecc71", "int8": "#e74c3c", "int4": "#3498db"}

    for r in rows:
        technique = r["technique"]
        x = r["peak_memory_mb"]
        y = r["perplexity"]
        # bubble size encodes latency — scale for visibility
        size = r["p50_latency_sec"] * 60

        ax.scatter(
            x, y, s=size,
            color=colors.get(technique, "gray"),
            alpha=0.6, edgecolors="black", linewidth=1.5,
            zorder=3
        )
        ax.annotate(
            f"{technique}\n{r['p50_latency_sec']:.1f}s latency",
            (x, y),
            textcoords="offset points", xytext=(15, 10),
            fontsize=10, fontweight="bold"
        )

    ax.set_xlabel("Peak Memory (MB) — lower is better", fontsize=11)
    ax.set_ylabel("Perplexity — lower is better", fontsize=11)
    ax.set_title(
        "Quantization Tradeoff: Memory vs. Accuracy\n(bubble size = p50 latency)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # invert both axes so "better" is toward top-right — origin = ideal (low mem, low perplexity)
    ax.invert_xaxis()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    rows = load_results("results/comparison_table.csv")
    frontier_techniques = [r["technique"] for r in rows]  # all 3 are on frontier
    plot_pareto(rows, frontier_techniques)