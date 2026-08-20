import json
import csv

files = {
    "fp16": "results/baseline_fp16.json",
    "int8": "results/int8_llm.json",
    "int4": "results/int4_llm.json",
}

rows = []
for technique, path in files.items():
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

with open("results/comparison_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved: results/comparison_table.csv")
for r in rows:
    print(r)