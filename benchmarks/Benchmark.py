import time
import json
import os
import torch
from statistics import mean
import numpy as np
from src.ModelLoader import ModelLoader


def get_model_size_mb(model_path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 ** 2)

def measure_ttft(model, tokenizer, prompt, device):
    """Time to first token — separate short generate call."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start

def run_single_inference(model, tokenizer, prompt, device, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # deterministic — important for reproducible benchmarking
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_len = outputs.shape[1] - input_len
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else None
    )

    return {
        "latency_sec": elapsed,
        "tokens_generated": output_len,
        "tokens_per_sec": output_len / elapsed if elapsed > 0 else 0,
        "peak_memory_mb": peak_mem_mb,
    }

def run_benchmark(model_path, prompts, num_runs=3, warmup=1, max_new_tokens=100):
    model, tokenizer, device = ModelLoader.load_model(model_path)

    # warmup — discard results
    for _ in range(warmup):
        run_single_inference(model, tokenizer, prompts[0], device, max_new_tokens)

    per_request_results = []
    for prompt in prompts:
        run_latencies = []
        for _ in range(num_runs):
            result = run_single_inference(model, tokenizer, prompt, device, max_new_tokens)
            run_latencies.append(result)

        ttft = measure_ttft(model, tokenizer, prompt, device)

        per_request_results.append({
            "prompt": prompt,
            "runs": run_latencies,
            "ttft_sec": ttft,
            "mean_latency_sec": mean(r["latency_sec"] for r in run_latencies),
            "mean_tokens_per_sec": mean(r["tokens_per_sec"] for r in run_latencies),
        })

    all_latencies = [r["latency_sec"] for pr in per_request_results for r in pr["runs"]]

    summary = {
        "model_path": model_path,
        "device": device,
        "num_prompts": len(prompts),
        "runs_per_prompt": num_runs,
        "p50_latency_sec": float(np.percentile(all_latencies, 50)),
        "p95_latency_sec": float(np.percentile(all_latencies, 95)),
        "p99_latency_sec": float(np.percentile(all_latencies, 99)),
        "mean_ttft_sec": mean(pr["ttft_sec"] for pr in per_request_results),
        "mean_throughput_tok_per_sec": mean(pr["mean_tokens_per_sec"] for pr in per_request_results),
        "peak_memory_mb": max(
            (r["peak_memory_mb"] for pr in per_request_results for r in pr["runs"] if r["peak_memory_mb"]),
            default=None,
        ),
        "model_size_mb": get_model_size_mb(model_path),
    }

    return {"summary": summary, "per_prompt": per_request_results}
