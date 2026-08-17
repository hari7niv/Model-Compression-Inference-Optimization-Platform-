import argparse
import json
import math
import os
import time
from statistics import mean

import numpy as np
import torch
from transformers import BitsAndBytesConfig

from src.ConfigLoader import ConfigLoader
from src.ModelLoader import ModelLoader


def compute_perplexity(model, tokenizer, texts, device):
    """Perplexity on a fixed set of held-out texts — standard LLM quality proxy."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # loss is mean NLL per token; multiply back to get sum for correct aggregation
        num_tokens = inputs["input_ids"].shape[1]
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    return math.exp(total_loss / total_tokens)

def get_quantization_config(quant_type):
    if quant_type in (None, "none"):
        return None
    elif quant_type == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    elif quant_type == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


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

def run_benchmark(config):
    model_path = config["model"]["path"]

    prompts = config["prompts"]

    num_runs = config["benchmark"]["num_runs"]
    warmup = config["benchmark"]["warmup_runs"]
    max_new_tokens = config["benchmark"]["max_new_tokens"]
    quantization = config["model"].get("quantization", None)
    quantization_config = get_quantization_config(quantization)
    precision = config["model"].get("precision", "fp16")
    model, tokenizer, device = ModelLoader.load_model(model_path, bnb_config=quantization_config, precision=precision)

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
        "device": str(device),
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, help="Path to the configuration file")

    args = parser.parse_args()

    config = ConfigLoader.load(args.config)

    results = run_benchmark(config)

    # Output configuration
    output_dir = config["output"]["directory"]
    output_filename = config["output"]["filename"]

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        output_filename
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\nBenchmark complete.")
    print(f"Results saved to: {output_path}")

    print("\nSummary:")
    print(
        json.dumps(
            results["summary"],
            indent=2
        )
    )