"""
Token Merging Benchmark Script (autoresearch-style)
====================================================
Evaluates a single token merging configuration and prints results in a
structured format that can be parsed by the experiment loop or notebook.

Usage:
    python benchmark_tome.py --strategy bipartite --ratio 0.5 --model vit
    python benchmark_tome.py --strategy kmeans --ratio 0.7 --model vlm --device cuda
"""

import argparse
import time
import os
import sys
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "model")

from models import load_vit, load_clip
from token_merging import apply_token_merging


def benchmark_config(model, sample_input, forward_fn, strategy, ratio,
                     device="cpu", warmup=3, runs=10):
    """Run a single token merging config and return metrics."""
    model.eval().to(device)
    sample_input = sample_input.to(device)
    fwd = forward_fn if forward_fn else lambda m, x: m(x)

    # --- Baseline (no merging) ---
    with torch.no_grad():
        for _ in range(warmup):
            fwd(model, sample_input)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            base_out = fwd(model, sample_input)
        if device == "cuda":
            torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) / runs * 1000

    # Extract logits
    if hasattr(base_out, "logits"):
        base_logits = base_out.logits.detach().float()
    elif isinstance(base_out, torch.Tensor):
        base_logits = base_out.detach().float()
    else:
        base_logits = list(base_out.values())[0].detach().float()

    base_pred = base_logits.argmax(dim=-1)

    # --- Merged model ---
    merged_model = apply_token_merging(model, strategy=strategy, ratio=ratio)
    merged_model.eval().to(device)

    # Measure peak memory on CUDA
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            try:
                fwd(merged_model, sample_input)
            except Exception:
                # Some configs may fail on first try due to shape mismatches
                merged_model.remove_hooks()
                return None

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            merged_out = fwd(merged_model, sample_input)
        if device == "cuda":
            torch.cuda.synchronize()
        merged_ms = (time.perf_counter() - t0) / runs * 1000

    peak_mem_mb = 0.0
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Extract merged logits
    if hasattr(merged_out, "logits"):
        merged_logits = merged_out.logits.detach().float()
    elif isinstance(merged_out, torch.Tensor):
        merged_logits = merged_out.detach().float()
    else:
        merged_logits = list(merged_out.values())[0].detach().float()

    merged_pred = merged_logits.argmax(dim=-1)

    # --- Metrics ---
    # Cosine similarity of logit vectors (quality proxy)
    cos_sim = F.cosine_similarity(
        base_logits.flatten().unsqueeze(0),
        merged_logits.flatten().unsqueeze(0),
    ).item()

    # Prediction agreement
    pred_match = (base_pred == merged_pred).float().mean().item()

    # KL divergence (merged vs base)
    base_probs = F.log_softmax(base_logits, dim=-1)
    merged_probs = F.softmax(merged_logits, dim=-1)
    kl_div = F.kl_div(base_probs, merged_probs, reduction="batchmean").item()

    speedup = base_ms / merged_ms if merged_ms > 0 else 0.0

    merged_model.remove_hooks()

    return {
        "strategy": strategy,
        "ratio": ratio,
        "base_ms": base_ms,
        "merged_ms": merged_ms,
        "speedup": speedup,
        "cosine_sim": cos_sim,
        "pred_match": pred_match,
        "kl_div": kl_div,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark a token merging config")
    parser.add_argument("--strategy", type=str, default="bipartite",
                        choices=["bipartite", "kmeans", "average_pool"])
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "vlm"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    if args.model == "vit":
        model, sample_input, forward_fn = load_vit()
    else:
        model, sample_input, forward_fn = load_clip()

    result = benchmark_config(
        model, sample_input, forward_fn,
        strategy=args.strategy, ratio=args.ratio,
        device=args.device, warmup=args.warmup, runs=args.runs,
    )

    if result is None:
        print("---")
        print("status: crash")
        print("description: shape mismatch or runtime error")
        sys.exit(1)

    # Print in autoresearch structured format
    print("---")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
