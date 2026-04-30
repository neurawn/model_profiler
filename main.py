"""
Main Pipeline
=============
Orchestrates the full profiling and recommendation pipeline across all models:
1. Load models (GPT-2, ResNet-18, ViT, Blackbox)
2. Run static profiling on each
3. Run dynamic profiling on each
4. Generate compression recommendations
5. Compare results and validate hypotheses

Usage:
    python main.py                  # Profile all models
    python main.py --model gpt2     # Profile only GPT-2
    python main.py --static-only    # Skip dynamic profiling
    python main.py --device cuda    # Use GPU for dynamic profiling
"""

import sys
import os
import json
import time
import argparse
import torch

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set HuggingFace cache to local model folder
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_DIR

from static_profiler import StaticProfiler
from dynamic_profiler import DynamicProfiler
from compression_recommender import CompressionRecommender, recommend_for_model
from models import (
    load_gpt2, load_resnet18, load_vit, load_blackbox_model, load_all_models,
    gpt2_input_fn, resnet_input_fn, vit_input_fn,
)


def profile_single_model(model_name, model, sample_input, forward_fn,
                          static_profiler, dynamic_profiler,
                          run_dynamic=True, target_device="gpu"):
    """Run full profiling pipeline on a single model."""
    print(f"\n{'#'*70}")
    print(f"#  PROFILING: {model_name}")
    print(f"{'#'*70}")

    # ── Static Profiling ──
    print(f"\n[Static Profiling] Analyzing {model_name} architecture...")
    t0 = time.time()
    static_profile = static_profiler.profile(model, model_name=model_name)
    static_time = time.time() - t0
    print(f"  Completed in {static_time:.2f}s")
    print(static_profile.summary())

    # ── Dynamic Profiling ──
    dynamic_profile = None
    if run_dynamic:
        print(f"[Dynamic Profiling] Benchmarking {model_name} at runtime...")
        t0 = time.time()
        dynamic_profile = dynamic_profiler.profile(
            model, sample_input,
            model_name=model_name,
            forward_fn=forward_fn,
        )
        dynamic_time = time.time() - t0
        print(f"  Completed in {dynamic_time:.2f}s")
        print(dynamic_profile.summary())

    # # ── Compression Recommendation ──
    # print(f"[Recommendations] Generating compression advice for {model_name}...")
    # report = recommend_for_model(
    #     static_profile, dynamic_profile,
    #     target_device=target_device,
    #     verbose=True,
    # )

    return {
        "static": static_profile,
        "dynamic": dynamic_profile,
        # "recommendation": report,
    }


def compare_results(all_results):
    """
    Compare profiling results across all models to validate the proposal's
    key hypotheses about parameter distribution and compression methods.
    """
    print(f"\n{'#'*70}")
    print(f"#  COMPARATIVE ANALYSIS")
    print(f"{'#'*70}")

    # ── Hypothesis 1: Transformer has more uniform parameter distribution ──
    print("\n[Hypothesis 1] Parameter Distribution Uniformity")
    print("  Proposal predicts: Transformer > ViT > CNN")
    print("  ─────────────────────────────────────────────")
    for name, result in all_results.items():
        sp = result["static"]
        print(f"  {name:20s}  uniformity = {sp.param_uniformity_score:.4f}")

    # ── Hypothesis 2: Quantization recommended for Transformers ──
    print("\n[Hypothesis 2] Top Recommended Method per Model")
    print("  Proposal predicts: Transformer→Quantization, CNN→Pruning")
    print("  ─────────────────────────────────────────────")
    for name, result in all_results.items():
        rec = result["recommendation"]
        top = rec.top_recommendation
        if top:
            print(f"  {name:20s}  → {top.method.value:30s} (score={top.score:.3f})")
            print(f"  {'':20s}    arch_class = {rec.inferred_architecture_class}")

    # ── Hypothesis 3: ViT should get a mixed recommendation ──
    print("\n[Hypothesis 3] ViT Recommendation Analysis")
    if "ViT-Base" in all_results:
        rec = all_results["ViT-Base"]["recommendation"]
        print(f"  Architecture classified as: {rec.inferred_architecture_class}")
        print(f"  Top 3 recommendations:")
        for i, r in enumerate(rec.recommendations[:3]):
            print(f"    {i+1}. {r.method.value:30s} score={r.score:.3f}")

    # ── Architecture classification for blackbox ──
    print("\n[Blackbox Analysis] Can we identify the unknown model?")
    if "Blackbox" in all_results:
        rec = all_results["Blackbox"]["recommendation"]
        sp = all_results["Blackbox"]["static"]
        print(f"  Inferred class:     {rec.inferred_architecture_class}")
        print(f"  Has attention:      {sp.has_attention}")
        print(f"  Has convolutions:   {sp.has_convolutions}")
        print(f"  Uniformity:         {sp.param_uniformity_score:.4f}")
        print(f"  Top recommendation: {rec.top_recommendation.method.value}")

    print(f"\n{'#'*70}\n")


def save_results(all_results, output_dir):
    """Save all profiling results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    for name, result in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        data = {
            "static_profile": result["static"].to_dict(),
            # "recommendation": result["recommendation"].to_dict(),
        }
        if result["dynamic"]:
            data["dynamic_profile"] = result["dynamic"].to_dict()

        path = os.path.join(output_dir, f"{safe_name}_profile.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Model Profiler & Compression Recommender"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gpt2", "gpt2-medium", "gpt2-large",
                                 "resnet", "vit", "blackbox"],
                        help="Which model to profile")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to a local PyTorch model file (.pt/.pth). "
                             "Overrides --model. Provide --input-shape for non-default input.")
    parser.add_argument("--input-shape", type=str, default=None,
                        help="Input shape as comma-separated ints (e.g. '1,3,224,224'). "
                             "Used with --local. Defaults to 1,3,224,224.")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip dynamic profiling")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for dynamic profiling (cpu/cuda)")
    parser.add_argument("--target-device", type=str, default="gpu",
                        choices=["gpu", "cpu", "mobile", "edge"],
                        help="Target deployment device for recommendations")
    parser.add_argument("--output-dir", type=str, default="./profiling_results",
                        help="Directory to save JSON results")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs for dynamic profiling")
    parser.add_argument("--benchmark-runs", type=int, default=20,
                        help="Number of benchmark runs for latency measurement")
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL PROFILER & COMPRESSION RECOMMENDER")
    print("  ML to Improve ML — NYU Final Project")
    print("=" * 70)
    print(f"  Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  Target: {args.target_device}")
    print(f"  Dynamic profiling: {'OFF' if args.static_only else 'ON'}")
    print("=" * 70)

    # Initialize profilers
    static_profiler = StaticProfiler()
    dynamic_profiler = DynamicProfiler(
        warmup_runs=args.warmup,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
    )

    # ── Local model path ──
    if args.local:
        local_path = os.path.abspath(args.local)
        if not os.path.isfile(local_path):
            print(f"Error: local model file not found: {local_path}")
            sys.exit(1)

        # Parse input shape
        if args.input_shape:
            input_shape = tuple(int(x) for x in args.input_shape.split(","))
        else:
            input_shape = (1, 3, 224, 224)

        print(f"\n[Local Model] Loading from {local_path}...")
        loaded = torch.load(local_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, nn.Module):
            local_model = loaded
        elif isinstance(loaded, dict):
            print("Error: file contains a state_dict, not a full model. "
                  "Wrap it in an nn.Module and save with torch.save(model, path).")
            sys.exit(1)
        else:
            print(f"Error: unexpected type in file: {type(loaded)}")
            sys.exit(1)

        local_name = os.path.splitext(os.path.basename(local_path))[0]
        sample_input = torch.randn(*input_shape)
        print(f"  Loaded {local_name} ({sum(p.numel() for p in local_model.parameters()):,} params)")

        all_results = {}
        result = profile_single_model(
            local_name, local_model, sample_input, None,
            static_profiler, dynamic_profiler,
            run_dynamic=not args.static_only,
            target_device=args.target_device,
        )
        all_results[local_name] = result

        print("\n[Saving Results]")
        save_results(all_results, args.output_dir)
        print("\n Pipeline complete!\n")
        return

    # Load and profile models
    all_results = {}
    model_loaders = {
        "gpt2": ("GPT-2", load_gpt2),
        "gpt2-medium": ("GPT-2-Medium", lambda: load_gpt2("gpt2-medium")),
        "gpt2-large": ("GPT-2-Large", lambda: load_gpt2("gpt2-large")),
        "resnet": ("ResNet-18", load_resnet18),
        "vit": ("ViT-Base", load_vit),
        "blackbox": ("Blackbox", load_blackbox_model),
    }

    if args.model == "all":
        targets = list(model_loaders.keys())
    else:
        targets = [args.model]

    for key in targets:
        name, loader = model_loaders[key]
        try:
            model, sample_input, forward_fn = loader()
            result = profile_single_model(
                name, model, sample_input, forward_fn,
                static_profiler, dynamic_profiler,
                run_dynamic=not args.static_only,
                target_device=args.target_device,
            )
            all_results[name] = result

            # Free memory
            del model, sample_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  ✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(all_results) > 1:
        compare_results(all_results)

    # Save results
    print("\n[Saving Results]")
    save_results(all_results, args.output_dir)

    print("\n✓ Pipeline complete!\n")


if __name__ == "__main__":
    main()
