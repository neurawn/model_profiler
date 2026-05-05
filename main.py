"""
Main Pipeline
=============
Orchestrates the full profiling and recommendation pipeline across all models:
1. Load models (GPT-2, ResNet-18, ViT, Blackbox)
2. Run static profiling on each
3. Generate compression recommendations
4. Compare results and validate hypotheses

Usage:
    python main.py                  # Profile all models
    python main.py --model gpt2     # Profile only GPT-2
    python main.py --graph-export   # Run graph export analysis
"""

import sys
import os
import json
import time
import argparse
import torch
import torch.nn as nn

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set HuggingFace cache to local model folder
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_DIR

from static_profiler import StaticProfiler
from compression_recommender import CompressionRecommender, recommend_for_model
from models import (
    load_gpt2, load_resnet18, load_vit, load_vlm,
    gpt2_input_fn, resnet_input_fn, vit_input_fn,
)
from token_merging import (
    compare_strategies, apply_token_merging, TokenMergingProfiler,
)
from quantize import quantize_model
from graph_export import run_graph_export, _export_model as export_single_model


def apply_quant_plan(model, model_name, compression_plan, sample_input=None,
                     forward_fn=None, output_dir="./profiling_results"):
    """
    Apply the compression plan's per-layer quantization assignment.

    Each layer gets quantized to its assigned bit-width (4, 8, or 16)
    using symmetric weight-only quantization. This produces a mixed-precision
    model where different layers use different bit-widths.
    """
    import copy

    print(f"\n{'#'*70}")
    print(f"#  APPLYING QUANTIZATION PLAN: {model_name}")
    print(f"#  Avg target bits: {compression_plan.avg_bits:.1f}")
    print(f"{'#'*70}")

    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D
    except ImportError:
        HFConv1D = None

    linear_types = (nn.Linear,)
    if HFConv1D is not None:
        linear_types = (nn.Linear, HFConv1D)

    quantized = copy.deepcopy(model)
    quantized.eval()

    # Build lookup: module_name -> assigned bits
    assignment_map = {}
    for qa in compression_plan.quant_assignments:
        assignment_map[qa.name] = qa.assigned_bits

    # Build reason lookup
    reason_map = {}
    for qa in compression_plan.quant_assignments:
        reason_map[qa.name] = qa.reason

    # Track stats
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    quantized_count = {4: 0, 8: 0, 16: 0}
    quantized_params = {4: 0, 8: 0, 16: 0}
    conversion_log = []  # (name, original_dtype, target_bits, params, reason)

    for mod_name, mod in quantized.named_modules():
        if not isinstance(mod, (*linear_types, nn.Conv2d, nn.Conv1d)):
            continue

        bits = assignment_map.get(mod_name, 8)  # default to INT8
        reason = reason_map.get(mod_name, "default INT8")

        weight = mod.weight.data.float()
        orig_dtype = str(mod.weight.dtype)

        if bits == 16:
            quantized_count[16] += 1
            quantized_params[16] += weight.numel()
            conversion_log.append((mod_name, orig_dtype, "FP16", weight.numel(), reason))
            continue
        elif bits == 8:
            w_max = weight.abs().max().clamp(min=1e-8)
            scale = w_max / 127.0
            w_int = torch.clamp(torch.round(weight / scale), -128, 127)
            w_deq = w_int * scale
            target_str = "INT8"
        elif bits == 4:
            if weight.ndim >= 2:
                reduce_dims = list(range(1, weight.ndim))
                w_max = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
                scale = w_max / 7.0
                w_int = torch.clamp(torch.round(weight / scale), -8, 7)
                w_deq = w_int * scale
            else:
                w_max = weight.abs().max().clamp(min=1e-8)
                scale = w_max / 7.0
                w_int = torch.clamp(torch.round(weight / scale), -8, 7)
                w_deq = w_int * scale
            target_str = "INT4"
        else:
            continue

        mod.weight.data = w_deq
        mod.register_buffer("_wq_scale", scale.squeeze())
        mod.register_buffer("_wq_bits", torch.tensor(bits))
        quantized_count[bits] += 1
        quantized_params[bits] += weight.numel()
        conversion_log.append((mod_name, orig_dtype, target_str, weight.numel(), reason))

    # Verify forward pass
    if sample_input is not None:
        print(f"  Verifying quantized model...")
        fwd = forward_fn if forward_fn else lambda m, x: m(x)
        with torch.no_grad():
            fwd(quantized, sample_input.cpu())
        print(f"  Forward pass OK")

    # Estimate compressed size
    compressed_bytes = sum(quantized_params[b] * (b / 8) for b in [4, 8, 16])
    other_bytes = sum(
        p.numel() * p.element_size() for n, p in quantized.named_parameters()
        if "weight" not in n
    )
    estimated_size_mb = (compressed_bytes + other_bytes) / (1024 * 1024)

    # Print per-layer conversion summary
    print(f"\n  Per-Layer Quantization:")
    print(f"  {'Layer':<45s} {'From':>8s} {'To':>6s} {'Params':>12s} Reason")
    print(f"  {'─'*45} {'─'*8} {'─'*6} {'─'*12} {'─'*35}")
    for name, orig, target, params, reason in conversion_log:
        print(f"  {name[:45]:<45s} {orig:>8s} → {target:<5s} {params:>12,} {reason}")

    # Print totals
    print(f"\n  Summary:")
    print(f"    INT4 layers:  {quantized_count[4]:>4d} ({quantized_params[4]:>12,} params)")
    print(f"    INT8 layers:  {quantized_count[8]:>4d} ({quantized_params[8]:>12,} params)")
    print(f"    FP16 layers:  {quantized_count[16]:>4d} ({quantized_params[16]:>12,} params)")
    print(f"    Original:     {original_size:.1f} MB")
    print(f"    Estimated:    {estimated_size_mb:.1f} MB ({1 - estimated_size_mb/original_size:.1%} reduction)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    save_path = os.path.join(output_dir, f"{safe_name}_plan_quantized.pt")
    torch.save({
        "model_state_dict": quantized.state_dict(),
        "quant_assignments": [
            {"name": qa.name, "bits": qa.assigned_bits, "reason": qa.reason}
            for qa in compression_plan.quant_assignments
        ],
        "avg_bits": compression_plan.avg_bits,
        "layer_counts": quantized_count,
        "layer_params": quantized_params,
        "estimated_size_mb": estimated_size_mb,
    }, save_path)
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  Saved: {save_path} ({file_size:.1f} MB)")
    print(f"{'#'*70}\n")
    return save_path


def apply_and_save_tome(model, model_name, forward_fn, sample_input,
                        strategy="bipartite", ratio=0.5, output_dir="./profiling_results"):
    """Apply token merging to a ViT/VLM model and save it locally."""
    print(f"\n{'#'*70}")
    print(f"#  APPLYING TOKEN MERGING: {model_name}")
    print(f"#  Strategy: {strategy}, Keep ratio: {ratio:.0%}")
    print(f"{'#'*70}")

    try:
        merged_model = apply_token_merging(
            model, strategy=strategy, ratio=ratio,
        )

        # Verify it runs
        print(f"  Verifying merged model...")
        merged_model.eval()
        fwd = forward_fn if forward_fn else lambda m, x: m(x)
        with torch.no_grad():
            fwd(merged_model, sample_input)
        print(f"  Forward pass OK")

        # Save
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        save_path = os.path.join(
            output_dir,
            f"{safe_name}_tome_{strategy}_r{ratio}.pt",
        )

        # Remove hooks before saving (closures can't be pickled)
        merged_model.remove_hooks()

        save_data = {
            "model_state_dict": merged_model.model.state_dict(),
            "model_class": type(merged_model.model).__name__,
            "merge_config": {
                "strategy": strategy,
                "ratio": ratio,
                "protect_cls": merged_model.protect_cls,
                "merge_layers": merged_model.merge_layers,
            },
        }
        torch.save(save_data, save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)

        print(f"\n  Saved: {save_path} ({size_mb:.1f} MB)")
        print(f"  Config: strategy={strategy}, ratio={ratio}")
        print(f"\n  To reload:")
        print(f"    data = torch.load('{save_path}')")
        print(f"    model.load_state_dict(data['model_state_dict'])")
        print(f"    wrapped = apply_token_merging(model, **data['merge_config'])")
        print(f"{'#'*70}\n")
        return save_path

    except Exception as e:
        print(f"  Error applying token merging: {e}")
        import traceback
        traceback.print_exc()
        return None


def profile_single_model(model_name, model, sample_input, forward_fn,
                          static_profiler, target_device="gpu"):
    """Run static profiling pipeline on a single model."""
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

    return {
        "static": static_profile,
    }


def analyze_token_merging(model, sample_input, forward_fn, model_name,
                          ratios=None,
                          save_merged=False, save_strategy="bipartite",
                          save_ratio=0.5, output_dir="./profiling_results"):
    """
    Run token merging analysis on a ViT model and estimate theoretical speedup.

    Compares all three strategies (bipartite, kmeans, average_pool) and
    computes theoretical FLOPs/latency reduction based on token count reduction.
    """
    if ratios is None:
        ratios = [0.3, 0.5, 0.7, 0.9]

    print(f"\n{'#'*70}")
    print(f"#  TOKEN MERGING ANALYSIS: {model_name}")
    print(f"{'#'*70}")

    # ── 1. Extract intermediate tokens from the model ──
    token_tensor = None

    def capture_hook(module, input, output):
        nonlocal token_tensor
        if isinstance(output, tuple):
            token_tensor = output[0].detach()
        elif isinstance(output, torch.Tensor) and output.ndim == 3:
            token_tensor = output.detach()

    # Find the first transformer layer to hook
    hook_handle = None
    for name, module in model.named_modules():
        module_name = type(module).__name__.lower()
        if any(kw in module_name for kw in ["vitlayer", "gpt2block", "encoderlayer", "bertlayer"]):
            hook_handle = module.register_forward_hook(capture_hook)
            break

    if hook_handle is None:
        print("  Could not find transformer layers for token merging analysis.")
        return None

    # Run forward pass to capture tokens
    model.eval()
    with torch.no_grad():
        if forward_fn:
            forward_fn(model, sample_input)
        else:
            model(sample_input)
    hook_handle.remove()

    if token_tensor is None:
        print("  Could not capture token embeddings.")
        return None

    B, T, D = token_tensor.shape
    print(f"\n  Token sequence: B={B}, T={T}, D={D}")

    # ── 2. Compare strategies at each ratio ──
    results = {}
    for ratio in ratios:
        print(f"\n{'─'*60}")
        print(f"  Keep ratio = {ratio:.0%} (merge {1 - ratio:.0%} of tokens)")
        print(f"{'─'*60}")

        profile = compare_strategies(
            token_tensor, ratio=ratio,
            model_name=f"{model_name} (ratio={ratio})",
            verbose=True,
        )

        # ── 3. Theoretical speedup estimation ──
        for strategy_name, merge_result in profile.results.items():
            T_new = merge_result.merged_count
            r = T_new / T  # reduction ratio

            attn_speedup = 1.0 / (r ** 2) if r > 0 else 1.0
            ffn_speedup = 1.0 / r if r > 0 else 1.0
            overall_speedup = 1.0 / (0.4 * r**2 + 0.6 * r) if r > 0 else 1.0

            print(f"\n  [{strategy_name.upper()}] Theoretical Speedup:")
            print(f"    Tokens:           {T} -> {T_new}")
            print(f"    Attention speedup: {attn_speedup:.2f}x  (O(T^2) reduction)")
            print(f"    FFN speedup:       {ffn_speedup:.2f}x  (O(T) reduction)")
            print(f"    Overall speedup:   {overall_speedup:.2f}x")

        results[ratio] = profile

    # ── 4. Save merged model if requested ──
    if save_merged:
        print(f"\n{'─'*60}")
        print(f"  SAVING MERGED MODEL (ratio={save_ratio}, {save_strategy})")
        print(f"{'─'*60}")
        try:
            merged_model = apply_token_merging(
                model, strategy=save_strategy, ratio=save_ratio,
            )
            os.makedirs(output_dir, exist_ok=True)
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            save_path = os.path.join(
                output_dir,
                f"{safe_name}_tome_{save_strategy}_r{save_ratio}.pt",
            )
            # Remove hooks before saving (closures can't be pickled)
            merged_model.remove_hooks()
            # Save the base model state dict + merging config
            save_data = {
                "model_state_dict": merged_model.model.state_dict(),
                "model_class": type(merged_model.model).__name__,
                "merge_config": {
                    "strategy": save_strategy,
                    "ratio": save_ratio,
                    "protect_cls": merged_model.protect_cls,
                    "merge_layers": merged_model.merge_layers,
                },
            }
            torch.save(save_data, save_path)
            print(f"    Saved merged model: {save_path}")
            print(f"    Load with: data = torch.load('{save_path}')")
            print(f"    Then:      model.load_state_dict(data['model_state_dict'])")
            print(f"               wrapped = apply_token_merging(model, **data['merge_config'])")
        except Exception as e:
            print(f"    Could not save merged model: {e}")

    print(f"\n{'#'*70}\n")
    return results


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

    print(f"\n{'#'*70}\n")


def save_results(all_results, output_dir):
    """Save all profiling results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    for name, result in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        data = {
            "static_profile": result["static"].to_dict(),
        }
        if "graph_profile" in result:
            data["graph_profile"] = result["graph_profile"].to_dict()

        path = os.path.join(output_dir, f"{safe_name}_profile.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")


def _is_safetensors(path: str) -> bool:
    """Check if a file is safetensors format by reading the header."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        return b'{"__meta' in header or b'"__metadata__"' in header
    except Exception:
        return False


def _load_arch_with_weights(arch: str, weight_path: str,
                            state_dict: dict = None,
                            is_safetensors: bool = False):
    """
    Instantiate a model architecture and load weights into it.

    Returns (model, forward_fn).
    """
    forward_fn = None

    # GPT-2 family
    if arch in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        from transformers import GPT2LMHeadModel
        print(f"  Architecture: GPT2LMHeadModel ({arch})")
        model = GPT2LMHeadModel.from_pretrained(arch, cache_dir=MODEL_DIR)
        forward_fn = lambda m, x: m(x, labels=x)

    # ViT family
    elif "vit" in arch.lower():
        from transformers import ViTForImageClassification
        print(f"  Architecture: ViTForImageClassification ({arch})")
        model = ViTForImageClassification.from_pretrained(arch, cache_dir=MODEL_DIR)
        forward_fn = lambda m, x: m(pixel_values=x)

    # ResNet
    elif arch.lower() in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        import torchvision.models as tv_models
        print(f"  Architecture: {arch}")
        model = getattr(tv_models, arch.lower())(weights=None)

    else:
        # Try as a HuggingFace model ID
        try:
            from transformers import AutoModel
            print(f"  Architecture: AutoModel ({arch})")
            model = AutoModel.from_pretrained(arch, cache_dir=MODEL_DIR)
        except Exception as e:
            print(f"Error: unknown architecture '{arch}': {e}")
            sys.exit(1)

    # Load weights
    if is_safetensors:
        try:
            from safetensors.torch import load_file
            sd = load_file(weight_path)
        except ImportError:
            print("Error: safetensors package required. Install with: pip install safetensors")
            sys.exit(1)
    elif state_dict is not None:
        sd = state_dict
    else:
        sd = None

    if sd is not None:
        # Try strict load first, fall back to non-strict
        try:
            model.load_state_dict(sd, strict=True)
            print(f"  Loaded weights (strict)")
        except RuntimeError:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Loaded weights (non-strict): "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

    model.eval()
    return model, forward_fn


def main():
    parser = argparse.ArgumentParser(
        description="Model Profiler & Compression Recommender"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gpt2", "gpt2-medium", "gpt2-large",
                                 "resnet", "vit", "vlm"],
                        help="Which model to profile")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to a local PyTorch model file (.pt/.pth). "
                             "Overrides --model. Provide --input-shape for non-default input.")
    parser.add_argument("--input-shape", type=str, default=None,
                        help="Input shape as comma-separated ints (e.g. '1,3,224,224'). "
                             "Used with --local. Defaults to 1,3,224,224.")
    parser.add_argument("--arch", type=str, default=None,
                        help="Model architecture for loading state dicts or safetensors. "
                             "e.g. 'gpt2', 'gpt2-medium', 'gpt2-large', "
                             "'google/vit-base-patch16-224', 'resnet18'")
    parser.add_argument("--target-device", type=str, default="gpu",
                        choices=["gpu", "cpu", "mobile", "edge"],
                        help="Target deployment device for recommendations")
    parser.add_argument("--output-dir", type=str, default="./profiling_results",
                        help="Directory to save JSON results")
    parser.add_argument("--token-merging", action="store_true",
                        help="Run token merging analysis on ViT/VLM models with "
                             "theoretical speedup estimation")
    parser.add_argument("--merge-ratios", type=str, default="0.3,0.5,0.7,0.9",
                        help="Comma-separated keep ratios for token merging "
                             "(e.g. '0.3,0.5,0.7')")
    parser.add_argument("--merge-strategy", type=str, default="bipartite",
                        choices=["bipartite", "kmeans", "average_pool"],
                        help="Strategy to use for token merging")
    parser.add_argument("--merge-ratio", type=float, default=0.5,
                        help="Keep ratio for token merging (default: 0.5)")
    parser.add_argument("--apply-tome", action="store_true",
                        help="Apply token merging to ViT/VLM model after profiling "
                             "and save the merged model locally")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["dynamic", "static", "float16",
                                 "weight_int8", "weight_int4", "smoothquant"],
                        help="Quantize the model and save the result. "
                             "dynamic=INT8 dynamic, static=INT8 calibrated, float16=half, "
                             "weight_int8=weight-only INT8, weight_int4=weight-only INT4, "
                             "smoothquant=SmoothQuant W8A8")
    parser.add_argument("--save-quantized", type=str, default=None,
                        help="Path to save the quantized model (default: auto-generated in output-dir)")
    parser.add_argument("--graph", action="store_true",
                        help="Run full graph analysis: export graph, compressible op detection, "
                             "per-node FLOPs/params/activation memory, KV cache, roofline, "
                             "and compression plan")
    parser.add_argument("--save-graph", action="store_true",
                        help="Save exported graphs and CSVs to output directory")
    parser.add_argument("--full-export", action="store_true",
                        help="Use torch.compile for graph export (CPU-intensive). "
                             "By default, uses fast module-walk which gives same "
                             "FLOPs/compression plan quality")
    parser.add_argument("--context-lengths", type=str, default="512,1024,2048,4096,8192",
                        help="Comma-separated context lengths for KV cache estimation")
    parser.add_argument("--apply-quantization", action="store_true",
                        help="Apply the compression plan's per-layer quantization assignment "
                             "from --graph and save the mixed-precision quantized model")
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL PROFILER & COMPRESSION RECOMMENDER")
    print("  ML to Improve ML — NYU Final Project")
    print("=" * 70)
    print(f"  Target: {args.target_device}")
    print("=" * 70)

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]

    # Initialize profiler
    static_profiler = StaticProfiler()

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

        local_model = None
        forward_fn = None
        is_safetensors = local_path.endswith(".safetensors") or _is_safetensors(local_path)

        if is_safetensors:
            # SafeTensors file — requires --arch to know the model structure
            if not args.arch:
                print("Error: safetensors file detected. Use --arch to specify the model "
                      "architecture (e.g. --arch gpt2, --arch google/vit-base-patch16-224)")
                sys.exit(1)
            local_model, forward_fn = _load_arch_with_weights(
                args.arch, local_path, is_safetensors=True)
        else:
            try:
                loaded = torch.load(local_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Error loading file: {e}")
                sys.exit(1)

            if isinstance(loaded, nn.Module):
                local_model = loaded
            elif isinstance(loaded, dict):
                # state_dict — requires --arch
                if not args.arch:
                    print("Error: file contains a state_dict. Use --arch to specify the model "
                          "architecture (e.g. --arch gpt2, --arch resnet18)")
                    sys.exit(1)
                local_model, forward_fn = _load_arch_with_weights(
                    args.arch, local_path, state_dict=loaded)
            else:
                print(f"Error: unexpected type in file: {type(loaded)}")
                sys.exit(1)

        local_name = os.path.splitext(os.path.basename(local_path))[0]
        sample_input = torch.randn(*input_shape)
        print(f"  Loaded {local_name} ({sum(p.numel() for p in local_model.parameters()):,} params)")

        all_results = {}
        result = profile_single_model(
            local_name, local_model, sample_input, forward_fn,
            static_profiler, target_device=args.target_device,
        )
        all_results[local_name] = result

        # Graph analysis (export + profiling + compression plan)
        if args.graph:
            graph_result = export_single_model(
                local_model, sample_input, forward_fn,
                model_name=local_name, model_family="local",
                save_graph=args.save_graph, output_dir=args.output_dir,
                full_export=args.full_export,
            )
            print(graph_result.summary())
            if graph_result.success:
                gsp = static_profiler.profile_from_graph(
                    local_model, graph_result,
                    model_name=local_name, model_family="local",
                    target_context_lengths=ctx_lens,
                )
                print(gsp.summary())
                all_results[local_name]["graph_profile"] = gsp

                # Apply plan-based quantization
                if args.apply_quantization and gsp.compression_plan:
                    apply_quant_plan(
                        local_model, local_name, gsp.compression_plan,
                        sample_input=sample_input, forward_fn=forward_fn,
                        output_dir=args.output_dir,
                    )

        if args.apply_quantization and not args.graph:
            print("Error: --apply-quantization requires --graph to generate the compression plan first")

        # Quantize if requested
        if args.quantize:
            safe_name = local_name.lower().replace(" ", "_").replace("-", "_")
            save_path = args.save_quantized or os.path.join(
                args.output_dir, f"{safe_name}_quantized_{args.quantize}.pt"
            )
            quantized, qresult = quantize_model(
                local_model, method=args.quantize,
                model_name=local_name, sample_input=sample_input,
                forward_fn=forward_fn, save_path=save_path,
            )

        # Apply token merging and save
        if args.apply_tome:
            apply_and_save_tome(
                local_model, local_name, forward_fn, sample_input,
                strategy=args.merge_strategy, ratio=args.merge_ratio,
                output_dir=args.output_dir,
            )

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
        "vlm": ("CLIP-ViT-Base", load_vlm),
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
                static_profiler, target_device=args.target_device,
            )
            all_results[name] = result

            # Graph analysis (export + profiling + compression plan)
            if args.graph:
                family = "llm" if "gpt" in key else (
                    "vit" if "vit" in key else (
                    "vlm" if "vlm" in key or "clip" in key else "unknown"))
                graph_result = export_single_model(
                    model, sample_input, forward_fn,
                    model_name=name, model_family=family,
                    save_graph=args.save_graph, output_dir=args.output_dir,
                    full_export=args.full_export,
                )
                print(graph_result.summary())
                if graph_result.success:
                    gsp = static_profiler.profile_from_graph(
                        model, graph_result,
                        model_name=name, model_family=family,
                        target_context_lengths=ctx_lens,
                    )
                    print(gsp.summary())
                    all_results[name]["graph_profile"] = gsp

                    # Apply plan-based quantization
                    if args.apply_quantization and gsp.compression_plan:
                        apply_quant_plan(
                            model, name, gsp.compression_plan,
                            sample_input=sample_input, forward_fn=forward_fn,
                            output_dir=args.output_dir,
                        )

            # Quantize if requested
            if args.quantize:
                safe_name = name.lower().replace(" ", "_").replace("-", "_")
                save_path = args.save_quantized or os.path.join(
                    args.output_dir, f"{safe_name}_quantized_{args.quantize}.pt"
                )
                quantized, qresult = quantize_model(
                    model, method=args.quantize,
                    model_name=name, sample_input=sample_input,
                    forward_fn=forward_fn, save_path=save_path,
                )

            # Token merging analysis for ViT/VLM models
            is_vision = any(k in key.lower() for k in ["vit", "vlm", "clip"])
            if args.token_merging and is_vision:
                merge_ratios = [float(r) for r in args.merge_ratios.split(",")]
                tm_results = analyze_token_merging(
                    model, sample_input, forward_fn, name,
                    ratios=merge_ratios,
                    save_merged=False,
                    save_strategy=args.merge_strategy,
                    save_ratio=args.merge_ratio,
                    output_dir=args.output_dir,
                )
                if tm_results:
                    all_results[name]["token_merging"] = tm_results

            # Apply token merging and save model
            if args.apply_tome and is_vision:
                apply_and_save_tome(
                    model, name, forward_fn, sample_input,
                    strategy=args.merge_strategy, ratio=args.merge_ratio,
                    output_dir=args.output_dir,
                )

            # Free memory
            del model, sample_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(all_results) > 1:
        compare_results(all_results)

    # Save results
    print("\n[Saving Results]")
    save_results(all_results, args.output_dir)

    print("\n Pipeline complete!\n")


if __name__ == "__main__":
    main()
