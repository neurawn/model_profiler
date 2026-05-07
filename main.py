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
from typing import Optional
import torch
import torch.nn as nn


def _patch_linear_extra_repr():
    """Compat shim for dynamic-quantized Linear pickles across torch versions.

    Older torch bound `_linear_extra_repr` as a method on the dynamic
    quantized Linear class; some pickled models reference it by name, so
    `torch.load` fails with `'Linear' object has no attribute
    '_linear_extra_repr'` on newer torch where the binding moved.
    """
    def _fallback(self):
        try:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}")
        except AttributeError:
            return ""

    try:
        from torch.ao.nn.quantized.dynamic.modules import linear as _qdl
        if not hasattr(_qdl, "_linear_extra_repr"):
            _qdl._linear_extra_repr = _fallback
        if not hasattr(_qdl.Linear, "_linear_extra_repr"):
            _qdl.Linear._linear_extra_repr = _fallback
    except ImportError:
        pass

    if not hasattr(nn.Linear, "_linear_extra_repr"):
        nn.Linear._linear_extra_repr = _fallback


_patch_linear_extra_repr()

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set HuggingFace cache to local model folder
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_DIR

from static_profiler import StaticProfiler
from models import (
    load_gpt2, load_resnet18, load_vit, load_vlm,
)
from token_merging import (
    compare_strategies, apply_token_merging,
)
from quantize import quantize_model
from graph_export import _export_model as export_single_model
from dynamic_profile import run_dynamic_profile
from perplexity import evaluate_perplexity, print_perplexity_table


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
        if "static" not in result:
            continue
        sp = result["static"]
        print(f"  {name:20s}  uniformity = {sp.param_uniformity_score:.4f}")

    print(f"\n{'#'*70}\n")


def save_results(all_results, output_dir):
    """Save all profiling results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    for name, result in all_results.items():
        if not result:
            continue
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        data = {}
        if "static" in result:
            data["static_profile"] = result["static"].to_dict()
        if "graph_profile" in result:
            data["graph_profile"] = result["graph_profile"].to_dict()

        if data:
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


def _find_torchao_act_quant_scale(model: nn.Module):
    """Find the first torchao Int8Tensor weight that carries an act_quant_scale.

    Returns (module_name, scale_tensor, weight_shape) or (None, None, None).
    Tries `module.weight`, `module.weight.data`, and the module's parameter
    dict to deal with Parameter subclass-stripping in different torch versions.
    """
    for name, module in model.named_modules():
        candidates = []
        w = getattr(module, "weight", None)
        if w is not None:
            candidates.append(w)
            d = getattr(w, "data", None)
            if d is not None and d is not w:
                candidates.append(d)
        # Also look in the raw _parameters dict — some torchao paths store the
        # subclassed tensor here without the Parameter wrapper losing attrs.
        params = getattr(module, "_parameters", {}) or {}
        if "weight" in params and params["weight"] is not None:
            candidates.append(params["weight"])

        for cand in candidates:
            scale = getattr(cand, "act_quant_scale", None)
            if scale is not None and hasattr(scale, "shape"):
                return name, scale, tuple(getattr(cand, "shape", ()))
    return None, None, None


def _detect_torchao_calibrated_seq_len(model: nn.Module) -> Optional[int]:
    """Per-token static activation quantization stores a scale of shape
    (..., seq_len, 1). Pull it out of the first quantized weight we find.
    """
    name, scale, _ = _find_torchao_act_quant_scale(model)
    if scale is None:
        return None
    shape = list(scale.shape)
    if len(shape) >= 2 and shape[-1] == 1 and shape[-2] > 1:
        return shape[-2]
    if len(shape) == 1 and shape[0] > 1:
        return shape[0]
    return None


def _strip_torchao_static_act_scales(model: nn.Module) -> int:
    """Convert any torchao Int8 weights with static activation scales to use
    dynamic activation quantization.

    Static scales are tied to the calibration input shape (rank + seq_len),
    which makes a saved model only runnable at that exact shape. Setting
    `act_quant_scale` / `act_quant_zero_point` to None lets torchao compute
    scales on the fly inside `Int8Tensor.from_hp`, which works for any input.
    Returns the number of tensors stripped.
    """
    stripped = 0
    seen = set()
    for module in model.modules():
        for cand in [getattr(module, "weight", None),
                     getattr(getattr(module, "weight", None), "data", None),
                     (getattr(module, "_parameters", {}) or {}).get("weight")]:
            if cand is None or id(cand) in seen:
                continue
            seen.add(id(cand))
            if getattr(cand, "act_quant_scale", None) is not None:
                cand.act_quant_scale = None
                if hasattr(cand, "act_quant_zero_point"):
                    cand.act_quant_zero_point = None
                stripped += 1
    return stripped


def _dump_torchao_quant_info(model: nn.Module) -> None:
    """Print debug info about torchao activation quantization scales found."""
    name, scale, weight_shape = _find_torchao_act_quant_scale(model)
    if scale is None:
        print("  [torchao] No static activation scales found on weights.")
        return
    print(f"  [torchao] Found static activation scale on '{name}':")
    print(f"            weight shape = {weight_shape}")
    print(f"            act_quant_scale shape = {tuple(scale.shape)}")
    print(f"            (per-token scale ⇒ inputs must have seq_len matching "
          f"shape[-2] of the scale)")


def _arch_sample_and_forward(arch: str, seq_len: Optional[int] = None):
    """Return an arch-appropriate (sample_input, forward_fn) without loading weights.

    Shared by `_load_arch_with_weights` and the full-`nn.Module` load path so a
    saved GPT-2 .pt loaded as a module still gets token-ID inputs (not floats,
    which embedding layers reject).

    `seq_len` overrides the default tokenized prompt length for LLMs — needed
    when running torchao-quantized models whose activation scales are baked in
    at a specific seq_len.
    """
    a = arch.lower()
    if arch in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        if seq_len is not None:
            # Random token IDs; vocab size for GPT-2 is 50257.
            sample_input = torch.randint(0, 50257, (1, seq_len), dtype=torch.long)
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)
            sample_input = tokenizer(
                "The quick brown fox jumps over the lazy dog",
                return_tensors="pt",
            )["input_ids"]
        return sample_input, lambda m, x: m(x, labels=x)
    if "vit" in a:
        return torch.randn(1, 3, 224, 224), lambda m, x: m(pixel_values=x)
    if "clip" in a:
        sample_input = torch.randn(1, 3, 224, 224)
        dummy_ids = torch.randint(0, 49408, (1, 77))
        dummy_mask = torch.ones(1, 77, dtype=torch.long)
        return sample_input, lambda m, x: m(
            pixel_values=x, input_ids=dummy_ids, attention_mask=dummy_mask)
    if a in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        return torch.randn(1, 3, 224, 224), None
    return None, None


def _restore_hf_conv1d_layout(state_dict, model, force_transpose_names=None):
    """Transpose weights for any HF Conv1D module whose state_dict entry has
    the nn.Linear (transposed) shape — undoes the layout swap done by
    `_convert_hf_conv1d_to_linear` during the SparseGPT 2:4 conversion.

    Args:
        state_dict: state dict (possibly with Linear-shaped weights for what
            should be Conv1D modules in `model`).
        model: destination module — Conv1D modules drive which keys to fix.
        force_transpose_names: optional iterable of module names known to
            have been Conv1D→Linear converted. Required for *square* Conv1D
            modules (e.g. GPT-2's attn.c_proj is 768×768) where the shape
            check can't tell whether the weight is already in the right
            orientation; the explicit list lets us transpose them.

    Returns a (shallow-copied) dict so the caller's state_dict isn't mutated.
    No-op when there are no Conv1D modules or shapes already align.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return state_dict

    force_set = set(force_transpose_names or [])
    fixed = dict(state_dict)
    n_fixed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, Conv1D):
            continue
        wkey = f"{name}.weight"
        w = fixed.get(wkey)
        if w is None or not hasattr(w, "shape"):
            continue
        target_shape = tuple(mod.weight.shape)
        src_shape = tuple(w.shape)
        if src_shape == target_shape:
            # Shape already matches. For square Conv1D layers the saved
            # weight may STILL be transposed (values, not shape) — that's
            # the ambiguous case force_transpose_names exists for.
            if (len(src_shape) == 2 and src_shape[0] == src_shape[1]
                    and name in force_set):
                fixed[wkey] = w.t().contiguous()
                n_fixed += 1
            continue
        if len(src_shape) == 2 and src_shape == target_shape[::-1]:
            fixed[wkey] = w.t().contiguous()
            n_fixed += 1
    if n_fixed:
        print(f"  Restored HF Conv1D layout for {n_fixed} weight(s) "
              f"(transposed from nn.Linear shape — saved by 2:4 prune flow)")
    return fixed


def _load_arch_with_weights(arch: str, weight_path: str,
                            state_dict: dict = None,
                            is_safetensors: bool = False,
                            force_transpose_names=None):
    """
    Instantiate a model architecture and load weights into it.

    Returns (model, forward_fn, sample_input).
    sample_input is arch-appropriate (token IDs for LLMs, images for vision).
    """
    forward_fn = None
    sample_input = None

    # GPT-2 family
    if arch in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print(f"  Architecture: GPT2LMHeadModel ({arch})")
        model = GPT2LMHeadModel.from_pretrained(arch, cache_dir=MODEL_DIR)
        tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)
        tokens = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
        sample_input = tokens["input_ids"]
        forward_fn = lambda m, x: m(x, labels=x)

    # ViT family
    elif "vit" in arch.lower():
        from transformers import ViTForImageClassification
        print(f"  Architecture: ViTForImageClassification ({arch})")
        model = ViTForImageClassification.from_pretrained(arch, cache_dir=MODEL_DIR)
        sample_input = torch.randn(1, 3, 224, 224)
        forward_fn = lambda m, x: m(pixel_values=x)

    # CLIP / VLM
    elif "clip" in arch.lower():
        from transformers import CLIPModel
        print(f"  Architecture: CLIPModel ({arch})")
        model = CLIPModel.from_pretrained(arch, cache_dir=MODEL_DIR)
        sample_input = torch.randn(1, 3, 224, 224)
        dummy_input_ids = torch.randint(0, 49408, (1, 77))
        dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)
        forward_fn = lambda m, x: m(pixel_values=x, input_ids=dummy_input_ids,
                                     attention_mask=dummy_attention_mask)

    # ResNet
    elif arch.lower() in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        import torchvision.models as tv_models
        print(f"  Architecture: {arch}")
        model = getattr(tv_models, arch.lower())(weights=None)
        sample_input = torch.randn(1, 3, 224, 224)

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
        # The 2:4 prune flow replaces HF Conv1D modules with nn.Linear and
        # transposes their weights. A fresh GPT-2 still has Conv1D modules,
        # so any saved Linear-layout weights need to be transposed back to
        # match. No-op when the shapes already align.
        sd = _restore_hf_conv1d_layout(
            sd, model, force_transpose_names=force_transpose_names)

        # Try strict load first, fall back to non-strict
        try:
            model.load_state_dict(sd, strict=True)
            print(f"  Loaded weights (strict)")
        except RuntimeError:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Loaded weights (non-strict): "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

    model.eval()
    return model, forward_fn, sample_input


def _gpt2_arch_name(arch_or_model: Optional[str]) -> Optional[str]:
    """Normalize --arch / --model to a HuggingFace GPT-2 checkpoint id, or None."""
    if not arch_or_model:
        return None
    a = arch_or_model.lower()
    if a in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        return a
    return None


def _evaluate_loaded_model_perplexity(model, args, label: str, notes: str = ""):
    """Run WikiText-2 perplexity on an already-loaded GPT-2-family model.
    Pulls tokenizer based on --arch / --model. Returns the PerplexityResult
    or None if not a supported architecture.
    """
    arch = _gpt2_arch_name(args.arch) or _gpt2_arch_name(args.model)
    if arch is None:
        print("  [perplexity] Skipped: --perplexity currently supports GPT-2 "
              "family only. Use --arch gpt2 or --model gpt2.")
        return None
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_perplexity(
        model, tokenizer, label=label, device=device,
        max_length=args.ppl_max_length, stride=args.ppl_stride,
        max_tokens=args.ppl_max_tokens, notes=notes,
    )


def run_perplexity_compare(args):
    """Two-way WikiText-2 perplexity comparison:
      1. base GPT-2 (FP32 reference)
      2. plan-quantized checkpoint (mixed INT4/INT8/FP16)

    For pruning comparisons, run --prune-sparsegpt --save-pruned to produce
    a pruned checkpoint, then `--compare-perplexity --local <each path>`.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    arch = _gpt2_arch_name(args.arch) or _gpt2_arch_name(args.model) or "gpt2"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.quantized_checkpoint

    print(f"\n{'#'*80}")
    print(f"#  WIKITEXT-2 PERPLEXITY COMPARISON")
    print(f"#  arch={arch}  device={device}  max_length={args.ppl_max_length}  "
          f"stride={args.ppl_stride}")
    if args.ppl_max_tokens:
        print(f"#  max_tokens={args.ppl_max_tokens:,} (truncated for speed)")
    print(f"#  quantized checkpoint: {ckpt_path}")
    print(f"{'#'*80}")

    tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)
    rows = []

    print(f"\n[1/2] Loading {arch} (FP32 baseline)...")
    base = GPT2LMHeadModel.from_pretrained(arch, cache_dir=MODEL_DIR)
    rows.append(evaluate_perplexity(
        base, tokenizer, label=f"{arch} (FP32 base)", device=device,
        max_length=args.ppl_max_length, stride=args.ppl_stride,
        max_tokens=args.ppl_max_tokens, notes="reference",
    ))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"\n[2/2] SKIPPED — checkpoint not found: {ckpt_path}")
        print_perplexity_table(rows)
        return

    print(f"\n[2/2] Loading plan-quantized checkpoint: {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        sd = loaded["model_state_dict"]
        avg_bits = loaded.get("avg_bits")
    else:
        sd = loaded
        avg_bits = None
    quant = GPT2LMHeadModel.from_pretrained(arch, cache_dir=MODEL_DIR)
    missing, unexpected = quant.load_state_dict(sd, strict=False)
    # `_wq_scale` / `_wq_bits` buffers in the saved dict are diagnostic and
    # land in `unexpected`; weights are pre-dequantized so the model is
    # ready to evaluate as-is.
    print(f"  Loaded (non-strict): {len(missing)} missing, "
          f"{len(unexpected)} unexpected (expected: _wq_* buffers)")
    quant_notes = "plan quant"
    if avg_bits:
        quant_notes += f", avg {avg_bits:.1f} bits"
    rows.append(evaluate_perplexity(
        quant, tokenizer, label=os.path.basename(ckpt_path), device=device,
        max_length=args.ppl_max_length, stride=args.ppl_stride,
        max_tokens=args.ppl_max_tokens, notes=quant_notes,
    ))
    del quant
    if device == "cuda":
        torch.cuda.empty_cache()

    print_perplexity_table(rows)


def run_prune(model, model_name, args, output_dir):
    """Drive a SparseGPT pruning pass on a loaded GPT-2 model.

    Loads WikiText-2 calibration via the active GPT-2 tokenizer, runs the
    block-by-block algorithm in `sparsegpt.py`, and (if --save-pruned was
    given) persists the result to disk.
    """
    arch = _gpt2_arch_name(args.arch) or _gpt2_arch_name(args.model)
    if arch is None:
        print("Error: --prune requires the GPT-2 family. "
              "Use --arch gpt2|gpt2-medium|gpt2-large|gpt2-xl.")
        sys.exit(1)

    from transformers import GPT2Tokenizer
    from sparsegpt import (
        load_wikitext_calibration, prune_gpt2_sparsegpt, save_sparsegpt_model,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pattern = args.prune
    sparsity = 0.5 if pattern == "2:4" else args.prune_sparsity

    print(f"\n{'#'*80}")
    print(f"#  SPARSEGPT PRUNING: {model_name}")
    print(f"#  pattern={pattern}  target_sparsity={sparsity:.2f}  "
          f"device={device}")
    print(f"#  calibration: WikiText-2 train, "
          f"{args.prune_samples} samples × {args.prune_seqlen} tokens")
    print(f"{'#'*80}")

    tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)
    calib = load_wikitext_calibration(
        tokenizer,
        n_samples=args.prune_samples,
        seqlen=args.prune_seqlen,
    )

    # 2:4 + CUDA → also wrap as SparseSemiStructuredTensor for tensor-core
    # speedup. Anything else stays dense (the zeros are still in place; just
    # no kernel acceleration).
    convert_to_sparse = (pattern == "2:4" and device == "cuda")

    result = prune_gpt2_sparsegpt(
        model, calib,
        pattern=pattern, sparsity=sparsity,
        device=device,
        convert_to_sparse=convert_to_sparse,
        sparse_dtype=torch.float16,
    )

    if args.save_pruned is not None:
        explicit = None if args.save_pruned == "auto" else args.save_pruned
        save_sparsegpt_model(
            model, model_name, result,
            save_path=explicit, output_dir=output_dir,
            sparse_dtype=torch.float16 if convert_to_sparse else None,
        )

    return result


def _load_local_for_perplexity(path: str, arch: str):
    """Load a local checkpoint into a fresh `arch` model for perplexity eval.

    Handles three layouts: safetensors, full nn.Module pickle, and state_dict
    (including the wrapper-dict produced by --apply-quantization and the 2:4
    pruning save flow). Returns the ready-to-eval model.
    """
    is_safetensors = path.endswith(".safetensors") or _is_safetensors(path)
    if is_safetensors:
        if not arch:
            raise ValueError("--arch is required for safetensors checkpoints")
        model, _, _ = _load_arch_with_weights(arch, path, is_safetensors=True)
        return model

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, nn.Module):
        return loaded
    if not isinstance(loaded, dict):
        raise ValueError(f"unexpected file content type: {type(loaded)}")
    if not arch:
        raise ValueError("--arch is required for state_dict checkpoints")

    sd = loaded
    pruned_names = None
    if "model_state_dict" in loaded and isinstance(
            loaded["model_state_dict"], dict):
        meta_keys = [k for k in loaded if k != "model_state_dict"]
        print(f"  Unwrapping checkpoint (other keys: {meta_keys})")
        sd = loaded["model_state_dict"]
        # Only thread pruned_layers as force_transpose_names when the saver
        # converted Conv1D → Linear (Linear-layout on disk). For native
        # Conv1D-layout saves (SparseGPT unstructured), force-transpose
        # would corrupt the square attn.c_proj weights.
        if loaded.get("weights_in_linear_layout", True):
            pruned_names = loaded.get("pruned_layers")

    model, _, _ = _load_arch_with_weights(
        arch, path, state_dict=sd, force_transpose_names=pruned_names)
    return model


def run_compare_perplexity(args):
    """N-way WikiText-2 perplexity comparison across --local checkpoints.

    Loads each path one at a time (so peak memory stays at ~1 model), runs
    sliding-window perplexity, then prints the comparison table. The first
    --local entry is treated as the baseline for the Δ columns.
    """
    paths = args.local or []
    if not paths:
        print("Error: --compare-perplexity requires one or more --local <path>")
        sys.exit(1)

    arch = _gpt2_arch_name(args.arch) or _gpt2_arch_name(args.model)
    if arch is None:
        print("Error: --compare-perplexity currently supports the GPT-2 "
              "family only. Use --arch gpt2|gpt2-medium|gpt2-large|gpt2-xl.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*80}")
    print(f"#  WIKITEXT-2 PERPLEXITY COMPARE  (n={len(paths)})")
    print(f"#  arch={arch}  device={device}  "
          f"max_length={args.ppl_max_length}  stride={args.ppl_stride}")
    if args.ppl_max_tokens:
        print(f"#  max_tokens={args.ppl_max_tokens:,} (truncated)")
    for i, p in enumerate(paths):
        print(f"#    [{i+1}] {p}")
    print(f"{'#'*80}")

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(arch, cache_dir=MODEL_DIR)

    rows = []
    for i, raw_path in enumerate(paths):
        path = os.path.abspath(raw_path)
        label = os.path.basename(path)
        print(f"\n[{i+1}/{len(paths)}] Loading {label}")
        if not os.path.isfile(path):
            print(f"  SKIP — file not found: {path}")
            continue
        try:
            model = _load_local_for_perplexity(path, arch)
        except Exception as e:
            print(f"  Error loading {label}: {e}")
            import traceback
            traceback.print_exc()
            continue

        notes = f"local: {label}"
        try:
            rows.append(evaluate_perplexity(
                model, tokenizer, label=label, device=device,
                max_length=args.ppl_max_length, stride=args.ppl_stride,
                max_tokens=args.ppl_max_tokens, notes=notes,
            ))
        except Exception as e:
            print(f"  Error evaluating {label}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print_perplexity_table(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Model Profiler & Compression Recommender"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gpt2", "gpt2-medium", "gpt2-large",
                                 "resnet", "vit", "vlm"],
                        help="Which model to profile")
    parser.add_argument("--local", type=str, default=None, action="append",
                        help="Path to a local PyTorch model file (.pt/.pth). "
                             "Overrides --model. Provide --input-shape for non-default input. "
                             "May be passed multiple times when used with "
                             "--compare-perplexity to evaluate several checkpoints.")
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
    parser.add_argument("--quantize-suggested", action="store_true",
                        help="Run graph analysis, then apply the compression plan's "
                             "per-layer mixed-precision quantization and save the model")
    parser.add_argument("--dynamic-profile", action="store_true",
                        help="Run torch.profiler dynamic profiling: per-op latency, "
                             "memory usage, tensor shapes, call stacks, FLOPs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for dynamic profiling (cpu/cuda, auto-detected)")
    parser.add_argument("--save-trace", action="store_true",
                        help="Save Chrome trace JSON for visualization in chrome://tracing")
    parser.add_argument("--profile-runs", type=int, default=20,
                        help="Number of profiled forward passes (default: 20)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override LLM input sequence length. Required when "
                             "loading a torchao-quantized model whose activation "
                             "scales were calibrated at a fixed seq_len.")
    parser.add_argument("--save-pruned", type=str, nargs="?", const="auto",
                        default=None,
                        help="Save the SparseGPT-pruned model. With no "
                             "argument, writes to "
                             "<output-dir>/<name>_sparsegpt_<pattern>.pt. "
                             "Pass an explicit path to override. Sparse-tensor "
                             "weights are materialized to dense (zeros "
                             "preserved) so the file loads portably.")
    parser.add_argument("--prune", type=str, default=None,
                        choices=["2:4", "unstructured"],
                        help="Apply SparseGPT one-shot Hessian-based pruning "
                             "(Frantar & Alistarh 2023) using a WikiText-2 "
                             "calibration set. GPT-2 family only. '2:4' "
                             "yields exact 50%% sparsity in cuSPARSELt-"
                             "compatible groups (auto-converts to sparse "
                             "tensor cores on CUDA). 'unstructured' picks "
                             "lowest-importance weights anywhere; target "
                             "ratio set by --prune-sparsity. Combine with "
                             "--save-pruned to persist.")
    parser.add_argument("--prune-sparsity", type=float, default=0.5,
                        help="Target sparsity for unstructured pruning "
                             "(ignored when --prune=2:4). Default: 0.5.")
    parser.add_argument("--prune-samples", type=int, default=128,
                        help="Number of WikiText-2 calibration sequences "
                             "(default 128 — paper recipe). Lower for "
                             "faster iteration; 32 is usually enough for "
                             "a sanity-check run.")
    parser.add_argument("--prune-seqlen", type=int, default=1024,
                        help="Calibration sequence length (default 1024 = "
                             "GPT-2 max ctx).")
    parser.add_argument("--perplexity", action="store_true",
                        help="Evaluate WikiText-2 perplexity on the loaded model "
                             "(GPT-2 family only). Combine with --model gpt2 or "
                             "--local <path> --arch gpt2.")
    parser.add_argument("--perplexity-compare", action="store_true",
                        help="Run a 3-way WikiText-2 perplexity comparison: "
                             "base GPT-2, plan-quantized checkpoint (see "
                             "--quantized-checkpoint), and that checkpoint with "
                             "2:4 magnitude pruning applied. Prints a table.")
    parser.add_argument("--compare-perplexity", action="store_true",
                        help="N-way WikiText-2 perplexity comparison across "
                             "multiple --local checkpoints. Pass --local "
                             "<path> once per model and a shared --arch. "
                             "Example: --compare-perplexity --local a.pt "
                             "--local b.pt --arch gpt2 --device cuda")
    parser.add_argument("--quantized-checkpoint", type=str,
                        default="./gpt_2_plan_quantized.pt",
                        help="Path to a plan-quantized .pt produced by "
                             "--apply-quantization. Used by --perplexity-compare.")
    parser.add_argument("--ppl-max-length", type=int, default=1024,
                        help="Sliding-window context length for perplexity "
                             "(default: 1024 = GPT-2 ctx).")
    parser.add_argument("--ppl-stride", type=int, default=512,
                        help="Sliding-window stride for perplexity (default: 512).")
    parser.add_argument("--ppl-max-tokens", type=int, default=None,
                        help="Cap total tokens evaluated for perplexity. None = "
                             "full WikiText-2 test split (~287K tokens).")
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL PROFILER & COMPRESSION RECOMMENDER")
    print("  ML to Improve ML — NYU Final Project")
    print("=" * 70)
    print(f"  Target: {args.target_device}")
    print("=" * 70)

    # --quantize-suggested implies --graph
    if args.quantize_suggested:
        args.graph = True

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]

    # Standalone perplexity comparison: handles all model loading itself.
    if args.perplexity_compare:
        run_perplexity_compare(args)
        return

    # N-way perplexity compare across multiple --local checkpoints.
    if args.compare_perplexity:
        run_compare_perplexity(args)
        return

    # Initialize profiler
    static_profiler = StaticProfiler()

    # ── Local model path ──
    if args.local:
        if len(args.local) > 1:
            print("Note: multiple --local paths given; using the first one for "
                  "the standard pipeline. Pass --compare-perplexity to evaluate "
                  "all of them.")
        local_path = os.path.abspath(args.local[0])
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
        arch_sample = None
        is_safetensors = local_path.endswith(".safetensors") or _is_safetensors(local_path)

        if is_safetensors:
            # SafeTensors file — requires --arch to know the model structure
            if not args.arch:
                print("Error: safetensors file detected. Use --arch to specify the model "
                      "architecture (e.g. --arch gpt2, --arch google/vit-base-patch16-224)")
                sys.exit(1)
            local_model, forward_fn, arch_sample = _load_arch_with_weights(
                args.arch, local_path, is_safetensors=True)
        else:
            try:
                loaded = torch.load(local_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Error loading file: {e}")
                sys.exit(1)

            if isinstance(loaded, nn.Module):
                local_model = loaded
                # A saved nn.Module already has weights, but we still need
                # arch-appropriate sample input + forward_fn (e.g. token IDs
                # for GPT-2). Derive them from --arch when provided.
                if args.arch:
                    seq_len = args.seq_len
                    if seq_len is None:
                        # torchao static-quant models bake activation scales at
                        # a fixed seq_len; auto-detect to avoid shape asserts.
                        seq_len = _detect_torchao_calibrated_seq_len(loaded)
                        if seq_len is not None:
                            print(f"  Detected torchao calibrated seq_len={seq_len}")
                        else:
                            _dump_torchao_quant_info(loaded)
                    arch_sample, forward_fn = _arch_sample_and_forward(
                        args.arch, seq_len=seq_len)
                # Static activation scales also encode rank — they only work at
                # the exact calibration shape. For dynamic profiling we want
                # arbitrary inputs, so convert to dynamic activation quant.
                if args.dynamic_profile:
                    n = _strip_torchao_static_act_scales(loaded)
                    if n:
                        print(f"  Stripped static activation scales from {n} "
                              f"torchao weight(s) (using dynamic act quant)")
            elif isinstance(loaded, dict):
                # state_dict — requires --arch
                if not args.arch:
                    print("Error: file contains a state_dict. Use --arch to specify the model "
                          "architecture (e.g. --arch gpt2, --arch resnet18)")
                    sys.exit(1)
                # Saved checkpoints from this repo wrap the state dict under
                # "model_state_dict" alongside metadata (e.g. plan_quantized,
                # tome, pruned). Unwrap so load_state_dict sees real param keys.
                sd = loaded
                pruned_names = None
                if "model_state_dict" in loaded and isinstance(
                        loaded["model_state_dict"], dict):
                    print(f"  Unwrapping checkpoint: state_dict under "
                          f"'model_state_dict' key (other keys: "
                          f"{[k for k in loaded if k != 'model_state_dict']})")
                    sd = loaded["model_state_dict"]
                    # Pruned-2:4 checkpoints went through a Conv1D→Linear
                    # transpose; thread the layer list to disambiguate
                    # square Conv1D modules during layout restoration. Skip
                    # this for SparseGPT-unstructured saves where weights
                    # stay in native Conv1D layout (no transpose needed).
                    if loaded.get("weights_in_linear_layout", True):
                        pruned_names = loaded.get("pruned_layers")
                local_model, forward_fn, arch_sample = _load_arch_with_weights(
                    args.arch, local_path, state_dict=sd,
                    force_transpose_names=pruned_names)
            else:
                print(f"Error: unexpected type in file: {type(loaded)}")
                sys.exit(1)

        local_name = os.path.splitext(os.path.basename(local_path))[0]
        # Use arch-specific sample input if available, otherwise generate from input_shape
        if args.arch and arch_sample is not None:
            sample_input = arch_sample
        else:
            sample_input = torch.randn(*input_shape)
        print(f"  Loaded {local_name} ({sum(p.numel() for p in local_model.parameters()):,} params)")

        all_results = {}
        run_static = not args.dynamic_profile or args.graph or args.quantize or args.quantize_suggested
        if run_static:
            result = profile_single_model(
                local_name, local_model, sample_input, forward_fn,
                static_profiler, target_device=args.target_device,
            )
            all_results[local_name] = result
        else:
            all_results[local_name] = {}

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
                if args.quantize_suggested and gsp.compression_plan:
                    apply_quant_plan(
                        local_model, local_name, gsp.compression_plan,
                        sample_input=sample_input, forward_fn=forward_fn,
                        output_dir=args.output_dir,
                    )

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

        # SparseGPT pruning runs before dynamic profile / perplexity so the
        # downstream evaluation operates on the pruned model.
        if args.prune is not None:
            run_prune(
                local_model, local_name, args, args.output_dir,
            )

        if args.dynamic_profile:
            device = args.device or (
                "cuda" if torch.cuda.is_available() else "cpu")
            run_dynamic_profile(
                local_model, sample_input, model_name=local_name,
                forward_fn=forward_fn, device=device,
                num_runs=args.profile_runs, save_trace=args.save_trace,
                output_dir=args.output_dir,
            )

        if args.perplexity:
            _evaluate_loaded_model_perplexity(
                local_model, args, label=local_name,
                notes=f"loaded from {os.path.basename(local_path)}",
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
            run_static = not args.dynamic_profile or args.graph or args.quantize or args.quantize_suggested
            if run_static:
                result = profile_single_model(
                    name, model, sample_input, forward_fn,
                    static_profiler, target_device=args.target_device,
                )
                all_results[name] = result
            else:
                all_results[name] = {}

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
                    if args.quantize_suggested and gsp.compression_plan:
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

            # SparseGPT pruning runs before dynamic profile / perplexity so
            # the downstream evaluation operates on the pruned model.
            if args.prune is not None:
                run_prune(model, name, args, args.output_dir)

            if args.dynamic_profile:
                device = args.device or (
                    "cuda" if torch.cuda.is_available() else "cpu")
                run_dynamic_profile(
                    model, sample_input, model_name=name,
                    forward_fn=forward_fn, device=device,
                    num_runs=args.profile_runs, save_trace=args.save_trace,
                    output_dir=args.output_dir,
                )

            if args.perplexity:
                _evaluate_loaded_model_perplexity(
                    model, args, label=name, notes="built-in model",
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
