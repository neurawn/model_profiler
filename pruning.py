"""
2:4 Structured Pruning for Ampere+ Sparse Tensor Cores
======================================================
Magnitude-based 2:4 semi-structured sparsity for nn.Linear (and HuggingFace
Conv1D, which is layout-transposed Linear).

For every 4 contiguous weights along the input/K dim of each output row, keep
the 2 with largest magnitude and zero the rest. Optionally convert pruned
weights to `torch.sparse.SparseSemiStructuredTensor` so GEMM dispatches to
sparse tensor cores via cuSPARSELt on Ampere+ — roughly halves linear-layer
time on supported shapes (K divisible by 16 for fp16/bf16).

This is a post-training transformation. A short calibration pass (running a
few forward passes on representative input) is included to verify the pruned
model produces sane output shapes/dtypes before profiling or saving.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Sequence, Any


@dataclass
class PruneResult:
    """Summary of a 2:4 pruning pass."""
    pruned_layers: List[str] = field(default_factory=list)
    converted_sparse: List[str] = field(default_factory=list)
    skipped_layers: List[Tuple[str, str]] = field(default_factory=list)
    total_zeros: int = 0
    total_weights: int = 0

    @property
    def sparsity(self) -> float:
        return self.total_zeros / max(self.total_weights, 1)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  2:4 STRUCTURED PRUNING SUMMARY",
            "=" * 70,
            f"  Pruned layers:     {len(self.pruned_layers)}",
            f"  Sparse-converted:  {len(self.converted_sparse)}",
            f"  Skipped:           {len(self.skipped_layers)}",
            f"  Total weights:     {self.total_weights:,}",
            f"  Achieved sparsity: {self.sparsity * 100:.2f}%",
        ]
        if self.skipped_layers:
            lines.append("  Skipped (first 10):")
            for name, reason in self.skipped_layers[:10]:
                lines.append(f"    {name}: {reason}")
            if len(self.skipped_layers) > 10:
                lines.append(f"    ... ({len(self.skipped_layers) - 10} more)")
        return "\n".join(lines)


# ── Mask construction ────────────────────────────────────────────────────────

def magnitude_2_4_mask(weight: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bool mask with the 2:4 pattern along `dim` — for every 4 contiguous
    elements, mark the 2 largest by magnitude.
    """
    if dim < 0:
        dim = weight.ndim + dim
    if weight.shape[dim] % 4 != 0:
        raise ValueError(
            f"dim {dim} (size={weight.shape[dim]}) must be divisible by 4")

    perm = list(range(weight.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    w = weight.permute(*perm).contiguous()  # (..., K)

    grouped = w.abs().reshape(*w.shape[:-1], -1, 4)  # (..., K/4, 4)
    _, top2_idx = grouped.topk(2, dim=-1)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, top2_idx, True)
    mask = mask.reshape(*w.shape)

    inv = [perm.index(i) for i in range(weight.ndim)]
    return mask.permute(*inv).contiguous()


# ── Per-layer application ───────────────────────────────────────────────────

def _hf_conv1d_cls():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:
        return None


def _convert_hf_conv1d_to_linear(model: nn.Module) -> int:
    """Replace HuggingFace Conv1D modules with functionally-equivalent
    nn.Linear (with transposed weight). cuSPARSELt's 2:4 path only
    accelerates `sparse @ dense`, so we need the sparse weight on the
    left of the matmul. nn.Linear's `F.linear(x, W) = x @ W.T` puts it
    there; HF Conv1D's `x @ W` does not.

    Returns the number of modules replaced.
    """
    Conv1D = _hf_conv1d_cls()
    if Conv1D is None:
        return 0
    converted = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, Conv1D):
                continue
            in_features = child.weight.shape[0]
            out_features = child.weight.shape[1]
            linear = nn.Linear(
                in_features, out_features,
                bias=child.bias is not None,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                linear.weight.copy_(child.weight.t().contiguous())
                if child.bias is not None:
                    linear.bias.copy_(child.bias)
            setattr(parent, child_name, linear)
            converted += 1
    return converted


def _k_dim_for(module: nn.Module) -> Optional[int]:
    """Which weight dim is the GEMM contraction (K)?

    nn.Linear weight: [out, in] → K is dim=1.
    HF Conv1D weight: [in, out] → K is dim=0 (forward does x @ W).
    """
    if isinstance(module, nn.Linear):
        return 1
    Conv1D = _hf_conv1d_cls()
    if Conv1D is not None and isinstance(module, Conv1D):
        return 0
    return None


def apply_2_4_mask(module: nn.Module) -> int:
    """Apply 2:4 magnitude mask to a Linear/Conv1D weight in-place. Returns
    the number of zeros introduced."""
    k_dim = _k_dim_for(module)
    if k_dim is None:
        raise TypeError(f"Unsupported module type: {type(module).__name__}")
    mask = magnitude_2_4_mask(module.weight.data, dim=k_dim)
    module.weight.data.mul_(mask)
    return int((~mask).sum().item())


# ── Sparse-tensor-core conversion ───────────────────────────────────────────

def _to_sparse_semi_structured(
    module: nn.Module, dtype: torch.dtype,
) -> Tuple[bool, str]:
    """Replace `module.weight` with a SparseSemiStructuredTensor so F.linear
    dispatches to sparse tensor cores. Returns (success, reason)."""
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError:
        return False, "torch.sparse.to_sparse_semi_structured unavailable"
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    w = module.weight.data
    k_dim = _k_dim_for(module)
    # cuSPARSELt requires K divisible by 16 for fp16/bf16, 32 for int8.
    min_k = 32 if dtype == torch.int8 else 16
    if w.shape[k_dim] % min_k != 0:
        return False, (
            f"K={w.shape[k_dim]} not divisible by {min_k} for dtype {dtype}")

    try:
        w_dev = w.to(device="cuda", dtype=dtype).contiguous()
        sparse = to_sparse_semi_structured(w_dev)
        module.weight = nn.Parameter(sparse, requires_grad=False)
        if getattr(module, "bias", None) is not None:
            module.bias.data = module.bias.data.to(device="cuda", dtype=dtype)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ── Top-level entry ─────────────────────────────────────────────────────────

def prune_model_2_4(
    model: nn.Module,
    skip_substrings: Optional[Sequence[str]] = ("lm_head", "embed", "wte", "wpe"),
    convert_to_sparse: bool = True,
    sparse_dtype: torch.dtype = torch.float16,
    calibration_input: Any = None,
    calibration_forward_fn: Optional[Callable] = None,
    n_calibration: int = 3,
    verbose: bool = True,
) -> PruneResult:
    """Apply 2:4 magnitude pruning across all eligible Linear/Conv1D layers.

    Args:
        model: model to mutate in place.
        skip_substrings: leave layers whose qualified name contains any of
            these substrings dense (defaults skip embeddings + LM head).
        convert_to_sparse: after masking, convert weights to
            SparseSemiStructuredTensor for cuSPARSELt acceleration. Requires
            CUDA + K divisible by 16 (fp16/bf16) or 32 (int8).
        sparse_dtype: dtype for sparse-converted weights.
        calibration_input / calibration_forward_fn: representative input
            and forward callable (`fwd(model, x)`). When both provided,
            run `n_calibration` forward passes to verify the pruned model
            doesn't error out.
        verbose: print progress.

    Returns: PruneResult.
    """
    skips = list(skip_substrings or [])

    # cuSPARSELt only accelerates sparse@dense. HF Conv1D's `x @ W` puts the
    # weight on the right, so the sparse path raises NotImplementedError on
    # the transpose-trick fallback. Replace with nn.Linear up front when we
    # plan to convert to SparseSemiStructuredTensor.
    if convert_to_sparse:
        n_converted = _convert_hf_conv1d_to_linear(model)
        if n_converted and verbose:
            print(f"  Converted {n_converted} HF Conv1D → nn.Linear "
                  f"(required for cuSPARSELt sparse@dense dispatch)")

    Conv1D = _hf_conv1d_cls()
    target_types: Tuple[type, ...] = (nn.Linear,)
    if Conv1D is not None:
        target_types = target_types + (Conv1D,)

    result = PruneResult()

    for name, module in model.named_modules():
        if not isinstance(module, target_types):
            continue
        if any(s in name for s in skips):
            result.skipped_layers.append((name, "name match in skip list"))
            continue

        k_dim = _k_dim_for(module)
        k_size = module.weight.shape[k_dim]
        if k_size % 4 != 0:
            result.skipped_layers.append(
                (name, f"K={k_size} not divisible by 4"))
            continue

        zeros = apply_2_4_mask(module)
        result.pruned_layers.append(name)
        result.total_zeros += zeros
        result.total_weights += module.weight.numel()

        if convert_to_sparse:
            ok, reason = _to_sparse_semi_structured(module, dtype=sparse_dtype)
            if ok:
                result.converted_sparse.append(name)
            else:
                result.skipped_layers.append((name, f"sparse-convert: {reason}"))

    if verbose:
        print(result.summary())

    if calibration_input is not None and calibration_forward_fn is not None:
        if verbose:
            print(f"  Calibration: {n_calibration} forward pass(es)...")
        model.eval()
        device = next(model.parameters()).device
        x = calibration_input.to(device) if isinstance(
            calibration_input, torch.Tensor) else calibration_input
        with torch.no_grad():
            for _ in range(n_calibration):
                calibration_forward_fn(model, x)
        if verbose:
            print("  Calibration: ok")

    return result


# ── Before/after comparison for dynamic profiling ───────────────────────────

def _aggregate_op_costs(result, device: str = "cuda"):
    """Group `DynamicProfileResult.op_profiles` by op name and return
    (per_op_dict, total_self_time, total_flops). Self-time matches
    torch.profiler's `Self CUDA %` denominator.
    """
    from dynamic_profile import _PRUNABLE_OP_KEYWORDS
    use_cuda = device == "cuda"
    agg = {}
    total_time = 0.0
    total_flops = 0
    for op in result.op_profiles:
        st = op.self_cuda_time_us if use_cuda else op.self_cpu_time_us
        total_time += st
        total_flops += op.flops
        if not any(k in op.name.lower() for k in _PRUNABLE_OP_KEYWORDS):
            continue
        a = agg.setdefault(op.name, {"self_time": 0.0, "flops": 0})
        a["self_time"] += st
        a["flops"] += op.flops
    return agg, max(total_time, 1.0), max(total_flops, 1)


def print_prune_comparison(before, after, device: str = "cuda") -> None:
    """Side-by-side comparison of prunable-op cost before vs after pruning."""
    b_agg, b_total_time, b_total_flops = _aggregate_op_costs(before, device)
    a_agg, a_total_time, a_total_flops = _aggregate_op_costs(after, device)

    label = "CUDA" if device == "cuda" else "CPU"
    all_ops = sorted(
        set(b_agg) | set(a_agg),
        key=lambda n: -b_agg.get(n, {}).get("self_time", 0.0),
    )

    def _tput(flops, self_time_us):
        # TFLOPS/sec = (flops / sec) / 1e12; self_time is in microseconds.
        if self_time_us <= 0 or flops <= 0:
            return 0.0
        return (flops / 1e6) / self_time_us

    width = 138
    print()
    print("=" * width)
    print(f"  2:4 PRUNING COMPARISON  (Self {label} % / GFLOPs / "
          f"effective TFLOPS/s, before → after)")
    print("=" * width)
    print(f"  Total self {label} time:  "
          f"before={b_total_time/1000:.3f} ms, "
          f"after={a_total_time/1000:.3f} ms  "
          f"(Δ {(a_total_time - b_total_time)/1000:+.3f} ms, "
          f"{100.0*(a_total_time - b_total_time)/b_total_time:+.1f}%)")
    print(f"  Total dense GFLOPs:     "
          f"before={b_total_flops/1e9:.3f}, after={a_total_flops/1e9:.3f}")
    print(f"  Note: TFLOPS/s uses dense FLOPs counter — for sparse mm the "
          f"\"real\" multiplies are half, so a ~2× jump = sparse tensor cores "
          f"engaged.")
    print("-" * width)
    print(f"  {'Op':<30s} "
          f"{'B %t':>8s} {'A %t':>8s} {'Δ %t':>8s} "
          f"{'B GF':>8s} {'B %F':>8s} {'B TF/s':>8s} "
          f"{'A GF':>8s} {'A %F':>8s} {'A TF/s':>8s} "
          f"{'Speedup':>8s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name in all_ops:
        b = b_agg.get(name, {"self_time": 0.0, "flops": 0})
        a = a_agg.get(name, {"self_time": 0.0, "flops": 0})
        b_pct_t = 100.0 * b["self_time"] / b_total_time
        a_pct_t = 100.0 * a["self_time"] / a_total_time
        b_gf = b["flops"] / 1e9
        a_gf = a["flops"] / 1e9
        b_pct_f = 100.0 * b["flops"] / b_total_flops
        a_pct_f = 100.0 * a["flops"] / a_total_flops
        b_tput = _tput(b["flops"], b["self_time"])
        a_tput = _tput(a["flops"], a["self_time"])
        # Speedup = before_time / after_time for this op (— if op
        # disappeared, (new) if it newly appeared post-prune).
        if a["self_time"] > 0 and b["self_time"] > 0:
            speedup = f"{b['self_time'] / a['self_time']:.2f}x"
        elif b["self_time"] > 0 and a["self_time"] == 0:
            speedup = "→sparse"
        elif a["self_time"] > 0 and b["self_time"] == 0:
            speedup = "(new)"
        else:
            speedup = "—"
        print(f"  {name[:30]:<30s} "
              f"{b_pct_t:>7.2f}% {a_pct_t:>7.2f}% {a_pct_t - b_pct_t:>+7.2f}% "
              f"{b_gf:>8.3f} {b_pct_f:>7.2f}% {b_tput:>8.2f} "
              f"{a_gf:>8.3f} {a_pct_f:>7.2f}% {a_tput:>8.2f} "
              f"{speedup:>8s}")
    print("=" * width)
