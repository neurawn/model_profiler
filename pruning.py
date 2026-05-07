"""
Pruning Support Utilities
=========================
Shared helpers for the SparseGPT pruning flow (`sparsegpt.py`):

  - PruneResult dataclass + save_pruned_model wrapper-dict format
  - HuggingFace Conv1D detection and Conv1D → nn.Linear conversion
    (cuSPARSELt's 2:4 path only accelerates `sparse @ dense`, so the
    sparse weight has to live on the left of the matmul — nn.Linear's
    `x @ Wᵀ` does that, HF Conv1D's `x @ W` does not)
  - SparseSemiStructuredTensor wrapping for cuSPARSELt dispatch on Ampere+

This module does NOT do the actual pruning — see `sparsegpt.py` for the
Hessian-based one-shot pruner that produces the masks.
"""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class PruneResult:
    """Summary of a pruning pass (filled by SparseGPT, consumed by save)."""
    pruned_layers: List[str] = field(default_factory=list)
    converted_sparse: List[str] = field(default_factory=list)
    skipped_layers: List[Tuple[str, str]] = field(default_factory=list)
    total_zeros: int = 0
    total_weights: int = 0

    @property
    def sparsity(self) -> float:
        return self.total_zeros / max(self.total_weights, 1)


# ── HuggingFace Conv1D helpers ──────────────────────────────────────────────

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


# ── Save / reload helpers ───────────────────────────────────────────────────

def _materialize_sparse_state_dict(model: nn.Module):
    """Return (state_dict, n_materialized) where SparseSemiStructuredTensor
    weights are converted back to dense via `to_dense()`.

    Saving the sparse subclass directly is fragile — pickle/load behavior
    varies across torch versions. The 2:4 pattern is encoded as zeros in the
    dense tensor, so a `to_dense()` round-trip preserves the sparsity and
    lets the .pt load anywhere.
    """
    out = {}
    n_materialized = 0
    for k, v in model.state_dict().items():
        is_sst = type(v).__name__.startswith("SparseSemiStructured")
        if is_sst and hasattr(v, "to_dense"):
            try:
                out[k] = v.to_dense().detach().cpu().contiguous()
                n_materialized += 1
                continue
            except Exception as e:
                print(f"  [save_pruned] to_dense failed for {k}: {e}; "
                      f"falling back to detach")
        out[k] = v.detach().cpu() if hasattr(v, "detach") else v
    return out, n_materialized


def save_pruned_model(
    model: nn.Module,
    model_name: str,
    prune_result: PruneResult,
    save_path: Optional[str] = None,
    output_dir: str = "./profiling_results",
    sparse_dtype: Optional[torch.dtype] = None,
) -> str:
    """Save a pruned model (SparseGPT 2:4 or unstructured) to disk.

    Materializes any SparseSemiStructuredTensor weights to dense (zeros
    preserved) so the file loads with plain `torch.load` +
    `model.load_state_dict` on any torch version. Wrapper dict format
    matches `apply_quant_plan`'s output.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not save_path:
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        save_path = os.path.join(output_dir, f"{safe_name}_pruned.pt")

    state_dict, n_materialized = _materialize_sparse_state_dict(model)

    # Detect on-disk layout. When all HF Conv1D modules have been replaced
    # with nn.Linear (the cuSPARSELt path), saved weights are transposed
    # relative to native HF and the reloader must transpose them back.
    # Pure-Conv1D models (e.g. SparseGPT unstructured, no sparse conversion)
    # save in native layout — no transpose needed.
    Conv1D = _hf_conv1d_cls()
    has_conv1d = (Conv1D is not None
                  and any(isinstance(m, Conv1D) for m in model.modules()))
    weights_in_linear_layout = not has_conv1d

    payload = {
        "model_state_dict": state_dict,
        "pruned_layers": list(prune_result.pruned_layers),
        "converted_sparse": list(prune_result.converted_sparse),
        "skipped_layers": list(prune_result.skipped_layers),
        "total_zeros": prune_result.total_zeros,
        "total_weights": prune_result.total_weights,
        "achieved_sparsity": prune_result.sparsity,
        "sparse_dtype": str(sparse_dtype) if sparse_dtype is not None else None,
        # Reload-side flag: True ⇒ weights are in nn.Linear (out, in) layout
        # and need transposition back to HF Conv1D (in, out) on load. False
        # ⇒ already in native Conv1D layout, no transpose needed. Existing
        # checkpoints predate this field; the loader defaults to True for
        # backcompat (which matches the only previously-emitted format).
        "weights_in_linear_layout": weights_in_linear_layout,
        "format": (
            f"Pruned dense weights (zeros preserved). "
            f"weights_in_linear_layout={weights_in_linear_layout}."
        ),
    }
    torch.save(payload, save_path)
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  Saved pruned model: {save_path} ({file_size:.1f} MB)")
    print(f"    Layers pruned:   {len(prune_result.pruned_layers)}")
    print(f"    Sparsity:        {100.0 * prune_result.sparsity:.2f}%")
    if n_materialized:
        print(f"    Materialized:    {n_materialized} sparse-tensor weight(s) "
              f"to dense (2:4 zeros preserved)")
    print(f"    Reload:          --local {save_path} --arch <gpt2|...>")
    return save_path
