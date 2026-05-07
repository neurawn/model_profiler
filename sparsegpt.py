"""
SparseGPT One-Shot Pruning
==========================
Hessian-based post-training pruning (Frantar & Alistarh, 2023). For each
prunable Linear/Conv1D, capture activations on a calibration set to build
H = (2/N) Σ X Xᵀ, then run an iterative column-by-column update that picks
the lowest-importance entries (|w|² / [H⁻¹]ᵢᵢ²) and compensates the
remaining weights via H⁻¹ to minimize ‖W X − W' X‖².

Compared to magnitude pruning (`pruning.py`), SparseGPT preserves accuracy
much better at the same sparsity — a 50%-sparse GPT-2 typically loses only
a couple of perplexity points without fine-tuning.

Two patterns are supported via `pattern`:
  - "2:4"          fixed 50% sparsity in 2-of-4 groups (Ampere+ tensor cores)
  - "unstructured" arbitrary positions, target sparsity given by `sparsity`

Limitations:
  - Sequential driver assumes a HuggingFace GPT-2 layout
    (model.transformer.h is a list of blocks). Other architectures could be
    added by wiring a similar block-walk.
  - Calibration data is sampled from WikiText-2 train. C4 (used in the paper)
    would be more general but is much heavier to download.
"""

import os
import math
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Sequence


def _hf_conv1d_cls():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:
        return None


@dataclass
class SparseGPTResult:
    """Summary of a SparseGPT pruning pass."""
    pattern: str
    sparsity_target: float
    pruned_layers: List[Tuple[str, float]] = field(default_factory=list)
    skipped_layers: List[Tuple[str, str]] = field(default_factory=list)
    converted_sparse: List[str] = field(default_factory=list)
    layer_losses: List[float] = field(default_factory=list)
    total_zeros: int = 0
    total_weights: int = 0
    elapsed_seconds: float = 0.0
    n_calibration_samples: int = 0
    calibration_seqlen: int = 0

    @property
    def achieved_sparsity(self) -> float:
        return self.total_zeros / max(self.total_weights, 1)

    @property
    def mean_layer_loss(self) -> float:
        if not self.layer_losses:
            return 0.0
        return sum(self.layer_losses) / len(self.layer_losses)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"  SPARSEGPT SUMMARY  pattern={self.pattern}  "
            f"target={self.sparsity_target:.2f}",
            "=" * 72,
            f"  Calibration:       {self.n_calibration_samples} samples × "
            f"{self.calibration_seqlen} tokens",
            f"  Pruned layers:     {len(self.pruned_layers)}",
            f"  Sparse-converted:  {len(self.converted_sparse)}",
            f"  Skipped:           {len(self.skipped_layers)}",
            f"  Total weights:     {self.total_weights:,}",
            f"  Achieved sparsity: {self.achieved_sparsity * 100:.2f}%",
            f"  Mean layer loss:   {self.mean_layer_loss:.4f}",
            f"  Elapsed:           {self.elapsed_seconds:.1f}s",
        ]
        if self.skipped_layers:
            lines.append("  Skipped (first 10):")
            for name, reason in self.skipped_layers[:10]:
                lines.append(f"    {name}: {reason}")
        return "\n".join(lines)


# ── Per-layer Hessian collector ─────────────────────────────────────────────

class _LayerHessian:
    """Streaming H = (2/N) Σ X_i X_iᵀ for a single Linear/Conv1D layer.

    The 2/N scaling matches the SparseGPT reference impl — comes from the
    Hessian of the squared-error objective ‖W X − W' X‖²/2.
    """

    def __init__(self, layer: nn.Module):
        Conv1D = _hf_conv1d_cls()
        if isinstance(layer, nn.Linear):
            self.in_features = layer.weight.shape[1]
        elif Conv1D is not None and isinstance(layer, Conv1D):
            self.in_features = layer.weight.shape[0]
        else:
            raise TypeError(f"unsupported layer: {type(layer).__name__}")
        self.dev = layer.weight.device
        # fp32 for numerical stability of the Cholesky chain.
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32,
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        # X is (D_in, N) after transpose; H grows by X X^T.
        x = inp.t().float()
        n_new = x.shape[1]
        # Running mean: scale old H down before adding new contribution.
        self.H *= self.nsamples / (self.nsamples + n_new)
        self.nsamples += n_new
        x = math.sqrt(2.0 / self.nsamples) * x
        self.H += x @ x.t()


# ── Core algorithm ──────────────────────────────────────────────────────────

def _sparsegpt_prune_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    pattern: str,
    sparsity: float = 0.5,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """Run the SparseGPT iterative column update on a single weight matrix.

    Args:
        W: (out, in) float32 weight to prune (Linear layout — Conv1D callers
           must transpose first and back).
        H: (in, in) float32 Hessian, already accumulated.
        pattern: "2:4" or "unstructured".
        sparsity: target sparsity (used only for unstructured; 2:4 is exact).
        blocksize: column block size for the iterative algorithm.
        percdamp: diagonal damping fraction (ridge added to H for stability).

    Returns: (W_pruned, layer_loss). layer_loss is Σ (Δw)² / d², a per-layer
    proxy for accumulated reconstruction error (smaller ⇒ better).
    """
    rows, columns = W.shape
    W = W.clone()

    # Rows with zero variance ⇒ skip (would blow up Cholesky).
    dead = torch.diag(H) == 0
    if dead.any():
        H = H.clone()
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Damping λ = percdamp · mean(diag H) — standard SparseGPT recipe.
    damp = percdamp * torch.mean(torch.diag(H)).item()
    diag_idx = torch.arange(columns, device=H.device)
    H = H.clone()
    H[diag_idx, diag_idx] += damp

    # Cholesky chain: factor H, invert via cholesky_inverse, then upper-Cholesky
    # of the inverse — the upper triangle is what the column updates need.
    L = torch.linalg.cholesky(H)
    H_inv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(H_inv_full, upper=True)

    losses = torch.zeros(rows, device=W.device)

    if pattern == "2:4":
        prunen, prunem = 2, 4
    elif pattern == "unstructured":
        prunen, prunem = 0, 0
    else:
        raise ValueError(f"unknown pattern: {pattern!r}")

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prunen == 0:
            # Unstructured: pick a per-block threshold over importance scores
            # |w|² / [H⁻¹]ᵢᵢ² so we prune the lowest-`sparsity` fraction.
            score = W1 ** 2 / (torch.diag(Hinv1).reshape(1, -1)) ** 2
            n_zero = int(score.numel() * sparsity)
            if n_zero >= score.numel():
                mask1 = torch.ones_like(W1, dtype=torch.bool)
            elif n_zero <= 0:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)
            else:
                thresh = torch.kthvalue(score.flatten(), n_zero).values
                mask1 = score <= thresh
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if prunen != 0 and i % prunem == 0:
                # 2:4 mask: in each row, mask the 2 lowest-score entries
                # within the upcoming 4-column window.
                window_score = (
                    W1[:, i:(i + prunem)] ** 2
                    / (torch.diag(Hinv1)[i:(i + prunem)].reshape(1, -1)) ** 2
                )
                _, lowest_idx = torch.topk(
                    window_score, prunen, dim=1, largest=False)
                mask1[:, i:(i + prunem)] = False
                mask1.scatter_(1, i + lowest_idx, True)

            q = w.clone()
            q[mask1[:, i]] = 0.0
            Q1[:, i] = q
            losses += (w - q) ** 2 / d ** 2

            # Compensation: distribute the pruned-column residual across the
            # remaining columns of this block via the upper-Cholesky inverse.
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        # Block-level error propagation to all later columns.
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W, float(losses.sum().item() / 2.0)


# ── Calibration data loader ─────────────────────────────────────────────────

def load_wikitext_calibration(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 1024,
    seed: int = 0,
    split: str = "train",
) -> torch.Tensor:
    """Sample `n_samples` random sequences of `seqlen` tokens from WikiText-2.

    Returns a (n_samples, seqlen) long tensor. Cached implicitly through the
    HF datasets cache.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if enc.numel() <= seqlen + 1:
        raise ValueError(
            f"WikiText-2 {split} has only {enc.numel()} tokens — too short "
            f"for seqlen={seqlen}.")

    g = torch.Generator()
    g.manual_seed(seed)
    samples = []
    for _ in range(n_samples):
        i = int(torch.randint(
            0, enc.numel() - seqlen - 1, (1,), generator=g).item())
        samples.append(enc[i:i + seqlen])
    return torch.stack(samples)


# ── GPT-2 sequential driver ─────────────────────────────────────────────────

def prune_gpt2_sparsegpt(
    model: nn.Module,
    calibration_token_ids: torch.Tensor,
    pattern: str = "2:4",
    sparsity: float = 0.5,
    blocksize: int = 128,
    percdamp: float = 0.01,
    skip_substrings: Sequence[str] = ("lm_head", "wte", "wpe"),
    device: str = "cuda",
    convert_to_sparse: bool = False,
    sparse_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> SparseGPTResult:
    """Apply SparseGPT to a HuggingFace GPT-2 model, block by block.

    Each block's prunable layers see calibration activations *as produced by
    the previously-pruned blocks* — this matters because errors compound, so
    the blocks downstream prune against the realistic input distribution.

    For pattern="2:4" with convert_to_sparse=True on CUDA, the model is
    additionally cast to fp16, Conv1D modules replaced with nn.Linear, and
    weights wrapped as SparseSemiStructuredTensor for cuSPARSELt dispatch.

    Returns a SparseGPTResult.
    """
    Conv1D = _hf_conv1d_cls()
    target_types: tuple = (nn.Linear,)
    if Conv1D is not None:
        target_types = target_types + (Conv1D,)

    if not (hasattr(model, "transformer") and hasattr(model.transformer, "h")):
        raise RuntimeError(
            "prune_gpt2_sparsegpt: model does not look like a HuggingFace "
            "GPT-2 (missing transformer.h). Add a per-arch driver to support "
            "other models.")

    result = SparseGPTResult(
        pattern=pattern,
        sparsity_target=(0.5 if pattern == "2:4" else sparsity),
        n_calibration_samples=calibration_token_ids.shape[0],
        calibration_seqlen=calibration_token_ids.shape[1],
    )
    t0 = time.time()

    model.eval()
    model.to(device)

    blocks = model.transformer.h
    n_samples, seqlen = calibration_token_ids.shape
    n_embd = model.config.n_embd

    # ── Capture block-0 input by hijacking blocks[0] ─────────────────────
    if verbose:
        print(f"  [sparsegpt] Capturing block-0 inputs: "
              f"{n_samples}×{seqlen} tokens, n_embd={n_embd}...")

    inps = torch.zeros(
        (n_samples, seqlen, n_embd), dtype=torch.float32, device=device)
    cache: dict = {"i": 0, "args": None, "kwargs": None}

    _DROP_KEYS = ("layer_past", "past_key_value", "past_key_values")

    class _Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args, **kwargs):
            # First positional is hidden_states; capture, save the rest of the
            # args/kwargs so we can replay the block ourselves with matching
            # auxiliary state (attention_mask, head_mask, …).
            hidden_states = args[0]
            inps[cache["i"]] = hidden_states.detach().float()
            cache["i"] += 1
            cache["args"] = args[1:]
            cache["kwargs"] = {
                k: v for k, v in kwargs.items() if k not in _DROP_KEYS
            }
            raise ValueError("captured")

    blocks[0] = _Catcher(blocks[0])
    for i in range(n_samples):
        try:
            with torch.no_grad():
                model(calibration_token_ids[i:i + 1].to(device))
        except ValueError:
            pass
    blocks[0] = blocks[0].m
    block_args = cache["args"] or ()
    block_kwargs = cache["kwargs"] or {}

    outs = torch.zeros_like(inps)

    # ── Sequential block sweep ──────────────────────────────────────────
    for block_idx, block in enumerate(blocks):
        if verbose:
            print(f"  [sparsegpt] Block {block_idx + 1}/{len(blocks)}...")

        # Find prunable submodules in this block.
        block_layers: List[Tuple[str, str, nn.Module]] = []
        for name, mod in block.named_modules():
            if not isinstance(mod, target_types):
                continue
            full_name = f"transformer.h.{block_idx}.{name}"
            if any(s in full_name for s in skip_substrings):
                continue
            block_layers.append((name, full_name, mod))

        if not block_layers:
            for i in range(n_samples):
                with torch.no_grad():
                    outs[i] = block(
                        inps[i:i + 1].to(model.dtype),
                        *block_args, **block_kwargs)[0]
            inps, outs = outs, inps
            continue

        stats = {short: _LayerHessian(mod) for short, _, mod in block_layers}
        handles = []

        def make_hook(short):
            def hook(_module, inputs, _output):
                stats[short].add_batch(inputs[0].detach())
            return hook

        for short, _, mod in block_layers:
            handles.append(mod.register_forward_hook(make_hook(short)))

        for i in range(n_samples):
            with torch.no_grad():
                outs[i] = block(
                    inps[i:i + 1].to(model.dtype), **block_kwargs)[0]

        for h in handles:
            h.remove()

        # Prune each layer using its accumulated Hessian.
        for short, full_name, mod in block_layers:
            H = stats[short].H
            is_conv1d = Conv1D is not None and isinstance(mod, Conv1D)

            if is_conv1d:
                W = mod.weight.data.t().contiguous().float()
            else:
                W = mod.weight.data.float()

            in_size = W.shape[1]
            if pattern == "2:4" and in_size % 4 != 0:
                result.skipped_layers.append(
                    (full_name, f"in={in_size} not divisible by 4"))
                continue

            W_pruned, layer_loss = _sparsegpt_prune_weight(
                W, H, pattern=pattern, sparsity=sparsity,
                blocksize=blocksize, percdamp=percdamp,
            )

            if is_conv1d:
                W_pruned = W_pruned.t().contiguous()
            mod.weight.data = W_pruned.to(dtype=mod.weight.dtype)

            zeros = int((W_pruned == 0).sum().item())
            ach = zeros / W_pruned.numel()
            result.pruned_layers.append((full_name, ach))
            result.layer_losses.append(layer_loss)
            result.total_zeros += zeros
            result.total_weights += W_pruned.numel()

            if verbose:
                print(f"    {full_name}: sparsity={ach * 100:5.2f}%, "
                      f"loss={layer_loss:.4f}")

        # Re-run the block with the pruned weights so the next block sees
        # the realistic activations it'll receive at inference time.
        for i in range(n_samples):
            with torch.no_grad():
                outs[i] = block(
                    inps[i:i + 1].to(model.dtype), **block_kwargs)[0]
        inps, outs = outs, inps

    result.elapsed_seconds = time.time() - t0

    # ── Optional: convert to SparseSemiStructuredTensor for cuSPARSELt ──
    if convert_to_sparse and pattern == "2:4":
        if device != "cuda":
            print("  [sparsegpt] convert_to_sparse=True needs CUDA; skipping.")
        else:
            from pruning import (
                _convert_hf_conv1d_to_linear, _to_sparse_semi_structured,
            )
            if verbose:
                print("  [sparsegpt] Casting model to fp16, converting "
                      "Conv1D → Linear, wrapping weights as "
                      "SparseSemiStructuredTensor...")
            model.to(dtype=sparse_dtype)
            n_conv = _convert_hf_conv1d_to_linear(model)
            if verbose and n_conv:
                print(f"    Converted {n_conv} HF Conv1D → nn.Linear")
            for name, mod in model.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                if any(s in name for s in skip_substrings):
                    continue
                k_size = mod.weight.shape[1]
                min_k = 32 if sparse_dtype == torch.int8 else 16
                if k_size % min_k != 0:
                    continue
                ok, reason = _to_sparse_semi_structured(
                    mod, dtype=sparse_dtype)
                if ok:
                    result.converted_sparse.append(name)

    if verbose:
        print(result.summary())

    return result


# ── Save helper (wraps pruning.save_pruned_model) ───────────────────────────

def save_sparsegpt_model(
    model: nn.Module,
    model_name: str,
    result: SparseGPTResult,
    save_path: Optional[str] = None,
    output_dir: str = "./profiling_results",
    sparse_dtype: Optional[torch.dtype] = None,
) -> str:
    """Persist a SparseGPT-pruned model. Reuses pruning.save_pruned_model so
    the on-disk layout matches the magnitude-pruned format (the perplexity
    `--local` loader already understands it).
    """
    from pruning import save_pruned_model, PruneResult

    pat_safe = result.pattern.replace(":", "_")
    if save_path is None:
        os.makedirs(output_dir, exist_ok=True)
        safe = model_name.lower().replace(" ", "_").replace("-", "_")
        save_path = os.path.join(
            output_dir, f"{safe}_sparsegpt_{pat_safe}.pt")

    compat = PruneResult(
        pruned_layers=[name for name, _ in result.pruned_layers],
        skipped_layers=list(result.skipped_layers),
        converted_sparse=list(result.converted_sparse),
        total_zeros=result.total_zeros,
        total_weights=result.total_weights,
    )
    return save_pruned_model(
        model, model_name, compat,
        save_path=save_path, output_dir=output_dir,
        sparse_dtype=sparse_dtype,
    )
