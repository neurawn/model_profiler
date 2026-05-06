"""
WikiText-2 Perplexity Evaluation for Causal LMs
================================================
Sliding-window perplexity on the WikiText-2 raw test split, the standard
benchmark for GPT-2-style models. Used to compare the accuracy impact of
compression methods (plan-based mixed-precision quantization, 2:4 magnitude
pruning) against the FP32 baseline.

Reference: https://huggingface.co/docs/transformers/perplexity — but with
exact token-weighted NLL accumulation (the doc's `torch.stack(nlls).mean()`
slightly over-weights short final windows).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn


@dataclass
class PerplexityResult:
    label: str
    perplexity: float
    avg_nll: float
    eval_tokens: int
    eval_seconds: float
    param_count: int
    size_mb: float
    notes: str = ""


_WIKITEXT_CACHE = {"text": None}


def _load_wikitext2_text() -> str:
    """Load WikiText-2 raw test split and return concatenated text. Cached
    in-process so repeat calls in the same run don't re-download.
    """
    if _WIKITEXT_CACHE["text"] is not None:
        return _WIKITEXT_CACHE["text"]
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "WikiText perplexity requires `datasets`. "
            "Install with: pip install datasets"
        ) from e
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # HF convention: join with \n\n (the dataset's lines preserve paragraph
    # breaks but strip the blank-line separators).
    text = "\n\n".join(ds["text"])
    _WIKITEXT_CACHE["text"] = text
    return text


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    label: str = "model",
    device: Optional[str] = None,
    max_length: int = 1024,
    stride: int = 512,
    max_tokens: Optional[int] = None,
    notes: str = "",
    verbose: bool = True,
) -> PerplexityResult:
    """Compute WikiText-2 perplexity with a sliding window.

    Each window of `max_length` tokens advances by `stride`. Tokens already
    counted in the previous window are masked with -100 so they don't
    contribute to the loss twice. NLL is accumulated token-weighted (mean
    of per-token NLL over the entire eval set), then exponentiated.

    Args:
        model: causal LM that returns ModelOutput with `.loss` when `labels=` is passed
            (HuggingFace GPT-2 et al.).
        tokenizer: HuggingFace tokenizer matching the model.
        label: human-readable name for the result row.
        device: 'cpu' or 'cuda'. Defaults to cuda when available.
        max_length: context length per window. GPT-2's max is 1024.
        stride: window step. Smaller stride → more accurate but slower.
            Standard is max_length // 2.
        max_tokens: optional cap on total tokens evaluated (for fast smoke
            testing). None = full test split.
        notes: free-text annotation (e.g. "INT4/INT8 mix, 4.5 avg bits").
        verbose: print progress every ~10 windows.

    Returns: PerplexityResult.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text = _load_wikitext2_text()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_full = encodings.input_ids
    seq_len = input_ids_full.size(1)
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)
        input_ids_full = input_ids_full[:, :seq_len]

    if verbose:
        print(f"  [perplexity:{label}] eval tokens={seq_len:,}, "
              f"window={max_length}, stride={stride}, device={device}")

    model.eval()
    # Don't move sparse models — SparseSemiStructuredTensor is already on cuda.
    try:
        model.to(device)
    except Exception as e:
        if verbose:
            print(f"  [perplexity:{label}] model.to({device}) skipped: {e}")

    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0
    n_windows = 0
    start = time.time()

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = input_ids_full[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Mask the prefix already evaluated in the prior window.
        if trg_len < (end_loc - begin_loc):
            target_ids[:, :-(trg_len)] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # HF causal LMs compute mean cross-entropy over shift_labels[1:]
            # where labels==-100 are ignored. Recover the exact token count
            # so accumulation is correctly token-weighted.
            shifted = target_ids[:, 1:]
            num_loss_tokens = int((shifted != -100).sum().item())
            if num_loss_tokens > 0 and torch.isfinite(outputs.loss):
                total_nll += outputs.loss.item() * num_loss_tokens
                total_tokens += num_loss_tokens

        prev_end_loc = end_loc
        n_windows += 1
        if verbose and n_windows % 10 == 0:
            cur_ppl = math.exp(total_nll / max(total_tokens, 1))
            print(f"    window {n_windows}: pos={end_loc:,}/{seq_len:,}, "
                  f"running ppl={cur_ppl:.3f}")
        if end_loc == seq_len:
            break

    elapsed = time.time() - start
    if total_tokens == 0:
        avg_nll = float("nan")
        ppl = float("nan")
    else:
        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)

    param_count = sum(p.numel() for p in model.parameters())
    # element_size for SparseSemiStructuredTensor reports the dense dtype's
    # size; the actual storage is ~half, but we want a comparable "weight
    # bytes" number for the table, so report the dense-equivalent bytes.
    size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    size_mb = size_bytes / (1024 * 1024)

    if verbose:
        print(f"  [perplexity:{label}] ppl={ppl:.3f}  "
              f"avg_nll={avg_nll:.4f}  tokens={total_tokens:,}  "
              f"elapsed={elapsed:.1f}s")

    return PerplexityResult(
        label=label, perplexity=ppl, avg_nll=avg_nll,
        eval_tokens=total_tokens, eval_seconds=elapsed,
        param_count=param_count, size_mb=size_mb, notes=notes,
    )


def print_perplexity_table(rows: List[PerplexityResult]) -> None:
    """Render a comparison table. First row is treated as the baseline for
    the Δ columns (absolute and percent perplexity change)."""
    if not rows:
        print("  (no perplexity rows to display)")
        return

    baseline = rows[0]
    width = 124
    print()
    print("=" * width)
    print(f"  WIKITEXT-2 PERPLEXITY COMPARISON  (baseline: {baseline.label})")
    print("=" * width)
    print(f"  {'Model':<28s} {'Params':>14s} {'Size MB':>10s} "
          f"{'PPL':>10s} {'ΔPPL':>10s} {'Δ%':>8s} "
          f"{'Tokens':>10s} {'Time s':>8s}  Notes")
    print(f"  {'─'*28} {'─'*14} {'─'*10} "
          f"{'─'*10} {'─'*10} {'─'*8} "
          f"{'─'*10} {'─'*8}  {'─'*30}")
    for r in rows:
        delta = r.perplexity - baseline.perplexity
        pct = (100.0 * delta / baseline.perplexity
               if baseline.perplexity > 0 else 0.0)
        delta_str = (f"{delta:+.3f}" if r is not baseline else "—")
        pct_str = (f"{pct:+.2f}%" if r is not baseline else "—")
        print(f"  {r.label[:28]:<28s} {r.param_count:>14,} "
              f"{r.size_mb:>10.1f} {r.perplexity:>10.3f} "
              f"{delta_str:>10s} {pct_str:>8s} "
              f"{r.eval_tokens:>10,} {r.eval_seconds:>8.1f}  {r.notes}")
    print("=" * width)
    print(f"  Lower PPL = better. ΔPPL is absolute change vs baseline; "
          f"Δ% is relative.")
    print("=" * width)
