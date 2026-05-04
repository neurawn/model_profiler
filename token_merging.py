"""
Token Merging Module
====================
Implements token merging as a compression/acceleration technique for
Vision Transformers and Transformer models. Token merging reduces the
number of tokens processed by later layers, cutting FLOPs and latency
without retraining.

Three strategies implemented:
1. Bipartite Soft Matching (ToMe) — Bolya et al., 2023
   Partition tokens into two sets, compute pairwise cosine similarity,
   greedily match top-r pairs, merge via averaging.

2. K-Means Clustering
   Cluster tokens into k groups, replace each cluster with its centroid.
   Simpler but requires choosing k and has O(n*k*d) cost per iteration.

3. Average Pooling Merging
   Merge adjacent (or strided) groups of tokens by averaging.
   Fastest method but ignores semantic similarity.

All strategies can be applied:
- As a standalone wrapper around any ViT/Transformer model
- Per-layer with configurable merge ratios
- With optional class-token protection (never merge the [CLS] token)

References:
  Bolya, D., Fu, C., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J.
  "Token Merging: Your ViT But Faster." ICLR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable, Literal
from enum import Enum
import time
import warnings


# ════════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════════

class MergeStrategy(Enum):
    BIPARTITE = "bipartite"
    KMEANS = "kmeans"
    AVERAGE_POOL = "average_pool"


@dataclass
class MergeResult:
    """Result of a token merging operation."""
    merged_tokens: torch.Tensor       # (B, T', D) — reduced token sequence
    original_count: int               # original number of tokens T
    merged_count: int                 # merged number of tokens T'
    reduction_ratio: float            # T' / T (lower = more compression)
    merge_map: Optional[torch.Tensor] = None  # (B, T) → assignment to merged token
    strategy: str = ""
    time_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"  Strategy:   {self.strategy}\n"
            f"  Tokens:     {self.original_count} → {self.merged_count} "
            f"({1 - self.reduction_ratio:.1%} reduction)\n"
            f"  Time:       {self.time_ms:.3f} ms"
        )


@dataclass
class TokenMergingProfile:
    """Profile comparing all strategies on a given input."""
    model_name: str
    input_shape: Tuple[int, ...]
    results: Dict[str, MergeResult] = field(default_factory=dict)
    best_strategy: str = ""
    best_reduction: float = 1.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  TOKEN MERGING PROFILE: {self.model_name}",
            f"{'='*60}",
            f"  Input shape: {self.input_shape}",
            f"{'─'*60}",
        ]
        for name, result in self.results.items():
            lines.append(f"\n  [{name.upper()}]")
            lines.append(result.summary())

        lines += [
            f"{'─'*60}",
            f"  Best strategy: {self.best_strategy} "
            f"(reduction={1 - self.best_reduction:.1%})",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Strategy 1: Bipartite Soft Matching (ToMe)
# ════════════════════════════════════════════════════════════════

class BipartiteSoftMatching:
    """
    Token Merging via Bipartite Soft Matching (Bolya et al., 2023).

    Algorithm:
    1. Split tokens into two disjoint sets A and B (alternating indices)
    2. Compute cosine similarity between all pairs (A_i, B_j)
    3. For each token in A, find its most similar token in B
    4. Select the top-r most similar pairs
    5. Merge each selected pair by averaging their features
    6. Return unmerged A tokens + unmerged B tokens + merged tokens

    This preserves token diversity while merging the most redundant pairs.
    """

    def __init__(self, r: Optional[int] = None, ratio: float = 0.5,
                 protect_cls: bool = True):
        """
        Args:
            r: Number of pairs to merge. If None, computed from ratio.
            ratio: Fraction of tokens to keep (0.5 = merge half). Used if r is None.
            protect_cls: If True, never merge the first token ([CLS]).
        """
        self.r = r
        self.ratio = ratio
        self.protect_cls = protect_cls

    def __call__(self, tokens: torch.Tensor) -> MergeResult:
        """
        Merge tokens using bipartite soft matching.

        Args:
            tokens: (B, T, D) — batch of token sequences

        Returns:
            MergeResult with merged tokens
        """
        B, T, D = tokens.shape
        start = time.perf_counter()

        # Determine how many pairs to merge
        if self.r is not None:
            r = min(self.r, T // 2)
        else:
            r = max(1, int(T * (1 - self.ratio) / 2))

        # Protect CLS token by separating it
        if self.protect_cls and T > 2:
            cls_token = tokens[:, :1, :]    # (B, 1, D)
            work_tokens = tokens[:, 1:, :]  # (B, T-1, D)
            T_work = T - 1
        else:
            cls_token = None
            work_tokens = tokens
            T_work = T

        if T_work < 2 or r < 1:
            return MergeResult(
                merged_tokens=tokens, original_count=T,
                merged_count=T, reduction_ratio=1.0,
                strategy="bipartite", time_ms=0
            )

        # Step 1: Split into sets A (even indices) and B (odd indices)
        a_idx = torch.arange(0, T_work, 2, device=tokens.device)
        b_idx = torch.arange(1, T_work, 2, device=tokens.device)

        a_tokens = work_tokens[:, a_idx]  # (B, |A|, D)
        b_tokens = work_tokens[:, b_idx]  # (B, |B|, D)

        # Step 2: Cosine similarity between A and B
        a_norm = F.normalize(a_tokens, dim=-1)  # (B, |A|, D)
        b_norm = F.normalize(b_tokens, dim=-1)  # (B, |B|, D)
        sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, |A|, |B|)

        # Step 3: For each A token, find best matching B token
        max_sim, max_idx = sim.max(dim=-1)  # (B, |A|)

        # Step 4: Select top-r most similar pairs
        r = min(r, a_tokens.shape[1], b_tokens.shape[1])
        _, top_a = max_sim.topk(r, dim=-1)  # (B, r) — indices into A

        # Gather the corresponding B indices
        top_b = torch.gather(max_idx, 1, top_a)  # (B, r) — indices into B

        # Step 5: Merge selected pairs by averaging
        # We need to ensure each B token is used at most once to get
        # consistent output sizes across the batch. Use greedy 1-to-1
        # matching: iterate A tokens in order of descending similarity,
        # skip if the B target is already taken.
        merged = []
        for b_i in range(B):
            # Greedy 1-to-1 assignment
            sorted_a_indices = max_sim[b_i].argsort(descending=True)
            used_b = set()
            pairs_a = []
            pairs_b = []
            for a_i in sorted_a_indices.tolist():
                b_j = max_idx[b_i, a_i].item()
                if b_j not in used_b and len(pairs_a) < r:
                    pairs_a.append(a_i)
                    pairs_b.append(b_j)
                    used_b.add(b_j)
                if len(pairs_a) >= r:
                    break

            actual_r = len(pairs_a)
            pairs_a_t = torch.tensor(pairs_a, device=tokens.device)
            pairs_b_t = torch.tensor(pairs_b, device=tokens.device)

            a_sel = a_tokens[b_i, pairs_a_t]  # (actual_r, D)
            b_sel = b_tokens[b_i, pairs_b_t]  # (actual_r, D)
            merged_pairs = (a_sel + b_sel) / 2.0

            # Unmerged tokens
            a_mask = torch.ones(a_tokens.shape[1], dtype=torch.bool, device=tokens.device)
            a_mask[pairs_a_t] = False
            b_mask = torch.ones(b_tokens.shape[1], dtype=torch.bool, device=tokens.device)
            b_mask[pairs_b_t] = False

            unmerged = torch.cat([
                a_tokens[b_i, a_mask],
                b_tokens[b_i, b_mask],
                merged_pairs,
            ], dim=0)

            merged.append(unmerged)

        # Stack batch — with 1-to-1 matching, all items have size T_work - actual_r
        merged_tokens = torch.stack(merged, dim=0)

        # Prepend CLS token if we separated it
        if cls_token is not None:
            merged_tokens = torch.cat([cls_token, merged_tokens], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        new_T = merged_tokens.shape[1]

        return MergeResult(
            merged_tokens=merged_tokens,
            original_count=T,
            merged_count=new_T,
            reduction_ratio=new_T / T,
            strategy="bipartite",
            time_ms=elapsed,
        )


# ════════════════════════════════════════════════════════════════
# Strategy 2: K-Means Clustering
# ════════════════════════════════════════════════════════════════

class KMeansMerging:
    """
    Merge tokens by clustering them with K-Means and replacing
    each cluster with its centroid.

    Pros: Semantically aware grouping, handles arbitrary reduction ratios.
    Cons: Slower due to iterative clustering, sensitive to initialization.
    """

    def __init__(self, k: Optional[int] = None, ratio: float = 0.5,
                 max_iters: int = 10, protect_cls: bool = True):
        """
        Args:
            k: Number of clusters (= number of output tokens). If None, computed from ratio.
            ratio: Fraction of tokens to keep. Used if k is None.
            max_iters: Maximum K-Means iterations.
            protect_cls: If True, the first token is kept as-is and not clustered.
        """
        self.k = k
        self.ratio = ratio
        self.max_iters = max_iters
        self.protect_cls = protect_cls

    def __call__(self, tokens: torch.Tensor) -> MergeResult:
        """
        Merge tokens using K-Means clustering.

        Args:
            tokens: (B, T, D)

        Returns:
            MergeResult
        """
        B, T, D = tokens.shape
        start = time.perf_counter()

        # Determine k
        if self.k is not None:
            k = min(self.k, T)
        else:
            k = max(1, int(T * self.ratio))

        # Separate CLS
        if self.protect_cls and T > 2:
            cls_token = tokens[:, :1, :]
            work_tokens = tokens[:, 1:, :]
            k_work = k - 1  # one slot reserved for CLS
        else:
            cls_token = None
            work_tokens = tokens
            k_work = k

        if k_work <= 0 or k_work >= work_tokens.shape[1]:
            return MergeResult(
                merged_tokens=tokens, original_count=T,
                merged_count=T, reduction_ratio=1.0,
                strategy="kmeans", time_ms=0,
            )

        merged_batch = []
        assignments_batch = []

        for b_i in range(B):
            x = work_tokens[b_i]  # (T_work, D)
            centroids, assignments = self._kmeans(x, k_work)
            merged_batch.append(centroids)
            assignments_batch.append(assignments)

        merged_tokens = torch.stack(merged_batch, dim=0)  # (B, k_work, D)

        if cls_token is not None:
            merged_tokens = torch.cat([cls_token, merged_tokens], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        new_T = merged_tokens.shape[1]

        return MergeResult(
            merged_tokens=merged_tokens,
            original_count=T,
            merged_count=new_T,
            reduction_ratio=new_T / T,
            strategy="kmeans",
            time_ms=elapsed,
        )

    def _kmeans(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run K-Means on a single sequence.

        Args:
            x: (T, D) token features
            k: number of clusters

        Returns:
            centroids: (k, D)
            assignments: (T,) cluster indices
        """
        T, D = x.shape
        device = x.device

        # Initialize centroids with K-Means++ style
        indices = [torch.randint(T, (1,), device=device).item()]
        for _ in range(k - 1):
            dists = torch.cdist(x, x[indices])  # (T, len(indices))
            min_dists = dists.min(dim=1).values  # (T,)
            probs = min_dists / (min_dists.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)

        centroids = x[indices].clone()  # (k, D)

        for _ in range(self.max_iters):
            # Assign each token to nearest centroid
            dists = torch.cdist(x, centroids)  # (T, k)
            assignments = dists.argmin(dim=1)   # (T,)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=device)
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = x[mask].mean(dim=0)
                    counts[c] = mask.sum()
                else:
                    # Empty cluster: reinitialize to random token
                    new_centroids[c] = x[torch.randint(T, (1,), device=device)]
                    counts[c] = 1

            # Check convergence
            shift = (new_centroids - centroids).norm(dim=1).max()
            centroids = new_centroids
            if shift < 1e-6:
                break

        return centroids, assignments


# ════════════════════════════════════════════════════════════════
# Strategy 3: Average Pooling Merging
# ════════════════════════════════════════════════════════════════

class AveragePoolMerging:
    """
    Merge tokens by grouping adjacent tokens and averaging.

    Simplest and fastest approach. Groups consecutive tokens into
    windows of size `window_size` and replaces each window with
    the mean of its tokens.

    Pros: O(T) complexity, no learnable parameters, very fast.
    Cons: Ignores semantic similarity — merges based purely on position.
    """

    def __init__(self, window_size: int = 2, stride: Optional[int] = None,
                 protect_cls: bool = True):
        """
        Args:
            window_size: Number of adjacent tokens to merge into one.
            stride: Step between windows. If None, equals window_size (non-overlapping).
            protect_cls: If True, the first token is kept as-is.
        """
        self.window_size = window_size
        self.stride = stride or window_size
        self.protect_cls = protect_cls

    def __call__(self, tokens: torch.Tensor) -> MergeResult:
        """
        Merge tokens using average pooling over windows.

        Args:
            tokens: (B, T, D)

        Returns:
            MergeResult
        """
        B, T, D = tokens.shape
        start = time.perf_counter()

        if self.protect_cls and T > 2:
            cls_token = tokens[:, :1, :]    # (B, 1, D)
            work_tokens = tokens[:, 1:, :]  # (B, T-1, D)
        else:
            cls_token = None
            work_tokens = tokens

        T_work = work_tokens.shape[1]

        # Use 1D average pooling on the token dimension
        # Reshape: (B, T_work, D) → (B, D, T_work) for avg_pool1d
        x = work_tokens.transpose(1, 2)  # (B, D, T_work)
        pooled = F.avg_pool1d(x, kernel_size=self.window_size,
                              stride=self.stride, ceil_mode=True)  # (B, D, T')
        merged_tokens = pooled.transpose(1, 2)  # (B, T', D)

        if cls_token is not None:
            merged_tokens = torch.cat([cls_token, merged_tokens], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        new_T = merged_tokens.shape[1]

        return MergeResult(
            merged_tokens=merged_tokens,
            original_count=T,
            merged_count=new_T,
            reduction_ratio=new_T / T,
            strategy="average_pool",
            time_ms=elapsed,
        )


# ════════════════════════════════════════════════════════════════
# Token Merging Wrapper for ViT / Transformer Models
# ════════════════════════════════════════════════════════════════

class TokenMergingWrapper(nn.Module):
    """
    Wraps a Vision Transformer (or any model that processes token sequences)
    to apply token merging at specified layers.

    This hooks into the model's transformer blocks and merges tokens
    between attention layers, reducing the sequence length progressively.

    Usage:
        model = ViTForImageClassification.from_pretrained(...)
        wrapped = TokenMergingWrapper(model, strategy="bipartite", ratio=0.5)
        output = wrapped(pixel_values=images)
    """

    def __init__(self, model: nn.Module,
                 strategy: str = "bipartite",
                 ratio: float = 0.5,
                 merge_layers: Optional[List[int]] = None,
                 protect_cls: bool = True,
                 **strategy_kwargs):
        """
        Args:
            model: The base transformer model.
            strategy: "bipartite", "kmeans", or "average_pool".
            ratio: Fraction of tokens to keep per merge operation.
            merge_layers: Which layer indices to merge at. If None, merges
                          at evenly spaced layers (every other layer).
            protect_cls: Never merge the CLS/first token.
            **strategy_kwargs: Extra args passed to the strategy constructor.
        """
        super().__init__()
        self.model = model
        self.ratio = ratio
        self.protect_cls = protect_cls
        self.merge_layers = merge_layers
        self._hooks = []
        self._merge_count = 0

        # Initialize strategy
        self.strategy_name = strategy
        self.merger = self._create_strategy(strategy, ratio, protect_cls,
                                            **strategy_kwargs)

        # Auto-detect transformer layers and install hooks
        self._install_hooks()

    def _create_strategy(self, name: str, ratio: float, protect_cls: bool,
                         **kwargs):
        if name == "bipartite":
            return BipartiteSoftMatching(ratio=ratio, protect_cls=protect_cls, **kwargs)
        elif name == "kmeans":
            return KMeansMerging(ratio=ratio, protect_cls=protect_cls, **kwargs)
        elif name == "average_pool":
            window = max(2, int(1 / ratio)) if ratio < 1 else 2
            return AveragePoolMerging(window_size=kwargs.get("window_size", window),
                                      protect_cls=protect_cls)
        else:
            raise ValueError(f"Unknown strategy: {name}. "
                             f"Choose from: bipartite, kmeans, average_pool")

    def _install_hooks(self):
        """Find transformer encoder layers and install merge hooks."""
        layers = self._find_transformer_layers()
        if not layers:
            warnings.warn("No transformer layers found — token merging will "
                          "not be applied. The model will run normally.")
            return

        # Determine which layers to merge at
        n_layers = len(layers)
        if self.merge_layers is not None:
            target_indices = set(self.merge_layers)
        else:
            # Default: merge at every other layer, starting from layer 1
            target_indices = set(range(1, n_layers, 2))

        for idx, (name, layer) in enumerate(layers):
            if idx in target_indices:
                hook = layer.register_forward_hook(self._make_merge_hook(name, idx))
                self._hooks.append(hook)

    def _find_transformer_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Heuristically find the repeated transformer blocks in a model.
        Works with HuggingFace ViT, GPT-2, and standard nn.TransformerEncoder.
        """
        layers = []

        # Strategy 1: Look for common HuggingFace patterns
        for name, module in self.model.named_modules():
            module_name = type(module).__name__.lower()
            # HuggingFace ViT: ViTLayer, BertLayer, etc.
            # HuggingFace GPT-2: GPT2Block
            # Standard: TransformerEncoderLayer
            if any(kw in module_name for kw in
                   ["vitlayer", "bertlayer", "gpt2block",
                    "transformerencoderlayer", "block", "decoderlayer"]):
                # Avoid matching parent containers (but not "encoder" inside "encoderlayer")
                if not any(module_name == kw for kw in
                           ["modulelist", "sequential", "transformerencoder",
                            "transformerdecoder"]):
                    layers.append((name, module))

        # Strategy 2: Look for numbered children under common parent names
        if not layers:
            for name, module in self.model.named_modules():
                children = list(module.named_children())
                if len(children) >= 3:
                    # Check if children are numbered and look like repeated blocks
                    try:
                        nums = [int(c[0]) for c in children if c[0].isdigit()]
                        if len(nums) >= 3 and nums == list(range(len(nums))):
                            for child_name, child in children:
                                if child_name.isdigit():
                                    layers.append((f"{name}.{child_name}", child))
                            break
                    except ValueError:
                        continue

        return layers

    def _make_merge_hook(self, layer_name: str, layer_idx: int):
        """Create a forward hook that merges tokens after a layer's output."""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden = output[0]  # (B, T, D)
                rest = output[1:]
            elif isinstance(output, torch.Tensor):
                hidden = output
                rest = None
            else:
                # Can't merge unknown output format
                return output

            if hidden.ndim != 3:
                return output  # Not a token sequence

            # Apply token merging
            result = self.merger(hidden)
            self._merge_count += 1

            if rest is not None:
                return (result.merged_tokens,) + rest
            return result.merged_tokens

        return hook

    def forward(self, *args, **kwargs):
        self._merge_count = 0
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        """Remove all merge hooks and restore original model behavior."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


# ════════════════════════════════════════════════════════════════
# Profiling: Compare All Strategies
# ════════════════════════════════════════════════════════════════

class TokenMergingProfiler:
    """
    Profiles all three token merging strategies on a given token sequence.
    Useful for determining which strategy works best for a given model.
    """

    def __init__(self, ratios: List[float] = [0.5, 0.7, 0.3]):
        """
        Args:
            ratios: List of keep-ratios to test (0.5 = keep 50% of tokens).
        """
        self.ratios = ratios

    def profile(self, tokens: torch.Tensor,
                model_name: str = "unknown",
                protect_cls: bool = True) -> TokenMergingProfile:
        """
        Run all three strategies and compare results.

        Args:
            tokens: (B, T, D) — a batch of token sequences.
                    Can be obtained from a ViT encoder's intermediate output.
        """
        B, T, D = tokens.shape
        ratio = self.ratios[0]  # primary ratio for comparison

        profile = TokenMergingProfile(
            model_name=model_name,
            input_shape=tuple(tokens.shape),
        )

        # Strategy 1: Bipartite Soft Matching
        bsm = BipartiteSoftMatching(ratio=ratio, protect_cls=protect_cls)
        result_bsm = bsm(tokens)
        profile.results["bipartite"] = result_bsm

        # Strategy 2: K-Means
        kmeans = KMeansMerging(ratio=ratio, protect_cls=protect_cls, max_iters=10)
        result_km = kmeans(tokens)
        profile.results["kmeans"] = result_km

        # Strategy 3: Average Pooling
        window = max(2, int(1 / (1 - ratio + 1e-8)))
        avgpool = AveragePoolMerging(window_size=window, protect_cls=protect_cls)
        result_ap = avgpool(tokens)
        profile.results["average_pool"] = result_ap

        # Determine best strategy (lowest reduction ratio = most compression)
        best_name = min(profile.results,
                        key=lambda k: profile.results[k].reduction_ratio)
        profile.best_strategy = best_name
        profile.best_reduction = profile.results[best_name].reduction_ratio

        return profile

    def profile_across_ratios(self, tokens: torch.Tensor,
                              model_name: str = "unknown",
                              protect_cls: bool = True) -> Dict[float, TokenMergingProfile]:
        """Profile all strategies across multiple keep-ratios."""
        results = {}
        for ratio in self.ratios:
            self_copy = TokenMergingProfiler(ratios=[ratio])
            results[ratio] = self_copy.profile(tokens, model_name, protect_cls)
        return results


# ════════════════════════════════════════════════════════════════
# Convenience Functions
# ════════════════════════════════════════════════════════════════

def apply_token_merging(model: nn.Module,
                        strategy: str = "bipartite",
                        ratio: float = 0.5,
                        merge_layers: Optional[List[int]] = None,
                        **kwargs) -> TokenMergingWrapper:
    """
    Apply token merging to any transformer model.

    Args:
        model: A ViT or Transformer model.
        strategy: "bipartite", "kmeans", or "average_pool".
        ratio: Fraction of tokens to keep (0.5 = 50% reduction).
        merge_layers: Which transformer layers to merge at.

    Returns:
        TokenMergingWrapper that applies merging during forward pass.

    Example:
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        merged_model = apply_token_merging(model, strategy="bipartite", ratio=0.7)
        output = merged_model(pixel_values=images)
    """
    return TokenMergingWrapper(
        model, strategy=strategy, ratio=ratio,
        merge_layers=merge_layers, **kwargs
    )


def merge_tokens(tokens: torch.Tensor,
                 strategy: str = "bipartite",
                 ratio: float = 0.5,
                 protect_cls: bool = True,
                 **kwargs) -> MergeResult:
    """
    One-shot token merging on a tensor.

    Args:
        tokens: (B, T, D) token features.
        strategy: "bipartite", "kmeans", or "average_pool".
        ratio: Fraction of tokens to keep.
        protect_cls: Protect the first token from merging.

    Returns:
        MergeResult with merged tokens.
    """
    if strategy == "bipartite":
        merger = BipartiteSoftMatching(ratio=ratio, protect_cls=protect_cls, **kwargs)
    elif strategy == "kmeans":
        merger = KMeansMerging(ratio=ratio, protect_cls=protect_cls, **kwargs)
    elif strategy == "average_pool":
        window = kwargs.get("window_size", max(2, int(1 / (1 - ratio + 1e-8))))
        merger = AveragePoolMerging(window_size=window, protect_cls=protect_cls)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return merger(tokens)


def compare_strategies(tokens: torch.Tensor,
                       ratio: float = 0.5,
                       model_name: str = "unknown",
                       verbose: bool = True) -> TokenMergingProfile:
    """
    Compare all three token merging strategies on the same input.

    Args:
        tokens: (B, T, D) token features.
        ratio: Keep ratio for comparison.
        verbose: Print summary.

    Returns:
        TokenMergingProfile with results for each strategy.
    """
    profiler = TokenMergingProfiler(ratios=[ratio])
    profile = profiler.profile(tokens, model_name=model_name)
    if verbose:
        print(profile.summary())
    return profile
