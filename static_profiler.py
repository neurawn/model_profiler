"""
Static Profiler Module
======================
Analyzes model architecture and parameter distributions WITHOUT running inference.
Extracts structural features that inform compression method selection.

Key metrics:
- Parameter count per layer and total
- Parameter distribution statistics (mean, std, skewness, kurtosis)
- Layer type composition (linear, conv, attention, norm, etc.)
- Weight matrix shapes and rank estimates
- Sparsity analysis (near-zero weights)
- Uniformity score (how evenly parameters are distributed across layers)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json


@dataclass
class LayerProfile:
    """Profile for a single layer."""
    name: str
    layer_type: str
    param_count: int
    shape: Tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    sparsity: float  # fraction of near-zero weights
    rank_estimate: Optional[int]  # estimated effective rank
    dtype: str


@dataclass
class StaticProfile:
    """Complete static profile for a model."""
    model_name: str
    total_params: int
    trainable_params: int
    total_size_mb: float
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    layer_type_counts: Dict[str, int] = field(default_factory=dict)
    layer_type_params: Dict[str, int] = field(default_factory=dict)

    # Distribution analysis
    param_uniformity_score: float = 0.0  # 0=very uneven, 1=perfectly uniform
    overall_sparsity: float = 0.0
    weight_entropy: float = 0.0

    # Architecture classification features
    has_attention: bool = False
    has_convolutions: bool = False
    has_residual_connections: bool = False
    dominant_layer_type: str = ""
    depth: int = 0  # number of major blocks/layers

    # Compression-relevant features
    quantization_friendliness: float = 0.0  # 0-1 score
    pruning_friendliness: float = 0.0  # 0-1 score
    distillation_friendliness: float = 0.0  # 0-1 score

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "total_size_mb": round(self.total_size_mb, 2),
            "layer_type_counts": self.layer_type_counts,
            "layer_type_params": self.layer_type_params,
            "param_uniformity_score": round(self.param_uniformity_score, 4),
            "overall_sparsity": round(self.overall_sparsity, 4),
            "weight_entropy": round(self.weight_entropy, 4),
            "has_attention": self.has_attention,
            "has_convolutions": self.has_convolutions,
            "dominant_layer_type": self.dominant_layer_type,
            "depth": self.depth,
            "quantization_friendliness": round(self.quantization_friendliness, 4),
            "pruning_friendliness": round(self.pruning_friendliness, 4),
            "distillation_friendliness": round(self.distillation_friendliness, 4),
        }
        return result

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  STATIC PROFILE: {self.model_name}",
            f"{'='*60}",
            f"  Total Parameters:     {self.total_params:,}",
            f"  Trainable Parameters: {self.trainable_params:,}",
            f"  Model Size:           {self.total_size_mb:.2f} MB",
            f"  Depth (major blocks): {self.depth}",
            f"{'─'*60}",
            f"  Layer Type Distribution:",
        ]
        for ltype, count in sorted(self.layer_type_counts.items(), key=lambda x: -x[1]):
            params = self.layer_type_params.get(ltype, 0)
            lines.append(f"    {ltype:20s}  count={count:4d}  params={params:>12,}")

        lines += [
            f"{'─'*60}",
            f"  Architecture Features:",
            f"    Has Attention:      {self.has_attention}",
            f"    Has Convolutions:   {self.has_convolutions}",
            f"    Dominant Type:      {self.dominant_layer_type}",
            f"{'─'*60}",
            f"  Distribution Analysis:",
            f"    Param Uniformity:   {self.param_uniformity_score:.4f}  (1=uniform, 0=uneven)",
            f"    Overall Sparsity:   {self.overall_sparsity:.4f}  (fraction near-zero)",
            f"    Weight Entropy:     {self.weight_entropy:.4f}",
            f"{'─'*60}",
            f"  Compression Friendliness Scores (0-1):",
            f"    Quantization:       {self.quantization_friendliness:.4f}",
            f"    Pruning:            {self.pruning_friendliness:.4f}",
            f"    Distillation:       {self.distillation_friendliness:.4f}",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


class StaticProfiler:
    """
    Analyzes a PyTorch model's architecture and parameter distributions
    without running any inference. Produces features useful for deciding
    which compression method to apply.
    """

    # Map common module types to categories
    LAYER_TYPE_MAP = {
        "Linear": "linear",
        "Conv1d": "conv",
        "Conv2d": "conv",
        "Conv3d": "conv",
        "ConvTranspose2d": "conv",
        "MultiheadAttention": "attention",
        "LayerNorm": "norm",
        "BatchNorm1d": "norm",
        "BatchNorm2d": "norm",
        "GroupNorm": "norm",
        "Embedding": "embedding",
        "LSTM": "recurrent",
        "GRU": "recurrent",
        "RNN": "recurrent",
        "Dropout": "dropout",
        "ReLU": "activation",
        "GELU": "activation",
        "SiLU": "activation",
        "Softmax": "activation",
    }

    SPARSITY_THRESHOLD = 1e-6  # weights below this magnitude are "near-zero"

    def __init__(self, sparsity_threshold: float = 1e-6):
        self.sparsity_threshold = sparsity_threshold

    def profile(self, model: nn.Module, model_name: str = "unknown") -> StaticProfile:
        """Run full static analysis on a model."""
        model.eval()

        profile = StaticProfile(model_name=model_name, total_params=0,
                                trainable_params=0, total_size_mb=0.0)

        # ── 1. Enumerate all layers and their parameters ──
        layer_param_counts = []
        all_weights = []

        for name, module in model.named_modules():
            module_type = type(module).__name__
            category = self._categorize_layer(module_type)

            # Count this module type
            if category != "container":
                profile.layer_type_counts[category] = \
                    profile.layer_type_counts.get(category, 0) + 1

            # Profile each parameter tensor in this module (only direct params)
            for pname, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname
                data = param.data.detach().float().cpu()

                lp = self._profile_tensor(full_name, category, data)
                profile.layer_profiles.append(lp)
                layer_param_counts.append(lp.param_count)

                profile.layer_type_params[category] = \
                    profile.layer_type_params.get(category, 0) + lp.param_count

                all_weights.append(data.flatten())

        # ── 2. Aggregate statistics ──
        profile.total_params = sum(p.numel() for p in model.parameters())
        profile.trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        profile.total_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        # ── 3. Distribution analysis ──
        if all_weights:
            all_flat = torch.cat(all_weights)
            profile.overall_sparsity = (
                (all_flat.abs() < self.sparsity_threshold).float().mean().item()
            )
            profile.weight_entropy = self._compute_entropy(all_flat)

        # Uniformity: how evenly distributed are params across layers
        if layer_param_counts:
            profile.param_uniformity_score = self._compute_uniformity(layer_param_counts)

        # ── 4. Architecture classification ──
        profile.has_attention = "attention" in profile.layer_type_counts or \
            self._detect_attention_pattern(model)
        profile.has_convolutions = "conv" in profile.layer_type_counts
        profile.dominant_layer_type = max(
            profile.layer_type_params, key=profile.layer_type_params.get
        ) if profile.layer_type_params else "unknown"
        profile.depth = self._estimate_depth(model)

        # ── 5. Compression friendliness scores ──
        self._compute_compression_scores(profile)

        return profile

    def _categorize_layer(self, module_type: str) -> str:
        """Map a module class name to a broad category."""
        if module_type in self.LAYER_TYPE_MAP:
            return self.LAYER_TYPE_MAP[module_type]
        # Heuristic fallbacks
        lower = module_type.lower()
        if "conv" in lower:
            return "conv"
        if "linear" in lower or "dense" in lower:
            return "linear"
        if "attention" in lower:
            return "attention"
        if "norm" in lower:
            return "norm"
        if "embed" in lower:
            return "embedding"
        if "drop" in lower:
            return "dropout"
        # Containers (Sequential, ModuleList, etc.) — skip
        return "container"

    def _profile_tensor(self, name: str, layer_type: str,
                        data: torch.Tensor) -> LayerProfile:
        """Compute statistics for a single parameter tensor."""
        flat = data.flatten().numpy()
        n = len(flat)

        mean = float(np.mean(flat))
        std = float(np.std(flat))
        min_val = float(np.min(flat))
        max_val = float(np.max(flat))

        # Skewness and kurtosis
        if std > 1e-10:
            skewness = float(np.mean(((flat - mean) / std) ** 3))
            kurtosis = float(np.mean(((flat - mean) / std) ** 4) - 3)  # excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        sparsity = float(np.mean(np.abs(flat) < self.sparsity_threshold))

        # Rank estimate for 2D weight matrices
        rank_estimate = None
        if data.ndim == 2 and min(data.shape) > 1:
            try:
                sv = torch.linalg.svdvals(data)
                # Effective rank: number of singular values > 1% of max
                threshold = 0.01 * sv[0].item()
                rank_estimate = int((sv > threshold).sum().item())
            except Exception:
                rank_estimate = None

        return LayerProfile(
            name=name,
            layer_type=layer_type,
            param_count=n,
            shape=tuple(data.shape),
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            skewness=skewness,
            kurtosis=kurtosis,
            sparsity=sparsity,
            rank_estimate=rank_estimate,
            dtype=str(data.dtype),
        )

    def _compute_uniformity(self, param_counts: List[int]) -> float:
        """
        Measure how uniformly parameters are distributed across layers.
        Uses normalized entropy: 1.0 = perfectly uniform, 0.0 = all in one layer.

        Key insight from proposal: Transformers have more uniform distribution,
        CNNs have uneven distribution (doubling channels at each stage).
        """
        counts = np.array(param_counts, dtype=np.float64)
        total = counts.sum()
        if total == 0 or len(counts) <= 1:
            return 1.0

        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(counts))

        if max_entropy == 0:
            return 1.0
        return float(entropy / max_entropy)

    def _compute_entropy(self, weights: torch.Tensor, n_bins: int = 256) -> float:
        """Compute histogram entropy of weight values."""
        flat = weights.numpy()
        hist, _ = np.histogram(flat, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (flat.max() - flat.min()) / n_bins if flat.max() != flat.min() else 1.0
        probs = hist * bin_width
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _detect_attention_pattern(self, model: nn.Module) -> bool:
        """Detect attention-like patterns (Q, K, V projections) even without
        explicit MultiheadAttention modules (e.g., HuggingFace models)."""
        param_names = [name for name, _ in model.named_parameters()]
        attention_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj",
                              "query", "key", "value", "self_attn"]
        for name in param_names:
            lower = name.lower()
            if any(kw in lower for kw in attention_keywords):
                return True
        return False

    def _estimate_depth(self, model: nn.Module) -> int:
        """Estimate the number of major repeated blocks in the model."""
        # Count modules that look like repeated blocks
        block_keywords = ["layer", "block", "stage", "encoder", "decoder",
                          "transformer", "resblock", "bottleneck"]
        block_count = 0
        for name, _ in model.named_modules():
            parts = name.split(".")
            for part in parts:
                lower = part.lower()
                if any(kw in lower for kw in block_keywords):
                    try:
                        # Check if it's indexed (e.g., layer.0, block.1)
                        idx = int(parts[parts.index(part) + 1]) if parts.index(part) + 1 < len(parts) else -1
                        block_count = max(block_count, idx + 1)
                    except (ValueError, IndexError):
                        pass
        return max(block_count, 1)

    def _compute_compression_scores(self, profile: StaticProfile) -> None:
        """
        Compute heuristic scores for how well each compression method
        would work on this model, based on static analysis.

        Hypothesis from proposal:
        - Quantization works better for models with uniform parameter distribution
          (e.g., Transformers)
        - Structured pruning works better for models with uneven distribution
          (e.g., CNNs with doubling channels)
        - Distillation is generally applicable but works best for larger models
        """
        # ── Quantization friendliness ──
        # Favors: uniform distribution, low outliers, normal-ish weight distributions
        quant_score = 0.0
        quant_score += profile.param_uniformity_score * 0.4  # uniform = good for quant

        # Low kurtosis (fewer outliers) is better for quantization
        avg_kurtosis = np.mean([lp.kurtosis for lp in profile.layer_profiles]) \
            if profile.layer_profiles else 0
        kurtosis_penalty = min(abs(avg_kurtosis) / 10.0, 0.3)
        quant_score += (0.3 - kurtosis_penalty)

        # Attention-heavy models tend to be more quantization-friendly
        if profile.has_attention:
            quant_score += 0.2

        # Larger models benefit more from quantization
        size_bonus = min(profile.total_size_mb / 500.0, 0.1)
        quant_score += size_bonus

        profile.quantization_friendliness = np.clip(quant_score, 0, 1)

        # ── Pruning friendliness ──
        # Favors: existing sparsity, conv-heavy architectures, uneven distribution
        prune_score = 0.0
        prune_score += (1 - profile.param_uniformity_score) * 0.3  # uneven = good for pruning
        prune_score += min(profile.overall_sparsity * 5, 0.3)  # existing sparsity

        if profile.has_convolutions:
            prune_score += 0.25  # structured pruning works well on conv filters

        # High sparsity in individual layers suggests pruning potential
        avg_sparsity = np.mean([lp.sparsity for lp in profile.layer_profiles]) \
            if profile.layer_profiles else 0
        prune_score += min(avg_sparsity * 3, 0.15)

        profile.pruning_friendliness = np.clip(prune_score, 0, 1)

        # ── Distillation friendliness ──
        # Favors: larger models (more to compress), deeper models
        distill_score = 0.0
        distill_score += min(profile.total_params / 1e9, 0.4)  # larger = more benefit
        distill_score += min(profile.depth / 50.0, 0.3)
        distill_score += 0.2  # generally applicable baseline

        # Models with clear block structure are easier to distill
        if profile.depth > 3:
            distill_score += 0.1

        profile.distillation_friendliness = np.clip(distill_score, 0, 1)


    def get_layer_parameter_distribution(self, profile: StaticProfile) -> Dict[str, List]:
        """
        Return data suitable for plotting parameter distribution across layers.
        Useful for visualizing the uniformity hypothesis.
        """
        names = [lp.name for lp in profile.layer_profiles]
        counts = [lp.param_count for lp in profile.layer_profiles]
        types = [lp.layer_type for lp in profile.layer_profiles]
        return {"names": names, "counts": counts, "types": types}

    def get_weight_statistics_table(self, profile: StaticProfile) -> List[Dict]:
        """Return a table of per-layer weight statistics."""
        rows = []
        for lp in profile.layer_profiles:
            rows.append({
                "name": lp.name,
                "type": lp.layer_type,
                "params": lp.param_count,
                "shape": str(lp.shape),
                "mean": round(lp.mean, 6),
                "std": round(lp.std, 6),
                "skewness": round(lp.skewness, 4),
                "kurtosis": round(lp.kurtosis, 4),
                "sparsity": round(lp.sparsity, 6),
                "rank_est": lp.rank_estimate,
            })
        return rows
