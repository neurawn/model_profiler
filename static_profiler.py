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
class NodeProfile:
    """Per-node profile derived from the exported graph."""
    name: str
    op_type: str
    flops: int = 0
    params: int = 0
    activation_memory_bytes: int = 0  # output tensor size
    arithmetic_intensity: float = 0.0  # FLOPs / bytes accessed
    roofline_bound: str = ""  # "compute" or "memory"
    compressible: bool = False
    compress_methods: List[str] = field(default_factory=list)


@dataclass
class BlockProfile:
    """Aggregated profile for a transformer/CNN block."""
    block_name: str
    block_index: int
    total_flops: int = 0
    total_params: int = 0
    total_activation_mb: float = 0.0
    node_count: int = 0
    compressible_count: int = 0
    dominant_op: str = ""
    avg_arithmetic_intensity: float = 0.0


@dataclass
class KVCacheEstimate:
    """KV cache memory estimate for transformer models."""
    num_layers: int = 0
    num_heads: int = 0
    head_dim: int = 0
    hidden_dim: int = 0
    dtype_bytes: int = 2  # fp16 default
    estimates: Dict[int, float] = field(default_factory=dict)
    # context_length -> memory_mb

    def summary(self) -> str:
        lines = [f"    KV Cache Budget (per batch, {self.num_layers}L x "
                 f"{self.num_heads}H x {self.head_dim}D, "
                 f"{self.dtype_bytes}B/elem):"]
        for ctx_len, mem_mb in sorted(self.estimates.items()):
            lines.append(f"      ctx={ctx_len:>6d}  →  {mem_mb:>8.2f} MB")
        return "\n".join(lines)


@dataclass
class ParetoLayer:
    """A layer ranked by cost share for compression targeting."""
    name: str
    op_type: str
    flops: int
    params: int
    flops_share: float  # fraction of total FLOPs
    param_share: float  # fraction of total params
    cumulative_flops_share: float  # cumulative from top
    cumulative_param_share: float


@dataclass
class QuantAssignment:
    """Bit-width assignment for a layer."""
    name: str
    op_type: str
    assigned_bits: int  # 4, 8, or 16
    reason: str
    arithmetic_intensity: float
    params: int


@dataclass
class PruningCandidate:
    """Layer identified as a pruning candidate."""
    name: str
    op_type: str
    params: int
    flops: int
    cost_to_param_ratio: float  # FLOPs per param — low = prune-friendly
    param_share: float
    reason: str


@dataclass
class KVQuantPriority:
    """KV cache quantization priority per layer."""
    layer_name: str
    layer_index: int
    kv_memory_mb: float  # at target context
    kv_share: float  # fraction of total KV memory
    recommended_bits: int  # 4, 8, or 16
    reason: str


@dataclass
class TokenMergingEstimate:
    """Token merging estimate for a given keep ratio."""
    keep_ratio: float
    original_tokens: int
    merged_tokens: int
    original_flops: int
    merged_flops: int
    flops_reduction: float  # fraction saved
    attn_speedup: float
    ffn_speedup: float
    overall_speedup: float
    recommended_strategy: str
    reason: str


@dataclass
class CompressionPlan:
    """Complete compression plan derived from graph static profiling."""
    model_name: str

    # 0. Token merging (ViT/VLM only, applied before quantization)
    token_merging: Optional[List[TokenMergingEstimate]] = None
    has_token_merging: bool = False

    # 1. Pareto ranking
    pareto_layers: List[ParetoLayer] = field(default_factory=list)
    top_k_cover_80_flops: int = 0  # how many layers cover 80% of FLOPs

    # 2. Quant assignment (accounts for token merging if applicable)
    quant_assignments: List[QuantAssignment] = field(default_factory=list)
    avg_bits: float = 0.0  # effective average bit-width

    # 3. Pruning candidates
    pruning_candidates: List[PruningCandidate] = field(default_factory=list)
    total_prunable_params: int = 0

    # 4. KV quant priority
    kv_priorities: List[KVQuantPriority] = field(default_factory=list)
    target_context: int = 0
    total_kv_mb: float = 0.0
    kv_after_quant_mb: float = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  COMPRESSION PLAN: {self.model_name}",
            f"{'='*70}",
        ]

        # 0. Token merging (ViT/VLM only)
        if self.has_token_merging and self.token_merging:
            lines += [
                f"\n  0. TOKEN MERGING (apply before quantization)",
                f"     Reduces sequence length → quadratic attention savings + linear FFN savings",
                f"     {'Ratio':<8s} {'Tokens':>12s} {'FLOPs':>16s} {'Saved':>8s} "
                f"{'Attn':>7s} {'FFN':>7s} {'Overall':>8s} {'Strategy'}",
                f"     {'─'*8} {'─'*12} {'─'*16} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*20}",
            ]
            for tm in self.token_merging:
                lines.append(
                    f"     {tm.keep_ratio:<8.0%} "
                    f"{tm.original_tokens:>5d} → {tm.merged_tokens:<5d} "
                    f"{tm.original_flops/1e9:.1f}G → {tm.merged_flops/1e9:.1f}G "
                    f"{tm.flops_reduction:>7.1%} "
                    f"{tm.attn_speedup:>6.2f}x {tm.ffn_speedup:>6.2f}x "
                    f"{tm.overall_speedup:>7.2f}x {tm.recommended_strategy}"
                )
            best = max(self.token_merging, key=lambda t: t.overall_speedup)
            lines += [
                f"",
                f"     Recommended: {best.keep_ratio:.0%} keep ratio with {best.recommended_strategy}",
                f"     {best.reason}",
                f"     Apply token merging FIRST, then quantize the merged model",
            ]

        # 1. Pareto ranking
        lines += [
            f"\n  1. PARETO RANKING (top layers by cost share)",
            f"     Layers covering 80% of FLOPs: {self.top_k_cover_80_flops}",
            f"     {'Layer':<40s} {'FLOPs%':>7s} {'Params%':>8s} {'Cum.F%':>7s}",
            f"     {'─'*40} {'─'*7} {'─'*8} {'─'*7}",
        ]
        for pl in self.pareto_layers[:15]:
            lines.append(
                f"     {pl.name[:40]:<40s} {pl.flops_share:>6.1%} "
                f"{pl.param_share:>7.1%} {pl.cumulative_flops_share:>6.1%}"
            )
        if len(self.pareto_layers) > 15:
            lines.append(f"     ... ({len(self.pareto_layers) - 15} more)")

        # 2. Quant assignment
        lines += [
            f"\n  2. QUANTIZATION ASSIGNMENT (avg bits: {self.avg_bits:.1f})",
            f"     {'Layer':<40s} {'Bits':>5s} {'Params':>12s} {'Reason'}",
            f"     {'─'*40} {'─'*5} {'─'*12} {'─'*30}",
        ]
        for qa in self.quant_assignments[:20]:
            lines.append(
                f"     {qa.name[:40]:<40s} {qa.assigned_bits:>4d}b "
                f"{qa.params:>12,} {qa.reason}"
            )
        if len(self.quant_assignments) > 20:
            lines.append(f"     ... ({len(self.quant_assignments) - 20} more)")

        # 3. Pruning candidates
        if self.pruning_candidates:
            lines += [
                f"\n  3. PRUNING CANDIDATES ({len(self.pruning_candidates)} layers, "
                f"{self.total_prunable_params:,} params)",
                f"     {'Layer':<40s} {'Params':>12s} {'FLOPs/P':>8s} {'Reason'}",
                f"     {'─'*40} {'─'*12} {'─'*8} {'─'*30}",
            ]
            for pc in self.pruning_candidates[:15]:
                lines.append(
                    f"     {pc.name[:40]:<40s} {pc.params:>12,} "
                    f"{pc.cost_to_param_ratio:>7.1f} {pc.reason}"
                )
            if len(self.pruning_candidates) > 15:
                lines.append(f"     ... ({len(self.pruning_candidates) - 15} more)")
        else:
            lines.append(f"\n  3. PRUNING CANDIDATES: none identified")

        # 4. KV quant priority
        if self.kv_priorities:
            lines += [
                f"\n  4. KV CACHE QUANTIZATION (ctx={self.target_context}, "
                f"{self.total_kv_mb:.1f}MB → {self.kv_after_quant_mb:.1f}MB)",
                f"     {'Layer':<30s} {'KV MB':>8s} {'Share':>7s} {'Bits':>5s} {'Reason'}",
                f"     {'─'*30} {'─'*8} {'─'*7} {'─'*5} {'─'*30}",
            ]
            for kvp in self.kv_priorities:
                lines.append(
                    f"     {kvp.layer_name[:30]:<30s} {kvp.kv_memory_mb:>7.2f} "
                    f"{kvp.kv_share:>6.1%} {kvp.recommended_bits:>4d}b {kvp.reason}"
                )
        else:
            lines.append(f"\n  4. KV CACHE QUANTIZATION: not applicable (non-transformer)")

        lines.append(f"\n{'='*70}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
        }
        if self.has_token_merging and self.token_merging:
            result["token_merging"] = [
                {
                    "keep_ratio": tm.keep_ratio,
                    "original_tokens": tm.original_tokens,
                    "merged_tokens": tm.merged_tokens,
                    "original_flops": tm.original_flops,
                    "merged_flops": tm.merged_flops,
                    "flops_reduction": round(tm.flops_reduction, 4),
                    "overall_speedup": round(tm.overall_speedup, 2),
                    "strategy": tm.recommended_strategy,
                }
                for tm in self.token_merging
            ]
        result["pareto"] = {
            "top_k_cover_80_flops": self.top_k_cover_80_flops,
            "layers": [
                {"name": pl.name, "flops_share": round(pl.flops_share, 4),
                 "param_share": round(pl.param_share, 4)}
                for pl in self.pareto_layers
            ],
        }
        result["quant_assignment"] = {
            "avg_bits": round(self.avg_bits, 2),
            "assignments": [
                {"name": qa.name, "bits": qa.assigned_bits,
                 "params": qa.params, "reason": qa.reason}
                for qa in self.quant_assignments
            ],
        }
        result["pruning_candidates"] = [
            {"name": pc.name, "params": pc.params,
             "cost_to_param_ratio": round(pc.cost_to_param_ratio, 2),
             "reason": pc.reason}
            for pc in self.pruning_candidates
        ]
        result["kv_quant"] = {
            "target_context": self.target_context,
            "total_kv_mb": round(self.total_kv_mb, 2),
            "after_quant_mb": round(self.kv_after_quant_mb, 2),
            "priorities": [
                {"layer": kvp.layer_name, "kv_mb": round(kvp.kv_memory_mb, 2),
                 "bits": kvp.recommended_bits, "reason": kvp.reason}
                for kvp in self.kv_priorities
            ],
        }
        return result


@dataclass
class GraphStaticProfile:
    """Static profile derived from an exported graph."""
    model_name: str
    model_family: str

    # Per-node profiles
    node_profiles: List[NodeProfile] = field(default_factory=list)

    # Per-block aggregation
    block_profiles: List[BlockProfile] = field(default_factory=list)

    # KV cache
    kv_cache: Optional[KVCacheEstimate] = None

    # Totals
    total_flops: int = 0
    total_params: int = 0
    total_activation_mb: float = 0.0
    avg_arithmetic_intensity: float = 0.0
    pct_compute_bound: float = 0.0  # % of ops that are compute-bound
    pct_memory_bound: float = 0.0

    # Compression plan
    compression_plan: Optional[CompressionPlan] = None

    # Roofline parameters used
    peak_flops_per_sec: float = 0.0
    peak_bandwidth_bytes_per_sec: float = 0.0

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "total_flops": self.total_flops,
            "total_params": self.total_params,
            "total_activation_mb": round(self.total_activation_mb, 2),
            "avg_arithmetic_intensity": round(self.avg_arithmetic_intensity, 4),
            "pct_compute_bound": round(self.pct_compute_bound, 2),
            "pct_memory_bound": round(self.pct_memory_bound, 2),
        }
        if self.block_profiles:
            result["blocks"] = [
                {
                    "name": bp.block_name,
                    "flops": bp.total_flops,
                    "params": bp.total_params,
                    "activation_mb": round(bp.total_activation_mb, 2),
                    "nodes": bp.node_count,
                    "compressible": bp.compressible_count,
                    "dominant_op": bp.dominant_op,
                }
                for bp in self.block_profiles
            ]
        # Per-node details
        active_nodes = [n for n in self.node_profiles
                        if n.flops > 0 or n.params > 0]
        if active_nodes:
            result["nodes"] = [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "flops": n.flops,
                    "params": n.params,
                    "activation_memory_bytes": n.activation_memory_bytes,
                    "arithmetic_intensity": round(n.arithmetic_intensity, 2)
                        if n.arithmetic_intensity != float('inf') else "inf",
                    "roofline_bound": n.roofline_bound,
                    "compressible": n.compressible,
                }
                for n in sorted(active_nodes, key=lambda x: -x.flops)
            ]
        if self.kv_cache:
            result["kv_cache"] = {
                "num_layers": self.kv_cache.num_layers,
                "num_heads": self.kv_cache.num_heads,
                "head_dim": self.kv_cache.head_dim,
                "hidden_dim": self.kv_cache.hidden_dim,
                "dtype_bytes": self.kv_cache.dtype_bytes,
                "estimates_mb": {str(k): round(v, 2) for k, v in self.kv_cache.estimates.items()},
            }
        if self.compression_plan:
            result["compression_plan"] = self.compression_plan.to_dict()
        return result

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  GRAPH-BASED STATIC PROFILE: {self.model_name}",
            f"{'='*70}",
            f"  Total FLOPs:          {self.total_flops / 1e9:.3f} GFLOPs",
            f"  Total Params:         {self.total_params:,}",
            f"  Activation Memory:    {self.total_activation_mb:.2f} MB",
            f"  Avg Arith. Intensity: {self.avg_arithmetic_intensity:.2f} FLOPs/byte",
            f"{'─'*70}",
            f"  Roofline Placement:",
            f"    Compute-bound ops:  {self.pct_compute_bound:.1f}%",
            f"    Memory-bound ops:   {self.pct_memory_bound:.1f}%",
        ]

        if self.kv_cache and self.kv_cache.estimates:
            lines += [f"{'─'*70}", self.kv_cache.summary()]

        if self.block_profiles:
            lines += [
                f"{'─'*70}",
                f"  Per-Block Aggregation ({len(self.block_profiles)} blocks):",
                f"  {'Block':<30s} {'FLOPs':>12s} {'Params':>12s} "
                f"{'Act.MB':>8s} {'Compress':>9s}",
                f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8} {'─'*9}",
            ]
            for bp in self.block_profiles:
                comp_pct = (bp.compressible_count / bp.node_count * 100
                            if bp.node_count > 0 else 0)
                lines.append(
                    f"  {bp.block_name:<30s} {bp.total_flops / 1e6:>10.1f}M "
                    f"{bp.total_params:>12,} {bp.total_activation_mb:>7.2f} "
                    f"{comp_pct:>8.1f}%"
                )

        # Per-node breakdown (all nodes with params or FLOPs > 0)
        active_nodes = [n for n in self.node_profiles
                        if n.flops > 0 or n.params > 0]
        if active_nodes:
            lines += [
                f"{'─'*70}",
                f"  Per-Node FLOPs, Params & Activation Memory "
                f"({len(active_nodes)} active nodes):",
                f"  {'Node':<40s} {'FLOPs':>10s} {'Params':>12s} "
                f"{'Act.Mem':>10s} {'AI':>7s} {'Bound':>8s}",
                f"  {'─'*40} {'─'*10} {'─'*12} {'─'*10} {'─'*7} {'─'*8}",
            ]
            for n in sorted(active_nodes, key=lambda x: -x.flops):
                flops_str = (f"{n.flops / 1e6:.1f}M" if n.flops >= 1e6
                             else f"{n.flops / 1e3:.1f}K" if n.flops >= 1e3
                             else str(n.flops))
                act_str = (f"{n.activation_memory_bytes / 1024:.1f}KB"
                           if n.activation_memory_bytes >= 1024
                           else f"{n.activation_memory_bytes}B")
                ai_str = (f"{n.arithmetic_intensity:.1f}"
                          if n.arithmetic_intensity != float('inf') else "inf")
                bound_str = n.roofline_bound if n.roofline_bound else ""
                lines.append(
                    f"  {n.name[:40]:<40s} {flops_str:>10s} "
                    f"{n.params:>12,} {act_str:>10s} "
                    f"{ai_str:>7s} {bound_str:>8s}"
                )

        # KV cache budget
        if self.kv_cache and self.kv_cache.estimates:
            lines += [f"{'─'*70}", self.kv_cache.summary()]

        # Compression plan
        if self.compression_plan:
            lines.append(self.compression_plan.summary())

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


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
            "depth": self.depth,
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

        lines.append(f"{'='*60}\n")
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


    # ════════════════════════════════════════════════════════════════
    # Graph-based static profiling
    # ════════════════════════════════════════════════════════════════

    # FLOPs estimation per op type
    # For weight shapes (out, in): FLOPs = 2 * in * out per token/sample
    # _seq_len is set per-model before profiling to scale by sequence length
    _seq_len = 1  # default, overridden per model

    @staticmethod
    def _make_flop_rules(seq_len: int = 1):
        """Create FLOPs estimation rules scaled by sequence/token count."""
        return {
            # Linear: 2 * in * out * seq_len (weight shape is [out, in])
            "linear": lambda s: 2 * s[0] * s[1] * seq_len if len(s) >= 2 else 0,
            "addmm": lambda s: 2 * s[0] * s[1] * seq_len if len(s) >= 2 else 0,
            "mm": lambda s: 2 * s[0] * s[1] * seq_len if len(s) >= 2 else 0,
            "matmul": lambda s: 2 * s[0] * s[1] * seq_len if len(s) >= 2 else 0,
            "bmm": lambda s: 2 * s[0] * s[1] * s[2] if len(s) >= 3 else 0,
            # Conv2d: 2 * Cout * Cin * Kh * Kw (weight shape is [Cout, Cin, Kh, Kw])
            "conv2d": lambda s: 2 * s[0] * s[1] * s[2] * s[3] if len(s) >= 4 else (
                2 * s[0] * s[1] * s[2] if len(s) >= 3 else 0),
            # Attention: 4 * T^2 * D (Q@K^T + attn@V)
            "scaled_dot_product_attention": lambda s: (
                4 * seq_len * seq_len * s[1] if len(s) >= 2 else 0),
            # Norms: 5 * hidden * seq_len
            "layer_norm": lambda s: 5 * s[0] * seq_len if len(s) >= 1 else 0,
            "batch_norm": lambda s: 5 * s[0] if len(s) >= 1 else 0,
            # Activations: hidden * seq_len
            "gelu": lambda s: s[0] * seq_len if len(s) >= 1 else 0,
            "relu": lambda s: s[0] * seq_len if len(s) >= 1 else 0,
            "silu": lambda s: s[0] * seq_len if len(s) >= 1 else 0,
            # Embedding: lookup, ~0 FLOPs
            "embedding": lambda s: 0,
        }

    # Default hardware roofline (A100-like)
    DEFAULT_PEAK_TFLOPS = 312e12  # FP16 peak
    DEFAULT_BANDWIDTH = 2e12  # 2 TB/s HBM

    def profile_from_graph(self, model: nn.Module, graph_result,
                           model_name: str = "unknown",
                           model_family: str = "unknown",
                           target_context_lengths: Optional[List[int]] = None,
                           peak_tflops: Optional[float] = None,
                           peak_bandwidth: Optional[float] = None,
                           ) -> 'GraphStaticProfile':
        """
        Build a static profile from an exported graph + the original model.

        Computes per-node FLOPs, params, activation memory, arithmetic
        intensity, roofline placement, KV cache budget, and per-block
        aggregation.

        Args:
            model: The original PyTorch model (for parameter shapes).
            graph_result: A GraphExportResult from graph_export.py.
            target_context_lengths: Context lengths for KV cache estimation.
            peak_tflops: Hardware peak FLOPs/s for roofline.
            peak_bandwidth: Hardware peak bandwidth bytes/s for roofline.
        """
        if target_context_lengths is None:
            target_context_lengths = [512, 1024, 2048, 4096, 8192]

        hw_flops = peak_tflops or self.DEFAULT_PEAK_TFLOPS
        hw_bw = peak_bandwidth or self.DEFAULT_BANDWIDTH
        ridge_point = hw_flops / hw_bw  # arithmetic intensity threshold

        gsp = GraphStaticProfile(
            model_name=model_name,
            model_family=model_family,
            peak_flops_per_sec=hw_flops,
            peak_bandwidth_bytes_per_sec=hw_bw,
        )

        model.eval()
        param_dict = dict(model.named_parameters())
        buffer_dict = dict(model.named_buffers())

        # Detect sequence length for FLOPs scaling
        config = getattr(model, 'config', None)
        if config:
            # ViT: num_patches + 1 (cls token)
            image_size = getattr(config, 'image_size', 0)
            patch_size = getattr(config, 'patch_size', 0)
            if image_size and patch_size:
                seq_len = (image_size // patch_size) ** 2 + 1  # +1 for CLS
            else:
                # LLM: infer from config
                n_positions = getattr(config, 'n_positions', 0) or getattr(config, 'max_position_embeddings', 0)
                seq_len = min(n_positions, 1024) if n_positions else 1
        else:
            seq_len = 1

        flop_rules = self._make_flop_rules(seq_len)

        # ── 1. Build param-to-node mapping ──
        # For torch.compile graphs, placeholders hold params with names like
        # "p_m_vit_encoder_layer_0_attention_attention_query_weight".
        # Map these back to actual param names by converting underscores to dots.
        placeholder_param_map = {}  # placeholder_name -> (param_name, param_tensor)
        for gnode in graph_result.nodes:
            if gnode.op == "placeholder" and gnode.name.startswith("p_"):
                # Convert: p_m_vit_encoder_layer_0_... -> vit.encoder.layer.0....
                # Strip "p_m_" or "p_" prefix, then try to match
                stripped = gnode.name
                for prefix in ["p_m_", "p_"]:
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix):]
                        break
                # Try dotted version
                dotted = stripped.replace("_", ".")
                # Find best matching param name
                best_match = None
                best_len = 0
                for pname in param_dict:
                    # Normalize both for comparison
                    pname_norm = pname.replace(".", "_")
                    if pname_norm == stripped or stripped.endswith(pname_norm):
                        if len(pname) > best_len:
                            best_match = pname
                            best_len = len(pname)
                    elif pname == dotted:
                        best_match = pname
                        best_len = len(pname)
                if best_match:
                    placeholder_param_map[gnode.name] = (best_match, param_dict[best_match])

        # Build a map: op_node_name -> list of param shapes consumed
        # by tracing which placeholders each op node uses
        # For module-walk fallback, we match by name substring instead
        is_compile_graph = any(n.name.startswith("p_") and n.op == "placeholder"
                               for n in graph_result.nodes)

        # ── 2. Build ordered module lists by type for index-based matching ──
        # HuggingFace GPT-2 uses Conv1D instead of nn.Linear, so detect it
        try:
            from transformers.pytorch_utils import Conv1D as HFConv1D
        except ImportError:
            HFConv1D = None

        linear_types = [nn.Linear]
        if HFConv1D is not None:
            linear_types.append(HFConv1D)
        linear_types = tuple(linear_types)

        # Ordered lists of modules by op category
        linear_modules = [(name, m) for name, m in model.named_modules()
                          if isinstance(m, linear_types)]
        conv2d_modules = [(name, m) for name, m in model.named_modules()
                          if isinstance(m, nn.Conv2d)]
        ln_modules = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, nn.LayerNorm)]
        embed_modules = [(name, m) for name, m in model.named_modules()
                         if isinstance(m, nn.Embedding)]

        # Counters for graph-op-to-module matching (compile graphs)
        import re as _re
        op_counters = {}  # base_op_name -> next index to assign

        def _get_op_index(node_name):
            """Get the sequential index for this op type."""
            base = _re.sub(r'_\d+$', '', node_name)
            idx_match = _re.search(r'_(\d+)$', node_name)
            if idx_match:
                return base, int(idx_match.group(1))
            # No numeric suffix — use a counter
            if base not in op_counters:
                op_counters[base] = 0
            idx = op_counters[base]
            op_counters[base] += 1
            return base, idx

        # For compile graphs, build a mapping of which graph ops correspond
        # to linear/matmul operations by scanning for embedding/addmm/mm patterns
        # that consume weight placeholders
        linear_op_counter = 0
        conv2d_op_counter = 0
        ln_op_counter = 0
        embed_op_counter = 0

        # ── 3. Per-node profiling from model modules directly ──
        # Graph node names from torch.compile are unreliable (hex pointers,
        # mangled names). Instead, build per-node profiles by walking the
        # model's module tree — this works for all model types.
        node_profiles = []

        for mod_name, mod in model.named_modules():
            # Only leaf modules with parameters
            if len(list(mod.children())) > 0:
                continue
            mod_params = sum(p.numel() for p in mod.parameters())
            if mod_params == 0:
                continue

            mod_type = type(mod).__name__
            shapes = [tuple(p.shape) for p in mod.parameters()]

            # Determine compressibility from graph result if available
            compressible = False
            compress_methods = []
            for gnode in graph_result.nodes:
                if mod_name in gnode.name or gnode.name in mod_name:
                    compressible = gnode.compressible
                    compress_methods = gnode.compress_methods
                    break

            np_ = NodeProfile(
                name=mod_name,
                op_type=mod_type,
                params=mod_params,
                compressible=compressible,
                compress_methods=compress_methods,
            )

            node_shapes = shapes
            # HuggingFace Conv1D is functionally a Linear layer
            is_hf_conv1d = (HFConv1D is not None and isinstance(mod, HFConv1D))
            match_str = mod_type.lower() + " " + mod_name.lower()
            if is_hf_conv1d:
                match_str += " linear"  # treat Conv1D as linear for FLOPs

            # Estimate FLOPs from op type AND module name
            flops = 0
            for rule_key, rule_fn in flop_rules.items():
                if rule_key in match_str:
                    for shape in node_shapes:
                        flops += rule_fn(list(shape))
                    if not node_shapes and mod_params > 0:
                        if any(k in match_str for k in ["linear", "proj", "dense", "fc"]):
                            flops = 2 * mod_params * seq_len
                    break

            if flops == 0 and mod_params > 0:
                # Generic fallback for parameterized ops
                if any(k in match_str for k in ["linear", "conv", "matmul",
                       "proj", "dense", "fc", "mlp", "intermediate", "attention"]):
                    flops = 2 * mod_params * seq_len
            np_.flops = flops

            # Activation memory: output tensor size = out_features * seq_len * 4 bytes
            act_bytes = 0
            if shapes:
                largest = max(shapes, key=lambda s: s[0] if s else 0)
                if len(largest) >= 2:
                    # Linear/Conv1D weight: (out, in) — output is out * seq_len
                    act_bytes = largest[0] * seq_len * 4
                elif len(largest) == 1:
                    act_bytes = largest[0] * seq_len * 4
            np_.activation_memory_bytes = act_bytes

            # Arithmetic intensity = FLOPs / bytes_accessed
            bytes_accessed = mod_params * 4 + act_bytes  # weights + output
            if bytes_accessed > 0 and flops > 0:
                np_.arithmetic_intensity = flops / bytes_accessed
                np_.roofline_bound = ("compute" if np_.arithmetic_intensity >= ridge_point
                                      else "memory")
            elif flops > 0:
                np_.arithmetic_intensity = float('inf')
                np_.roofline_bound = "compute"
            else:
                np_.roofline_bound = "memory"

            node_profiles.append(np_)

        gsp.node_profiles = node_profiles

        # ── 2. Totals ──
        gsp.total_flops = sum(n.flops for n in node_profiles)
        gsp.total_params = sum(p.numel() for p in model.parameters())
        gsp.total_activation_mb = sum(
            n.activation_memory_bytes for n in node_profiles
        ) / (1024 * 1024)

        intensities = [n.arithmetic_intensity for n in node_profiles
                       if n.arithmetic_intensity > 0 and n.arithmetic_intensity != float('inf')]
        gsp.avg_arithmetic_intensity = (
            sum(intensities) / len(intensities) if intensities else 0.0
        )

        ops_with_bound = [n for n in node_profiles if n.roofline_bound]
        if ops_with_bound:
            compute_count = sum(1 for n in ops_with_bound if n.roofline_bound == "compute")
            gsp.pct_compute_bound = compute_count / len(ops_with_bound) * 100
            gsp.pct_memory_bound = 100 - gsp.pct_compute_bound

        # ── 3. KV cache estimation ──
        kv = self._estimate_kv_cache(model, target_context_lengths)
        if kv:
            gsp.kv_cache = kv

        # ── 4. Per-block aggregation ──
        gsp.block_profiles = self._aggregate_blocks(model, node_profiles)

        # ── 5. Compression plan ──
        gsp.compression_plan = self._build_compression_plan(
            gsp, model, ridge_point, target_context_lengths,
        )

        return gsp

    def _build_compression_plan(self, gsp: 'GraphStaticProfile',
                                model: nn.Module, ridge_point: float,
                                context_lengths: List[int],
                                ) -> CompressionPlan:
        """
        Build a compression plan from graph static profile data.

        Components:
        0. Token merging (ViT/VLM): reduce tokens before quantization
        1. Pareto ranking: top-K layers by cost share
        2. Quant assignment: bit-widths from intensity + layer role
        3. Pruning candidates: layers with low cost-to-param ratio
        4. KV quant priority: based on KV share at target context
        """
        plan = CompressionPlan(model_name=gsp.model_name)

        # Filter to nodes that have params or FLOPs
        active = [n for n in gsp.node_profiles if n.flops > 0 or n.params > 0]
        total_flops = max(gsp.total_flops, 1)
        total_params = max(gsp.total_params, 1)

        # ═══ 0. Token merging for ViT/VLM ═══
        # Detect if model has patch-based token sequence (ViT, CLIP vision)
        config = getattr(model, 'config', None)
        seq_len = 0
        image_size = 0
        patch_size = 0

        if config:
            image_size = getattr(config, 'image_size', 0)
            patch_size = getattr(config, 'patch_size', 0)
            # Also check for CLIP's vision config
            vision_config = getattr(config, 'vision_config', None)
            if vision_config and not image_size:
                image_size = getattr(vision_config, 'image_size', 0)
                patch_size = getattr(vision_config, 'patch_size', 0)

        if image_size and patch_size:
            seq_len = (image_size // patch_size) ** 2 + 1  # +1 for CLS
            plan.has_token_merging = True
            plan.token_merging = []

            for keep_ratio in [0.9, 0.7, 0.5, 0.3]:
                merged_tokens = max(2, int(seq_len * keep_ratio))
                r = merged_tokens / seq_len

                # Attention FLOPs scale O(T^2 * D): speedup = (1/r)^2
                attn_speedup = 1.0 / (r ** 2) if r > 0 else 1.0
                # FFN FLOPs scale O(T * D^2): speedup = 1/r
                ffn_speedup = 1.0 / r if r > 0 else 1.0
                # Weighted: attention ~40% of ViT compute, FFN ~60%
                overall_speedup = 1.0 / (0.4 * r**2 + 0.6 * r) if r > 0 else 1.0

                merged_flops = int(total_flops / overall_speedup) if overall_speedup > 0 else total_flops
                # Correct: merged_flops should be smaller
                merged_flops = int(total_flops * (0.4 * r**2 + 0.6 * r))
                flops_reduction = 1.0 - (merged_flops / total_flops) if total_flops > 0 else 0

                # Strategy recommendation
                if keep_ratio >= 0.7:
                    strategy = "bipartite"
                    reason = (f"At {keep_ratio:.0%} keep: bipartite matching preserves "
                              f"semantic similarity with {overall_speedup:.2f}x speedup "
                              f"and minimal accuracy loss (<1% on ImageNet)")
                elif keep_ratio >= 0.5:
                    strategy = "bipartite"
                    reason = (f"At {keep_ratio:.0%} keep: bipartite matching gives "
                              f"{overall_speedup:.2f}x speedup with moderate accuracy tradeoff "
                              f"(~1-3% on ImageNet)")
                else:
                    strategy = "kmeans"
                    reason = (f"At {keep_ratio:.0%} keep: k-means clustering better preserves "
                              f"spatial structure at high compression ({overall_speedup:.2f}x speedup)")

                plan.token_merging.append(TokenMergingEstimate(
                    keep_ratio=keep_ratio,
                    original_tokens=seq_len,
                    merged_tokens=merged_tokens,
                    original_flops=total_flops,
                    merged_flops=merged_flops,
                    flops_reduction=flops_reduction,
                    attn_speedup=attn_speedup,
                    ffn_speedup=ffn_speedup,
                    overall_speedup=overall_speedup,
                    recommended_strategy=strategy,
                    reason=reason,
                ))

        # ═══ 1. Pareto ranking ═══
        ranked = sorted(active, key=lambda n: -n.flops)
        cum_flops = 0.0
        cum_params = 0.0
        cover_80 = 0
        found_80 = False
        for n in ranked:
            fs = n.flops / total_flops
            ps = n.params / total_params if n.params > 0 else 0.0
            cum_flops += fs
            cum_params += ps
            plan.pareto_layers.append(ParetoLayer(
                name=n.name, op_type=n.op_type,
                flops=n.flops, params=n.params,
                flops_share=fs, param_share=ps,
                cumulative_flops_share=cum_flops,
                cumulative_param_share=cum_params,
            ))
            if not found_80 and cum_flops >= 0.8:
                cover_80 = len(plan.pareto_layers)
                found_80 = True
        plan.top_k_cover_80_flops = cover_80 if found_80 else len(plan.pareto_layers)

        # ═══ 2. Quant assignment ═══
        total_weighted_bits = 0
        total_quant_params = 0
        for n in active:
            if n.params == 0:
                continue
            target_lower = n.op_type.lower()
            name_lower = n.name.lower()

            # Determine layer role
            is_attention = any(k in target_lower or k in name_lower
                               for k in ["attn", "attention", "query", "key", "value",
                                          "q_proj", "k_proj", "v_proj"])
            is_embedding = any(k in target_lower or k in name_lower
                               for k in ["embed", "wte", "wpe"])
            is_head = any(k in name_lower for k in ["lm_head", "classifier", "head"])
            is_norm = any(k in target_lower for k in ["norm", "layernorm", "batchnorm"])

            # Assignment logic
            if is_embedding:
                bits = 8
                reason = "embedding: moderate quant, lookup table sensitive"
            elif is_head:
                bits = 8
                reason = "output head: keep higher precision for accuracy"
            elif is_norm:
                bits = 16
                reason = "normalization: keep FP16, small param count"
            elif is_attention and n.arithmetic_intensity < ridge_point * 0.5:
                bits = 4
                reason = f"attention (memory-bound, AI={n.arithmetic_intensity:.1f}): aggressive INT4"
            elif n.arithmetic_intensity < ridge_point * 0.3:
                bits = 4
                reason = f"memory-bound (AI={n.arithmetic_intensity:.1f}): INT4 for bandwidth savings"
            elif n.arithmetic_intensity < ridge_point:
                bits = 8
                reason = f"moderate intensity (AI={n.arithmetic_intensity:.1f}): INT8 balanced"
            else:
                bits = 8
                reason = f"compute-bound (AI={n.arithmetic_intensity:.1f}): INT8 sufficient"

            # Small layers: keep higher precision
            if n.params < 1024 and bits < 16:
                bits = 16
                reason = "tiny layer (<1K params): keep FP16"

            plan.quant_assignments.append(QuantAssignment(
                name=n.name, op_type=n.op_type,
                assigned_bits=bits, reason=reason,
                arithmetic_intensity=n.arithmetic_intensity,
                params=n.params,
            ))
            total_weighted_bits += bits * n.params
            total_quant_params += n.params

        plan.avg_bits = total_weighted_bits / total_quant_params if total_quant_params > 0 else 16.0

        # ═══ 3. Pruning candidates ═══
        # Layers with low FLOPs-per-parameter ratio are good pruning targets
        # (lots of params but not much compute contribution)
        for n in active:
            if n.params < 1000:
                continue
            target_lower = n.op_type.lower()
            is_prunable = any(k in target_lower or k in n.name.lower()
                              for k in ["linear", "conv", "proj", "dense", "fc",
                                         "mlp", "intermediate"])
            if not is_prunable:
                continue

            cost_ratio = n.flops / n.params if n.params > 0 else float('inf')
            param_share = n.params / total_params

            reasons = []
            if cost_ratio < 2.0:
                reasons.append(f"low FLOPs/param ratio ({cost_ratio:.1f})")
            if param_share > 0.05:
                reasons.append(f"large param share ({param_share:.1%})")
            if n.arithmetic_intensity < ridge_point * 0.5:
                reasons.append("memory-bound: pruning reduces bandwidth pressure")

            if reasons:
                plan.pruning_candidates.append(PruningCandidate(
                    name=n.name, op_type=n.op_type,
                    params=n.params, flops=n.flops,
                    cost_to_param_ratio=cost_ratio,
                    param_share=param_share,
                    reason="; ".join(reasons),
                ))

        plan.pruning_candidates.sort(key=lambda c: c.cost_to_param_ratio)
        plan.total_prunable_params = sum(pc.params for pc in plan.pruning_candidates)

        # ═══ 4. KV cache quant priority ═══
        if gsp.kv_cache and gsp.kv_cache.num_layers > 0:
            # Use the largest context length as target
            target_ctx = max(context_lengths) if context_lengths else 4096
            plan.target_context = target_ctx

            kv = gsp.kv_cache
            # Per-layer KV memory
            per_layer_kv_bytes = (2 * kv.num_heads * kv.head_dim *
                                  target_ctx * kv.dtype_bytes)
            per_layer_kv_mb = per_layer_kv_bytes / (1024 * 1024)
            plan.total_kv_mb = per_layer_kv_mb * kv.num_layers

            total_kv_after = 0.0
            for i in range(kv.num_layers):
                layer_name = f"layer_{i}"
                kv_share = 1.0 / kv.num_layers  # uniform across layers

                # Assignment logic:
                # - Early layers (first 20%): keep FP16, they capture positional info
                # - Middle layers (20-80%): INT8, bulk of computation
                # - Late layers (last 20%): INT4, most redundant KV patterns
                layer_frac = i / max(kv.num_layers - 1, 1)

                if layer_frac < 0.2:
                    bits = 16
                    reason = "early layer: preserves positional encoding fidelity"
                    kv_after = per_layer_kv_mb  # no reduction
                elif layer_frac < 0.8:
                    bits = 8
                    reason = "middle layer: INT8 KV cache, good accuracy/memory tradeoff"
                    kv_after = per_layer_kv_mb * 8 / (kv.dtype_bytes * 8)
                else:
                    bits = 4
                    reason = "late layer: INT4 KV cache, high redundancy in late attention"
                    kv_after = per_layer_kv_mb * 4 / (kv.dtype_bytes * 8)

                total_kv_after += kv_after

                plan.kv_priorities.append(KVQuantPriority(
                    layer_name=layer_name,
                    layer_index=i,
                    kv_memory_mb=per_layer_kv_mb,
                    kv_share=kv_share,
                    recommended_bits=bits,
                    reason=reason,
                ))

            plan.kv_after_quant_mb = total_kv_after

        return plan

    def _estimate_kv_cache(self, model: nn.Module,
                           context_lengths: List[int]) -> Optional[KVCacheEstimate]:
        """Estimate KV cache memory for transformer models."""
        # Try to detect transformer config
        num_layers = 0
        num_heads = 0
        head_dim = 0
        hidden_dim = 0

        # Check for HuggingFace config
        config = getattr(model, 'config', None)
        if config:
            num_layers = getattr(config, 'n_layer', 0) or getattr(config, 'num_hidden_layers', 0)
            num_heads = getattr(config, 'n_head', 0) or getattr(config, 'num_attention_heads', 0)
            hidden_dim = getattr(config, 'n_embd', 0) or getattr(config, 'hidden_size', 0)
            head_dim = hidden_dim // num_heads if num_heads > 0 else 0
            # Some models have separate kv heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        else:
            # Heuristic: count attention layers and infer dimensions
            for name, module in model.named_modules():
                mtype = type(module).__name__.lower()
                if any(k in mtype for k in ["attention", "attn"]):
                    num_layers += 1
                    # Try to get head info from child linear layers
                    for cname, child in module.named_modules():
                        if isinstance(child, nn.Linear):
                            if any(k in cname.lower() for k in ["q_proj", "query", "c_attn"]):
                                hidden_dim = child.out_features
                                break
            num_kv_heads = num_heads

        if num_layers == 0 or hidden_dim == 0:
            return None

        if num_heads == 0:
            num_heads = max(1, hidden_dim // 64)  # assume head_dim=64
        if head_dim == 0:
            head_dim = hidden_dim // num_heads
        num_kv_heads = min(num_kv_heads, num_heads) if num_kv_heads > 0 else num_heads

        dtype_bytes = 2  # fp16

        kv = KVCacheEstimate(
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            dtype_bytes=dtype_bytes,
        )

        for ctx_len in context_lengths:
            # KV cache = 2 (K+V) * layers * heads * head_dim * ctx_len * dtype
            mem_bytes = 2 * num_layers * num_kv_heads * head_dim * ctx_len * dtype_bytes
            kv.estimates[ctx_len] = mem_bytes / (1024 * 1024)

        return kv

    def _aggregate_blocks(self, model: nn.Module,
                          node_profiles: List[NodeProfile]) -> List[BlockProfile]:
        """Aggregate node profiles into per-block summaries."""
        from collections import Counter as _Counter

        # Find block structure from model
        block_modules = []
        for name, module in model.named_modules():
            mtype = type(module).__name__
            # Detect repeated blocks
            if any(k in mtype for k in ["Block", "Layer", "EncoderLayer",
                   "DecoderLayer", "ViTLayer", "GPT2Block", "CLIPEncoderLayer"]):
                # Only direct blocks, not nested
                parts = name.split(".")
                try:
                    idx = int(parts[-1])
                    block_modules.append((name, idx, module))
                except (ValueError, IndexError):
                    pass

        if not block_modules:
            return []

        blocks = []
        for bname, bidx, bmodule in block_modules:
            bp = BlockProfile(block_name=bname, block_index=bidx)

            # Find nodes that belong to this block
            block_param_names = {n for n, _ in bmodule.named_parameters()}
            op_counts = _Counter()

            for np_ in node_profiles:
                # Check if node name contains the block path
                if bname in np_.name or any(
                    np_.name.startswith(pn.rsplit(".", 1)[0])
                    for pn in block_param_names if "." in pn
                ):
                    bp.total_flops += np_.flops
                    bp.total_params += np_.params
                    bp.total_activation_mb += np_.activation_memory_bytes / (1024 * 1024)
                    bp.node_count += 1
                    if np_.compressible:
                        bp.compressible_count += 1
                    op_counts[np_.op_type] += 1

            if not op_counts:
                # Fallback: aggregate from module parameters directly
                bp.total_params = sum(p.numel() for p in bmodule.parameters())
                bp.total_flops = 2 * sum(
                    p.numel() for p in bmodule.parameters()
                    if p.ndim >= 2
                )
                bp.node_count = sum(1 for _ in bmodule.modules()
                                    if len(list(_.children())) == 0)

            bp.dominant_op = op_counts.most_common(1)[0][0] if op_counts else ""
            intensities = [n.arithmetic_intensity for n in node_profiles
                           if bname in n.name and n.arithmetic_intensity > 0
                           and n.arithmetic_intensity != float('inf')]
            bp.avg_arithmetic_intensity = (
                sum(intensities) / len(intensities) if intensities else 0.0
            )

            blocks.append(bp)

        return blocks

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
