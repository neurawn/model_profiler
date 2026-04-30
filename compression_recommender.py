"""
Compression Recommender Module
===============================
Uses static and dynamic profiling results to recommend the optimal
compression method for a given model and target hardware.

Methods considered:
- Quantization (INT8, INT4, mixed-precision)
- Structured Pruning (channel/filter pruning)
- Unstructured Pruning (weight-level sparsity)
- Knowledge Distillation
- Low-Rank Factorization (LoRA-style decomposition)
- Token Merging (bipartite matching, k-means, average pooling)

Key hypothesis from proposal:
- Transformers with uniform parameter distribution → Quantization
- CNNs with uneven distribution (doubling channels) → Structured Pruning
- ViT (hybrid) → framework should automatically determine the best method
- Blackbox models → rely on observable features to infer architecture class
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from static_profiler import StaticProfile
from dynamic_profiler import DynamicProfile


class CompressionMethod(Enum):
    QUANTIZATION_INT8 = "quantization_int8"
    QUANTIZATION_INT4 = "quantization_int4"
    MIXED_PRECISION = "mixed_precision"
    STRUCTURED_PRUNING = "structured_pruning"
    UNSTRUCTURED_PRUNING = "unstructured_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    LOW_RANK_FACTORIZATION = "low_rank_factorization"
    TOKEN_MERGING = "token_merging"


@dataclass
class CompressionRecommendation:
    """A single compression method recommendation with rationale."""
    method: CompressionMethod
    score: float  # 0-1 confidence/suitability score
    estimated_speedup: float  # rough multiplier (e.g., 2.0 = 2x faster)
    estimated_size_reduction: float  # fraction (e.g., 0.5 = 50% smaller)
    rationale: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def __repr__(self):
        return (f"  {self.method.value:30s}  score={self.score:.3f}  "
                f"speedup~{self.estimated_speedup:.1f}x  "
                f"size_red~{self.estimated_size_reduction:.0%}")


@dataclass
class RecommendationReport:
    """Full recommendation report for a model."""
    model_name: str
    inferred_architecture_class: str  # "transformer", "cnn", "hybrid", "unknown"
    recommendations: List[CompressionRecommendation] = field(default_factory=list)
    top_recommendation: Optional[CompressionRecommendation] = None
    feature_vector: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  COMPRESSION RECOMMENDATIONS: {self.model_name}",
            f"{'='*70}",
            f"  Inferred Architecture Class: {self.inferred_architecture_class}",
            f"{'─'*70}",
            f"  Feature Vector (inputs to decision):",
        ]
        for k, v in sorted(self.feature_vector.items()):
            lines.append(f"    {k:35s} = {v:.4f}")

        lines += [
            f"{'─'*70}",
            f"  Rankings (best → worst):",
        ]
        for i, rec in enumerate(self.recommendations):
            marker = " ★" if i == 0 else "  "
            lines.append(f"  {marker} {i+1}. {rec}")
            if rec.rationale:
                for r in rec.rationale:
                    lines.append(f"        → {r}")
            if rec.caveats:
                for c in rec.caveats:
                    lines.append(f"        ⚠ {c}")
            lines.append("")

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "inferred_architecture_class": self.inferred_architecture_class,
            "feature_vector": {k: round(v, 4) for k, v in self.feature_vector.items()},
            "recommendations": [
                {
                    "method": r.method.value,
                    "score": round(r.score, 4),
                    "estimated_speedup": round(r.estimated_speedup, 2),
                    "estimated_size_reduction": round(r.estimated_size_reduction, 3),
                    "rationale": r.rationale,
                    "caveats": r.caveats,
                }
                for r in self.recommendations
            ],
        }


class CompressionRecommender:
    """
    Recommends compression methods based on static and dynamic profile data.

    The recommender works in three stages:
    1. Feature extraction: Combines static + dynamic profiles into a feature vector
    2. Architecture classification: Infers model family from features
    3. Scoring: Scores each compression method based on features + architecture class

    This is designed to work even for "blackbox" models where only the
    nn.Module graph is available (no knowledge of the original architecture).
    """

    # Weights for each feature in scoring different methods
    # These encode the proposal's hypotheses about which methods work best
    METHOD_FEATURE_WEIGHTS = {
        CompressionMethod.QUANTIZATION_INT8: {
            "param_uniformity": 0.25,
            "has_attention": 0.15,
            "low_kurtosis": 0.15,
            "model_size_mb": 0.10,
            "is_transformer": 0.20,
            "not_tiny": 0.10,
            "low_sparsity": 0.05,
        },
        CompressionMethod.QUANTIZATION_INT4: {
            "param_uniformity": 0.20,
            "has_attention": 0.15,
            "low_kurtosis": 0.10,
            "model_size_mb": 0.15,
            "is_transformer": 0.20,
            "not_tiny": 0.15,
            "low_sparsity": 0.05,
        },
        CompressionMethod.MIXED_PRECISION: {
            "param_uniformity": 0.15,
            "has_attention": 0.10,
            "model_size_mb": 0.15,
            "depth": 0.15,
            "has_varied_layer_types": 0.20,
            "not_tiny": 0.15,
            "has_outlier_layers": 0.10,
        },
        CompressionMethod.STRUCTURED_PRUNING: {
            "param_non_uniformity": 0.25,
            "has_convolutions": 0.25,
            "is_cnn": 0.20,
            "existing_sparsity": 0.10,
            "depth": 0.10,
            "has_redundant_channels": 0.10,
        },
        CompressionMethod.UNSTRUCTURED_PRUNING: {
            "existing_sparsity": 0.25,
            "param_non_uniformity": 0.15,
            "high_kurtosis": 0.20,
            "model_size_mb": 0.10,
            "depth": 0.10,
            "has_large_linear_layers": 0.20,
        },
        CompressionMethod.KNOWLEDGE_DISTILLATION: {
            "model_size_mb": 0.25,
            "depth": 0.20,
            "not_tiny": 0.20,
            "has_clear_block_structure": 0.20,
            "is_overparameterized": 0.15,
        },
        CompressionMethod.LOW_RANK_FACTORIZATION: {
            "has_low_rank_layers": 0.30,
            "has_large_linear_layers": 0.25,
            "param_uniformity": 0.15,
            "has_attention": 0.15,
            "model_size_mb": 0.15,
        },
        CompressionMethod.TOKEN_MERGING: {
            "has_attention": 0.30,
            "is_transformer": 0.20,
            "is_hybrid": 0.15,
            "depth": 0.10,
            "not_tiny": 0.10,
            "param_uniformity": 0.10,
            "model_size_mb": 0.05,
        },
    }

    def recommend(self, static_profile: StaticProfile,
                  dynamic_profile: Optional[DynamicProfile] = None,
                  target_device: str = "gpu",
                  target_speedup: float = 2.0,
                  target_size_reduction: float = 0.5) -> RecommendationReport:
        """
        Generate compression recommendations.

        Args:
            static_profile: Static analysis results.
            dynamic_profile: Optional runtime profiling results.
            target_device: "gpu", "cpu", "mobile", "edge"
            target_speedup: Desired speedup multiplier.
            target_size_reduction: Desired model size reduction fraction.
        """
        # ── 1. Extract feature vector ──
        features = self._extract_features(static_profile, dynamic_profile)

        # ── 2. Classify architecture ──
        arch_class = self._classify_architecture(features, static_profile)

        # ── 3. Score each method ──
        report = RecommendationReport(
            model_name=static_profile.model_name,
            inferred_architecture_class=arch_class,
            feature_vector=features,
        )

        for method in CompressionMethod:
            rec = self._score_method(method, features, arch_class,
                                     target_device, target_speedup,
                                     target_size_reduction, static_profile)
            report.recommendations.append(rec)

        # Sort by score (descending)
        report.recommendations.sort(key=lambda r: -r.score)
        report.top_recommendation = report.recommendations[0] if report.recommendations else None

        return report

    def _extract_features(self, sp: StaticProfile,
                          dp: Optional[DynamicProfile] = None) -> Dict[str, float]:
        """Extract a normalized feature vector from profiles."""
        f = {}

        # From static profile
        f["param_uniformity"] = sp.param_uniformity_score
        f["param_non_uniformity"] = 1 - sp.param_uniformity_score
        f["has_attention"] = 1.0 if sp.has_attention else 0.0
        f["has_convolutions"] = 1.0 if sp.has_convolutions else 0.0
        f["overall_sparsity"] = sp.overall_sparsity
        f["existing_sparsity"] = min(sp.overall_sparsity * 10, 1.0)  # amplified
        f["model_size_mb"] = min(sp.total_size_mb / 1000.0, 1.0)  # normalized to 1GB
        f["not_tiny"] = min(sp.total_params / 1e7, 1.0)  # >10M params = 1.0
        f["depth"] = min(sp.depth / 50.0, 1.0)
        f["weight_entropy"] = min(sp.weight_entropy / 5.0, 1.0)

        # Derived features
        f["is_transformer"] = 1.0 if (sp.has_attention and not sp.has_convolutions) else 0.0
        f["is_cnn"] = 1.0 if (sp.has_convolutions and not sp.has_attention) else 0.0
        f["is_hybrid"] = 1.0 if (sp.has_attention and sp.has_convolutions) else 0.0

        # Kurtosis-related features
        if sp.layer_profiles:
            kurtosis_values = [lp.kurtosis for lp in sp.layer_profiles]
            avg_kurtosis = np.mean(kurtosis_values)
            f["low_kurtosis"] = max(0, 1 - abs(avg_kurtosis) / 5.0)
            f["high_kurtosis"] = min(abs(avg_kurtosis) / 5.0, 1.0)
        else:
            f["low_kurtosis"] = 0.5
            f["high_kurtosis"] = 0.5

        # Rank-based features (for low-rank factorization)
        rank_ratios = []
        for lp in sp.layer_profiles:
            if lp.rank_estimate is not None and len(lp.shape) == 2:
                max_rank = min(lp.shape)
                if max_rank > 0:
                    rank_ratios.append(lp.rank_estimate / max_rank)

        f["has_low_rank_layers"] = 1 - np.mean(rank_ratios) if rank_ratios else 0.3

        # Large linear layer detection
        large_linear_params = sum(
            lp.param_count for lp in sp.layer_profiles
            if lp.layer_type == "linear" and lp.param_count > 1e6
        )
        f["has_large_linear_layers"] = min(large_linear_params / sp.total_params, 1.0) \
            if sp.total_params > 0 else 0.0

        # Layer type variety
        n_types = len(sp.layer_type_counts)
        f["has_varied_layer_types"] = min(n_types / 6.0, 1.0)

        # Check for outlier layers (very different param counts)
        if sp.layer_profiles:
            param_counts = [lp.param_count for lp in sp.layer_profiles if lp.param_count > 0]
            if len(param_counts) > 1:
                cv = np.std(param_counts) / np.mean(param_counts)
                f["has_outlier_layers"] = min(cv / 3.0, 1.0)
            else:
                f["has_outlier_layers"] = 0.0
        else:
            f["has_outlier_layers"] = 0.0

        # Block structure
        f["has_clear_block_structure"] = min(sp.depth / 10.0, 1.0) if sp.depth > 2 else 0.2

        # Redundant channels (CNN-specific)
        conv_layers = [lp for lp in sp.layer_profiles if lp.layer_type == "conv"]
        if conv_layers:
            avg_conv_sparsity = np.mean([lp.sparsity for lp in conv_layers])
            f["has_redundant_channels"] = min(avg_conv_sparsity * 10, 1.0) + 0.3
            f["has_redundant_channels"] = min(f["has_redundant_channels"], 1.0)
        else:
            f["has_redundant_channels"] = 0.0

        # Overparameterization estimate
        f["is_overparameterized"] = min(sp.total_params / 1e8, 1.0)

        # Dynamic features (if available)
        if dp and dp.latency:
            f["latency_ms"] = min(dp.latency.mean_ms / 100.0, 1.0)
        if dp and dp.memory:
            f["peak_memory_mb"] = min(dp.memory.peak_memory_mb / 1000.0, 1.0)
        if dp and dp.flops:
            f["gflops"] = min(dp.flops.total_flops / 1e10, 1.0)

        return f

    def _classify_architecture(self, features: Dict[str, float],
                               sp: StaticProfile) -> str:
        """
        Classify the model into an architecture family.
        This is especially important for blackbox models where we don't know
        the original architecture.
        """
        scores = {
            "transformer": 0.0,
            "cnn": 0.0,
            "hybrid_vit": 0.0,
            "rnn": 0.0,
            "unknown": 0.1,  # small baseline
        }

        # Transformer signals
        if features["has_attention"] > 0:
            scores["transformer"] += 0.4
        if features["param_uniformity"] > 0.7:
            scores["transformer"] += 0.2
        if sp.dominant_layer_type == "linear":
            scores["transformer"] += 0.15
        if "embedding" in sp.layer_type_counts:
            scores["transformer"] += 0.1

        # CNN signals
        if features["has_convolutions"] > 0 and features["has_attention"] == 0:
            scores["cnn"] += 0.5
        if features["param_non_uniformity"] > 0.5:
            scores["cnn"] += 0.15
        if sp.dominant_layer_type == "conv":
            scores["cnn"] += 0.2

        # ViT / Hybrid signals (has both conv and attention)
        if features["has_convolutions"] > 0 and features["has_attention"] > 0:
            scores["hybrid_vit"] += 0.5
            scores["cnn"] -= 0.2
            scores["transformer"] -= 0.2

        # RNN signals
        if "recurrent" in sp.layer_type_counts:
            rnn_params = sp.layer_type_params.get("recurrent", 0)
            if rnn_params > 0.3 * sp.total_params:
                scores["rnn"] += 0.6

        return max(scores, key=scores.get)

    def _score_method(self, method: CompressionMethod,
                      features: Dict[str, float],
                      arch_class: str,
                      target_device: str,
                      target_speedup: float,
                      target_size_reduction: float,
                      sp: StaticProfile) -> CompressionRecommendation:
        """Score a compression method and generate rationale."""
        weights = self.METHOD_FEATURE_WEIGHTS[method]
        score = 0.0
        rationale = []
        caveats = []

        # Weighted feature scoring
        for feat_name, weight in weights.items():
            feat_val = features.get(feat_name, 0.0)
            score += weight * feat_val

        # ── Architecture-specific adjustments ──
        if method == CompressionMethod.QUANTIZATION_INT8:
            if arch_class == "transformer":
                score += 0.15
                rationale.append("Transformer's uniform parameter distribution is ideal for INT8 quantization")
            if arch_class == "cnn":
                score -= 0.05
                rationale.append("CNNs can be quantized but structured pruning often yields better results")
            estimated_speedup = 1.5 if target_device == "gpu" else 2.5
            estimated_size = 0.75

        elif method == CompressionMethod.QUANTIZATION_INT4:
            if arch_class == "transformer":
                score += 0.12
                rationale.append("INT4 gives aggressive 4x compression for transformer weights")
            estimated_speedup = 2.0 if target_device == "gpu" else 3.0
            estimated_size = 0.875
            caveats.append("May cause noticeable accuracy degradation")

        elif method == CompressionMethod.MIXED_PRECISION:
            if features.get("has_outlier_layers", 0) > 0.5:
                score += 0.1
                rationale.append("Model has outlier layers that benefit from per-layer precision tuning")
            estimated_speedup = 1.3
            estimated_size = 0.5

        elif method == CompressionMethod.STRUCTURED_PRUNING:
            if arch_class == "cnn":
                score += 0.2
                rationale.append("CNN's uneven channel distribution makes structured pruning highly effective")
                rationale.append("Doubling channels at each stage creates natural redundancy")
            if arch_class == "hybrid_vit":
                score += 0.1
                rationale.append("Conv stem layers in ViT can benefit from channel pruning")
            estimated_speedup = 1.5
            estimated_size = 0.4

        elif method == CompressionMethod.UNSTRUCTURED_PRUNING:
            if features.get("existing_sparsity", 0) > 0.3:
                rationale.append(f"Model already has {sp.overall_sparsity:.1%} near-zero weights")
            estimated_speedup = 1.2  # needs sparse hardware for real speedup
            estimated_size = 0.5
            caveats.append("Requires sparse tensor support for actual inference speedup")

        elif method == CompressionMethod.KNOWLEDGE_DISTILLATION:
            if sp.total_params > 5e7:
                score += 0.1
                rationale.append("Large model has significant room for distillation")
            estimated_speedup = target_speedup  # depends on student
            estimated_size = target_size_reduction
            caveats.append("Requires training a student model and access to training data")

        elif method == CompressionMethod.LOW_RANK_FACTORIZATION:
            if features.get("has_low_rank_layers", 0) > 0.5:
                rationale.append("Weight matrices show low effective rank — decomposition viable")
            if features.get("has_large_linear_layers", 0) > 0.3:
                rationale.append("Large linear layers have high potential for rank reduction")
            estimated_speedup = 1.4
            estimated_size = 0.3

        elif method == CompressionMethod.TOKEN_MERGING:
            if arch_class == "transformer":
                score += 0.20
                rationale.append("Token merging is highly effective for pure transformer architectures")
                rationale.append("Reduces sequence length without retraining — bipartite matching preserves quality")
            if arch_class == "hybrid_vit":
                score += 0.25
                rationale.append("ViT is the ideal target for token merging (ToMe was designed for ViT)")
                rationale.append("Patch tokens have high spatial redundancy suitable for merging")
            if arch_class == "cnn":
                score -= 0.3
                caveats.append("Token merging is not applicable to CNN architectures (no token sequence)")
            if arch_class == "rnn":
                score -= 0.2
                caveats.append("Token merging is not directly applicable to RNNs")
            if features.get("depth", 0) > 0.1:
                rationale.append(f"Deeper model benefits from progressive token reduction across layers")
            estimated_speedup = 2.0  # ToMe can achieve ~2x speedup with minimal accuracy loss
            estimated_size = 0.0  # No model size reduction — only compute reduction
            caveats.append("Reduces compute/latency but does NOT reduce model file size")

        else:
            estimated_speedup = 1.0
            estimated_size = 0.0

        # ── Device-specific adjustments ──
        if target_device == "mobile":
            if method in (CompressionMethod.QUANTIZATION_INT8, CompressionMethod.STRUCTURED_PRUNING):
                score += 0.05
                rationale.append(f"Well-suited for mobile deployment")
            if method == CompressionMethod.UNSTRUCTURED_PRUNING:
                score -= 0.1
                caveats.append("Mobile hardware rarely supports sparse operations efficiently")

        elif target_device == "edge":
            if method == CompressionMethod.QUANTIZATION_INT4:
                score += 0.05
            if method == CompressionMethod.KNOWLEDGE_DISTILLATION:
                score += 0.05

        # Clip score
        score = np.clip(score, 0, 1)

        if not rationale:
            rationale.append(f"Baseline score from feature matching: {score:.3f}")

        return CompressionRecommendation(
            method=method,
            score=float(score),
            estimated_speedup=estimated_speedup,
            estimated_size_reduction=estimated_size,
            rationale=rationale,
            caveats=caveats,
        )


def recommend_for_model(static_profile: StaticProfile,
                        dynamic_profile: Optional[DynamicProfile] = None,
                        target_device: str = "gpu",
                        verbose: bool = True) -> RecommendationReport:
    """Convenience function to get compression recommendations."""
    recommender = CompressionRecommender()
    report = recommender.recommend(static_profile, dynamic_profile,
                                   target_device=target_device)
    if verbose:
        print(report.summary())
    return report
