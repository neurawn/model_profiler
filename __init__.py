"""
Model Profiler Framework
========================
A framework for profiling neural network models and recommending
optimal compression methods based on architecture analysis.
"""

from .static_profiler import StaticProfiler, StaticProfile
from .dynamic_profiler import DynamicProfiler, DynamicProfile
from .compression_recommender import CompressionRecommender, recommend_for_model

__all__ = [
    "StaticProfiler",
    "StaticProfile",
    "DynamicProfiler",
    "DynamicProfile",
    "CompressionRecommender",
    "recommend_for_model",
]
