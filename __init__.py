"""
Model Profiler Framework
========================
A framework for profiling neural network models and recommending
optimal compression methods based on architecture analysis.
"""

from .static_profiler import StaticProfiler, StaticProfile

__all__ = [
    "StaticProfiler",
    "StaticProfile",
]
