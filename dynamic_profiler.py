"""
Dynamic Profiler Module
=======================
Profiles model behavior AT RUNTIME with actual inputs.
Measures latency, memory usage, throughput, and per-layer execution costs.

Key metrics:
- Inference latency (mean, std, p50, p95, p99)
- Peak memory usage during forward pass
- FLOPs estimation (via torch.profiler or manual counting)
- Per-layer latency breakdown
- Throughput (samples/second)
- Memory timeline during inference
"""

import torch
import torch.nn as nn
import time
import gc
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np

try:
    from torch.profiler import profile as torch_profile, ProfilerActivity, record_function
    HAS_TORCH_PROFILER = True
except ImportError:
    HAS_TORCH_PROFILER = False

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


@dataclass
class LatencyStats:
    """Latency statistics from multiple runs."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    num_runs: int
    raw_times_ms: List[float] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_memory_mb: float
    allocated_before_mb: float
    allocated_after_mb: float
    reserved_peak_mb: float
    memory_delta_mb: float  # net change during forward pass


@dataclass
class LayerLatency:
    """Per-layer timing information."""
    name: str
    layer_type: str
    total_time_ms: float
    percentage: float  # % of total forward pass time
    call_count: int


@dataclass
class FLOPStats:
    """FLOPs and computational cost estimates."""
    total_flops: int
    total_macs: int  # multiply-accumulate operations
    flops_by_module: Dict[str, int] = field(default_factory=dict)
    method: str = "estimated"  # "fvcore", "profiler", or "estimated"


@dataclass
class DynamicProfile:
    """Complete dynamic profile for a model."""
    model_name: str
    device: str
    input_shape: Tuple[int, ...]
    dtype: str

    latency: Optional[LatencyStats] = None
    memory: Optional[MemoryStats] = None
    flops: Optional[FLOPStats] = None
    throughput_samples_per_sec: float = 0.0
    layer_latencies: List[LayerLatency] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "device": self.device,
            "input_shape": list(self.input_shape),
            "dtype": self.dtype,
        }
        if self.latency:
            result["latency"] = {
                "mean_ms": round(self.latency.mean_ms, 3),
                "std_ms": round(self.latency.std_ms, 3),
                "p50_ms": round(self.latency.p50_ms, 3),
                "p95_ms": round(self.latency.p95_ms, 3),
                "p99_ms": round(self.latency.p99_ms, 3),
            }
        if self.memory:
            result["memory"] = {
                "peak_memory_mb": round(self.memory.peak_memory_mb, 2),
                "memory_delta_mb": round(self.memory.memory_delta_mb, 2),
            }
        if self.flops:
            result["flops"] = {
                "total_flops": self.flops.total_flops,
                "total_macs": self.flops.total_macs,
                "method": self.flops.method,
            }
        result["throughput_samples_per_sec"] = round(self.throughput_samples_per_sec, 2)

        if self.layer_latencies:
            result["top_10_layers_by_time"] = [
                {"name": ll.name, "type": ll.layer_type,
                 "time_ms": round(ll.total_time_ms, 3),
                 "pct": round(ll.percentage, 2)}
                for ll in sorted(self.layer_latencies,
                                 key=lambda x: -x.total_time_ms)[:10]
            ]
        return result

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  DYNAMIC PROFILE: {self.model_name}",
            f"{'='*60}",
            f"  Device:       {self.device}",
            f"  Input Shape:  {self.input_shape}",
            f"  Dtype:        {self.dtype}",
        ]
        if self.latency:
            lines += [
                f"{'─'*60}",
                f"  Latency ({self.latency.num_runs} runs):",
                f"    Mean:  {self.latency.mean_ms:>10.3f} ms",
                f"    Std:   {self.latency.std_ms:>10.3f} ms",
                f"    P50:   {self.latency.p50_ms:>10.3f} ms",
                f"    P95:   {self.latency.p95_ms:>10.3f} ms",
                f"    P99:   {self.latency.p99_ms:>10.3f} ms",
            ]
        if self.memory:
            lines += [
                f"{'─'*60}",
                f"  Memory:",
                f"    Peak Allocated:  {self.memory.peak_memory_mb:>8.2f} MB",
                f"    Delta:           {self.memory.memory_delta_mb:>8.2f} MB",
            ]
        if self.flops:
            gflops = self.flops.total_flops / 1e9
            lines += [
                f"{'─'*60}",
                f"  FLOPs ({self.flops.method}):",
                f"    Total:  {gflops:.3f} GFLOPs",
                f"    MACs:   {self.flops.total_macs / 1e9:.3f} GMACs",
            ]
        lines += [
            f"{'─'*60}",
            f"  Throughput:  {self.throughput_samples_per_sec:.2f} samples/sec",
        ]
        if self.layer_latencies:
            top = sorted(self.layer_latencies, key=lambda x: -x.total_time_ms)[:5]
            lines += [
                f"{'─'*60}",
                f"  Top 5 Layers by Time:",
            ]
            for ll in top:
                lines.append(
                    f"    {ll.name[:40]:40s}  {ll.total_time_ms:>8.3f} ms  ({ll.percentage:.1f}%)"
                )
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class DynamicProfiler:
    """
    Profiles a PyTorch model at runtime with actual inference passes.
    Measures latency, memory, FLOPs, and per-layer costs.
    """

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 50,
                 device: Optional[str] = None):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def profile(self, model: nn.Module, input_tensor: torch.Tensor,
                model_name: str = "unknown",
                forward_fn: Optional[Callable] = None) -> DynamicProfile:
        """
        Run full dynamic profiling on a model.

        Args:
            model: The PyTorch model to profile.
            input_tensor: A sample input tensor (or tuple of tensors).
            model_name: Name for identification.
            forward_fn: Optional custom forward function. If None, uses model(input_tensor).
                        Useful for HuggingFace models that return dicts.
        """
        model = model.to(self.device)
        model.eval()

        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.to(self.device)

        if forward_fn is None:
            forward_fn = lambda m, x: m(x)

        dp = DynamicProfile(
            model_name=model_name,
            device=self.device,
            input_shape=tuple(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else (),
            dtype=str(input_tensor.dtype) if isinstance(input_tensor, torch.Tensor) else "unknown",
        )

        # ── 1. Latency benchmarking ──
        dp.latency = self._benchmark_latency(model, input_tensor, forward_fn)

        # ── 2. Memory profiling ──
        dp.memory = self._profile_memory(model, input_tensor, forward_fn)

        # ── 3. FLOPs estimation ──
        dp.flops = self._estimate_flops(model, input_tensor)

        # ── 4. Per-layer latency breakdown ──
        dp.layer_latencies = self._profile_layer_latencies(model, input_tensor, forward_fn)

        # ── 5. Throughput ──
        if dp.latency and dp.latency.mean_ms > 0:
            batch_size = input_tensor.shape[0] if isinstance(input_tensor, torch.Tensor) else 1
            dp.throughput_samples_per_sec = (batch_size * 1000.0) / dp.latency.mean_ms

        return dp

    def _benchmark_latency(self, model: nn.Module, input_tensor: torch.Tensor,
                           forward_fn: Callable) -> LatencyStats:
        """Measure inference latency with warmup."""
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                forward_fn(model, input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                forward_fn(model, input_tensor)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000.0)  # to ms

        times_arr = np.array(times)
        return LatencyStats(
            mean_ms=float(np.mean(times_arr)),
            std_ms=float(np.std(times_arr)),
            min_ms=float(np.min(times_arr)),
            max_ms=float(np.max(times_arr)),
            p50_ms=float(np.percentile(times_arr, 50)),
            p95_ms=float(np.percentile(times_arr, 95)),
            p99_ms=float(np.percentile(times_arr, 99)),
            num_runs=len(times),
            raw_times_ms=times,
        )

    def _profile_memory(self, model: nn.Module, input_tensor: torch.Tensor,
                        forward_fn: Callable) -> MemoryStats:
        """Profile peak memory usage during forward pass."""
        if self.device != "cuda":
            # CPU memory profiling via tracemalloc
            import tracemalloc
            gc.collect()

            tracemalloc.start()
            with torch.no_grad():
                forward_fn(model, input_tensor)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return MemoryStats(
                peak_memory_mb=peak / (1024 * 1024),
                allocated_before_mb=0,
                allocated_after_mb=current / (1024 * 1024),
                reserved_peak_mb=peak / (1024 * 1024),
                memory_delta_mb=current / (1024 * 1024),
            )
        else:
            # CUDA memory profiling
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            before = torch.cuda.memory_allocated() / (1024 * 1024)

            with torch.no_grad():
                forward_fn(model, input_tensor)

            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved_peak = torch.cuda.max_memory_reserved() / (1024 * 1024)

            return MemoryStats(
                peak_memory_mb=peak,
                allocated_before_mb=before,
                allocated_after_mb=after,
                reserved_peak_mb=reserved_peak,
                memory_delta_mb=after - before,
            )

    def _estimate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> FLOPStats:
        """Estimate FLOPs using fvcore if available, otherwise manual estimation."""
        if HAS_FVCORE:
            try:
                flops_analysis = FlopCountAnalysis(model, input_tensor)
                flops_analysis.unsupported_ops_warnings(False)
                flops_analysis.uncalled_modules_warnings(False)
                total_flops = flops_analysis.total()
                by_module = dict(flops_analysis.by_module())
                return FLOPStats(
                    total_flops=int(total_flops),
                    total_macs=int(total_flops // 2),
                    flops_by_module=by_module,
                    method="fvcore",
                )
            except Exception as e:
                warnings.warn(f"fvcore FLOPs estimation failed: {e}")

        # Manual estimation fallback
        total_flops = self._manual_flop_estimate(model, input_tensor)
        return FLOPStats(
            total_flops=total_flops,
            total_macs=total_flops // 2,
            method="estimated",
        )

    def _manual_flop_estimate(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """
        Rough FLOPs estimation by counting operations per layer type.
        Not as accurate as fvcore but works for any model.
        """
        total = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 2 * in_features * out_features (multiply + add)
                total += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                # 2 * Cout * Cin * Kh * Kw * Hout * Wout / groups
                # Rough: assume output spatial = input spatial / stride
                k = module.kernel_size[0] * module.kernel_size[1]
                out_channels = module.out_channels
                in_channels = module.in_channels // module.groups
                # Estimate output spatial dims
                if isinstance(input_tensor, torch.Tensor) and input_tensor.ndim == 4:
                    h, w = input_tensor.shape[2], input_tensor.shape[3]
                    stride = module.stride[0]
                    h_out = h // stride
                    w_out = w // stride
                else:
                    h_out, w_out = 1, 1
                total += 2 * out_channels * in_channels * k * h_out * w_out
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'normalized_shape'):
                    total += 5 * np.prod(module.normalized_shape)  # mean, var, normalize, scale, bias
            elif isinstance(module, nn.BatchNorm2d):
                total += 4 * module.num_features  # per spatial position

        # Scale by batch size
        batch_size = input_tensor.shape[0] if isinstance(input_tensor, torch.Tensor) else 1
        total *= batch_size
        return int(total)

    def _profile_layer_latencies(self, model: nn.Module, input_tensor: torch.Tensor,
                                 forward_fn: Callable) -> List[LayerLatency]:
        """Profile per-layer execution times using hooks."""
        timings: Dict[str, List[float]] = {}
        handles = []

        def make_hooks(name: str):
            start_times = {}

            def pre_hook(module, inp):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start_times[name] = time.perf_counter()

            def post_hook(module, inp, out):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start_times.get(name, time.perf_counter())) * 1000
                if name not in timings:
                    timings[name] = []
                timings[name].append(elapsed)

            return pre_hook, post_hook

        # Register hooks on all non-container modules
        module_types = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                pre, post = make_hooks(name)
                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post))
                module_types[name] = type(module).__name__

        # Run a few passes to get stable timings
        n_passes = min(10, self.benchmark_runs)
        with torch.no_grad():
            for _ in range(n_passes):
                forward_fn(model, input_tensor)

        # Remove hooks
        for h in handles:
            h.remove()

        # Compute results
        results = []
        total_time = sum(np.mean(t) for t in timings.values()) if timings else 1.0

        for name, times in timings.items():
            mean_time = np.mean(times)
            results.append(LayerLatency(
                name=name,
                layer_type=module_types.get(name, "unknown"),
                total_time_ms=float(mean_time),
                percentage=float(mean_time / total_time * 100) if total_time > 0 else 0,
                call_count=len(times),
            ))

        return results

    def profile_batch_scaling(self, model: nn.Module,
                              input_fn: Callable[[int], torch.Tensor],
                              batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
                              model_name: str = "unknown") -> Dict[int, LatencyStats]:
        """
        Profile how latency scales with batch size.
        Useful for understanding throughput characteristics.

        Args:
            input_fn: Function that takes batch_size and returns an input tensor.
        """
        results = {}
        model = model.to(self.device).eval()

        for bs in batch_sizes:
            try:
                inp = input_fn(bs).to(self.device)
                stats = self._benchmark_latency(model, inp, lambda m, x: m(x))
                results[bs] = stats
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch_size={bs}, stopping.")
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

        return results
