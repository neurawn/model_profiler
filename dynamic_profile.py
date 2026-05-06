"""
Dynamic Profile Module
======================
Uses torch.profiler to capture detailed runtime information:
- Per-operator latency (CPU and CUDA)
- Memory allocation/deallocation per op
- Tensor shapes at each operator
- Call stack traces
- Peak memory usage

Outputs a readable summary to console and optionally saves
a Chrome trace JSON for visualization in chrome://tracing.
"""

import torch
import torch.nn as nn
import time
import os
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable


@dataclass
class OpProfile:
    """Profile for a single operator."""
    name: str
    input_shapes: List[str]
    cpu_time_us: float
    cuda_time_us: float
    self_cpu_time_us: float  # excludes time in nested ops — matches profiler "Self" cols
    self_cuda_time_us: float
    cpu_memory_usage: int  # bytes allocated
    cuda_memory_usage: int
    call_stack: str
    flops: int


@dataclass
class DynamicProfileResult:
    """Complete dynamic profile result."""
    model_name: str
    device: str
    num_runs: int

    # Top-level metrics
    total_cpu_time_ms: float = 0.0
    total_cuda_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    total_params: int = 0
    total_flops: int = 0

    # Per-op breakdown
    op_profiles: List[OpProfile] = field(default_factory=list)

    # Aggregated by op type
    op_type_summary: Dict[str, Dict] = field(default_factory=dict)

    # Trace file path
    trace_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*80}",
            f"  DYNAMIC PROFILE: {self.model_name}",
            f"{'='*80}",
            f"  Device:            {self.device}",
            f"  Profiling runs:    {self.num_runs}",
            f"  Total CPU time:    {self.total_cpu_time_ms:.3f} ms",
        ]
        if self.total_cuda_time_ms > 0:
            lines.append(f"  Total CUDA time:   {self.total_cuda_time_ms:.3f} ms")
        lines += [
            f"  Peak memory:       {self.peak_memory_mb:.2f} MB",
            f"  Total FLOPs:       {self.total_flops / 1e9:.3f} GFLOPs",
        ]

        # Op type summary
        if self.op_type_summary:
            lines += [
                f"{'─'*80}",
                f"  Op Type Summary (aggregated):",
                f"  {'Op Type':<35s} {'Calls':>6s} {'CPU ms':>10s} "
                f"{'CUDA ms':>10s} {'Mem MB':>9s}",
                f"  {'─'*35} {'─'*6} {'─'*10} {'─'*10} {'─'*9}",
            ]
            sorted_ops = sorted(self.op_type_summary.items(),
                                key=lambda x: -x[1].get("cpu_time_ms", 0))
            for op_name, stats in sorted_ops[:25]:
                cuda_str = f"{stats.get('cuda_time_ms', 0):.3f}" if stats.get('cuda_time_ms', 0) > 0 else "─"
                lines.append(
                    f"  {op_name[:35]:<35s} {stats['count']:>6d} "
                    f"{stats['cpu_time_ms']:>10.3f} {cuda_str:>10s} "
                    f"{stats.get('memory_mb', 0):>9.2f}"
                )
            if len(self.op_type_summary) > 25:
                lines.append(f"  ... ({len(self.op_type_summary) - 25} more)")

        # Top ops by latency
        if self.op_profiles:
            top_by_cpu = sorted(self.op_profiles, key=lambda x: -x.cpu_time_us)[:15]
            lines += [
                f"{'─'*80}",
                f"  Top 15 Ops by CPU Time:",
                f"  {'Op':<40s} {'CPU us':>10s} {'CUDA us':>10s} {'Shapes'}",
                f"  {'─'*40} {'─'*10} {'─'*10} {'─'*30}",
            ]
            for op in top_by_cpu:
                cuda_str = f"{op.cuda_time_us:.0f}" if op.cuda_time_us > 0 else "─"
                shapes_str = ", ".join(op.input_shapes[:3])
                if len(op.input_shapes) > 3:
                    shapes_str += f" +{len(op.input_shapes)-3}"
                lines.append(
                    f"  {op.name[:40]:<40s} {op.cpu_time_us:>10.0f} "
                    f"{cuda_str:>10s} {shapes_str}"
                )

        # Top ops by memory
        mem_ops = [op for op in self.op_profiles if op.cpu_memory_usage > 0 or op.cuda_memory_usage > 0]
        if mem_ops:
            top_by_mem = sorted(mem_ops, key=lambda x: -(x.cpu_memory_usage + x.cuda_memory_usage))[:10]
            lines += [
                f"{'─'*80}",
                f"  Top 10 Ops by Memory Allocation:",
                f"  {'Op':<40s} {'CPU MB':>10s} {'CUDA MB':>10s} {'Shapes'}",
                f"  {'─'*40} {'─'*10} {'─'*10} {'─'*30}",
            ]
            for op in top_by_mem:
                cpu_mb = op.cpu_memory_usage / (1024 * 1024)
                cuda_mb = op.cuda_memory_usage / (1024 * 1024)
                shapes_str = ", ".join(op.input_shapes[:3])
                lines.append(
                    f"  {op.name[:40]:<40s} {cpu_mb:>10.2f} "
                    f"{cuda_mb:>10.2f} {shapes_str}"
                )

        # Sample call stacks
        ops_with_stacks = [op for op in self.op_profiles if op.call_stack]
        if ops_with_stacks:
            lines += [
                f"{'─'*80}",
                f"  Sample Call Stacks (top 5 by CPU time):",
            ]
            for op in sorted(ops_with_stacks, key=lambda x: -x.cpu_time_us)[:5]:
                lines.append(f"\n  {op.name} ({op.cpu_time_us:.0f} us):")
                # Show last 5 frames of the call stack
                stack_lines = op.call_stack.strip().split("\n")
                for sl in stack_lines[-5:]:
                    lines.append(f"    {sl.strip()}")

        if self.trace_path:
            lines += [
                f"{'─'*80}",
                f"  Chrome trace saved: {self.trace_path}",
                f"  View at: chrome://tracing",
            ]

        lines.append(f"{'='*80}\n")
        return "\n".join(lines)


def run_dynamic_profile(model: nn.Module, sample_input: torch.Tensor,
                        model_name: str = "model",
                        forward_fn: Optional[Callable] = None,
                        device: Optional[str] = None,
                        num_warmup: int = 3,
                        num_runs: int = 5,
                        save_trace: bool = False,
                        output_dir: str = "./profiling_results",
                        ) -> DynamicProfileResult:
    """
    Run dynamic profiling using torch.profiler.

    Args:
        model: PyTorch model to profile.
        sample_input: Input tensor for forward pass.
        model_name: Name for display.
        forward_fn: Custom forward function (for HuggingFace models).
        device: "cpu" or "cuda". Auto-detected if None.
        num_warmup: Warmup iterations before profiling.
        num_runs: Number of profiled iterations.
        save_trace: Save Chrome trace JSON for visualization.
        output_dir: Directory for trace output.
    """
    from torch.profiler import profile, ProfilerActivity, record_function

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#'*80}")
    print(f"#  DYNAMIC PROFILING: {model_name}")
    print(f"#  Device: {device}, Warmup: {num_warmup}, Runs: {num_runs}")
    print(f"{'#'*80}")

    model = model.to(device).eval()
    if isinstance(sample_input, torch.Tensor):
        sample_input = sample_input.to(device)

    fwd = forward_fn if forward_fn else lambda m, x: m(x)

    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            fwd(model, sample_input)
    if device == "cuda":
        torch.cuda.synchronize()

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    gc.collect()

    # Profile
    print(f"  Profiling ({num_runs} runs)...")
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_path = None
    if save_trace:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        trace_path = os.path.join(output_dir, f"{safe_name}_trace.json")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                fwd(model, sample_input)
                if device == "cuda":
                    torch.cuda.synchronize()

    # Save trace
    if trace_path:
        prof.export_chrome_trace(trace_path)

    # Build result
    result = DynamicProfileResult(
        model_name=model_name,
        device=device,
        num_runs=num_runs,
        total_params=sum(p.numel() for p in model.parameters()),
        trace_path=trace_path,
    )

    # Peak memory
    if device == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # Use profiler memory data
        result.peak_memory_mb = 0

    # Parse profiler events
    op_profiles = []
    op_type_agg = {}  # op_name -> {count, cpu_time, cuda_time, memory}

    # group_by_stack_n=0 → one row per op kind (matches torch.profiler's
    # table view, so Self CUDA % lines up with addmm = 50.71% etc.).
    key_averages = prof.key_averages(group_by_stack_n=0)

    total_cpu = 0.0
    total_cuda = 0.0
    total_flops = 0

    for evt in key_averages:
        cpu_time = evt.cpu_time_total  # microseconds, includes child ops
        self_cpu_time = getattr(evt, 'self_cpu_time_total', cpu_time)
        # PyTorch 2.4+ renamed cuda_time_total → device_time_total. Try the new
        # name first, fall back to the legacy attribute for older builds.
        if hasattr(evt, 'device_time_total'):
            cuda_time = evt.device_time_total
            self_cuda_time = getattr(evt, 'self_device_time_total', cuda_time)
        elif hasattr(evt, 'cuda_time_total'):
            cuda_time = evt.cuda_time_total
            self_cuda_time = getattr(evt, 'self_cuda_time_total', cuda_time)
        else:
            cuda_time = 0
            self_cuda_time = 0
        cpu_mem = evt.cpu_memory_usage if hasattr(evt, 'cpu_memory_usage') else 0
        cuda_mem = evt.cuda_memory_usage if hasattr(evt, 'cuda_memory_usage') else 0
        flops = evt.flops if hasattr(evt, 'flops') and evt.flops else 0

        # Get input shapes
        shapes = []
        if hasattr(evt, 'input_shapes') and evt.input_shapes:
            for s in evt.input_shapes:
                if s:
                    shapes.append(str(s))

        # Get call stack
        stack = ""
        if hasattr(evt, 'stack') and evt.stack:
            stack = "\n".join(evt.stack)

        op = OpProfile(
            name=evt.key,
            input_shapes=shapes,
            cpu_time_us=cpu_time,
            cuda_time_us=cuda_time,
            self_cpu_time_us=self_cpu_time,
            self_cuda_time_us=self_cuda_time,
            cpu_memory_usage=cpu_mem,
            cuda_memory_usage=cuda_mem,
            call_stack=stack,
            flops=flops,
        )
        op_profiles.append(op)

        total_cpu += cpu_time
        total_cuda += cuda_time
        total_flops += flops

        # Aggregate by op type
        op_name = evt.key
        if op_name not in op_type_agg:
            op_type_agg[op_name] = {
                "count": 0, "cpu_time_ms": 0, "cuda_time_ms": 0,
                "memory_mb": 0,
            }
        op_type_agg[op_name]["count"] += evt.count
        op_type_agg[op_name]["cpu_time_ms"] += cpu_time / 1000
        op_type_agg[op_name]["cuda_time_ms"] += cuda_time / 1000
        op_type_agg[op_name]["memory_mb"] += (cpu_mem + cuda_mem) / (1024 * 1024)

    result.op_profiles = op_profiles
    result.op_type_summary = op_type_agg
    result.total_cpu_time_ms = total_cpu / 1000
    result.total_cuda_time_ms = total_cuda / 1000
    result.total_flops = total_flops

    # CPU peak memory from profiler if no CUDA
    if device == "cpu" and op_profiles:
        total_cpu_alloc = sum(max(0, op.cpu_memory_usage) for op in op_profiles)
        result.peak_memory_mb = max(result.peak_memory_mb, total_cpu_alloc / (1024 * 1024))

    print(result.summary())

    # Also print torch.profiler's built-in table
    print(f"  torch.profiler table (top 20 by CPU time):")
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages(group_by_stack_n=0).table(
        sort_by=sort_key, row_limit=20,
    ))

    # Pruning targets: rank compressible ops by cost on the active device.
    # Pass the same key_averages used by .table() above so % time matches the
    # profiler's "Self CUDA %" column exactly.
    _print_pruning_targets(op_profiles, device, prof.key_averages(group_by_stack_n=0))

    return result


# Op kinds whose cost can be reduced by structured/unstructured pruning.
_PRUNABLE_OP_KEYWORDS = (
    "linear", "addmm", "matmul", "mm", "bmm",
    "conv1d", "conv2d", "conv3d", "convolution",
)


def _print_pruning_targets(op_profiles: List[OpProfile], device: str,
                           key_averages, top_k: int = 10) -> None:
    """Pick the top ops to prune, ranked by per-op cost on the active device.

    The denominator for `% time` is computed from `key_averages` the same way
    torch.profiler's table does, so the percentages match its `Self CUDA %` /
    `Self CPU %` columns exactly.
    """
    if not op_profiles:
        return

    use_cuda = device == "cuda"
    cost = lambda op: op.self_cuda_time_us if use_cuda else op.self_cpu_time_us
    label = "Self CUDA us" if use_cuda else "Self CPU us"

    # Replicate torch.profiler's denominator: sum of self device/cpu time over
    # the same key_averages it uses to build the table.
    def _self_time(evt):
        if use_cuda:
            return getattr(evt, 'self_device_time_total',
                           getattr(evt, 'self_cuda_time_total', 0)) or 0
        return getattr(evt, 'self_cpu_time_total', 0) or 0

    total_time = sum(_self_time(evt) for evt in key_averages) or 1.0
    total_flops = sum(getattr(evt, 'flops', 0) or 0 for evt in key_averages) or 1.0

    candidates = [
        op for op in op_profiles
        if any(k in op.name.lower() for k in _PRUNABLE_OP_KEYWORDS) and cost(op) > 0
    ]
    if not candidates:
        return
    candidates.sort(key=lambda op: -cost(op))

    print()
    print(f"  Pruning targets (top {min(top_k, len(candidates))} by {label}):")
    print(f"  {'Op':<32s} {label:>14s} {'% time':>8s} "
          f"{'GFLOPs':>10s} {'% flops':>8s} {'Shapes'}")
    print(f"  {'─'*32} {'─'*14} {'─'*8} {'─'*10} {'─'*8} {'─'*24}")
    for op in candidates[:top_k]:
        time_pct = 100.0 * cost(op) / total_time
        flops_pct = 100.0 * op.flops / total_flops if op.flops else 0.0
        gflops = op.flops / 1e9 if op.flops else 0.0
        shapes_str = ", ".join(op.input_shapes[:2])
        if len(op.input_shapes) > 2:
            shapes_str += f" +{len(op.input_shapes)-2}"
        print(
            f"  {op.name[:32]:<32s} {cost(op):>14.0f} "
            f"{time_pct:>7.2f}% {gflops:>10.3f} {flops_pct:>7.2f}% {shapes_str}"
        )
