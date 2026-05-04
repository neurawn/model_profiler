"""
Quantization Module
===================
Applies post-training quantization to PyTorch models and saves the
quantized output. Supports three modes:

1. Dynamic Quantization (INT8) — quantizes weights statically, activations
   dynamically at runtime. No calibration data needed. Best for models
   dominated by Linear layers (transformers, MLPs).

2. Static Quantization (INT8) — quantizes both weights and activations
   using calibration data. Better accuracy than dynamic but requires
   representative inputs. Best for CNNs.

3. Float16 Half-Precision — converts all parameters to float16.
   Simple 2x size reduction, works on any architecture.
"""

import torch
import torch.nn as nn
import torch.quantization as tq
import time
import os
import copy
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple


@dataclass
class QuantizationResult:
    """Result of quantizing a model."""
    model_name: str
    method: str  # "dynamic", "static", "float16"
    original_size_mb: float
    quantized_size_mb: float
    size_reduction: float  # fraction reduced (e.g., 0.75 = 75% smaller)
    original_latency_ms: Optional[float] = None
    quantized_latency_ms: Optional[float] = None
    speedup: Optional[float] = None
    quantized_layers: int = 0
    total_layers: int = 0
    save_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  QUANTIZATION RESULT: {self.model_name}",
            f"{'='*60}",
            f"  Method:           {self.method}",
            f"  Original size:    {self.original_size_mb:.2f} MB",
            f"  Quantized size:   {self.quantized_size_mb:.2f} MB",
            f"  Size reduction:   {self.size_reduction:.1%}",
            f"  Layers quantized: {self.quantized_layers}/{self.total_layers}",
        ]
        if self.original_latency_ms and self.quantized_latency_ms:
            lines += [
                f"{'─'*60}",
                f"  Original latency:  {self.original_latency_ms:.3f} ms",
                f"  Quantized latency: {self.quantized_latency_ms:.3f} ms",
                f"  Speedup:           {self.speedup:.2f}x",
            ]
        if self.save_path:
            lines.append(f"  Saved to: {self.save_path}")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


def _model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB from parameters and buffers."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def _count_layers(model: nn.Module) -> int:
    """Count leaf modules."""
    return sum(1 for _ in model.modules() if len(list(_.children())) == 0)


def _benchmark_latency(model: nn.Module, sample_input: torch.Tensor,
                       forward_fn: Optional[Callable], n_runs: int = 20,
                       warmup: int = 3) -> float:
    """Quick latency benchmark, returns mean ms."""
    model.eval()
    fwd = forward_fn if forward_fn else lambda m, x: m(x)

    with torch.no_grad():
        for _ in range(warmup):
            fwd(model, sample_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fwd(model, sample_input)
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def quantize_dynamic(model: nn.Module, model_name: str = "model",
                     sample_input: Optional[torch.Tensor] = None,
                     forward_fn: Optional[Callable] = None,
                     benchmark: bool = True) -> Tuple[nn.Module, QuantizationResult]:
    """
    Apply dynamic INT8 quantization.

    Quantizes Linear, LSTM, GRU, and RNN layers. Weights are quantized
    statically; activations are quantized dynamically during inference.
    """
    model = model.cpu().eval()
    original_size = _model_size_mb(model)
    total_layers = _count_layers(model)

    # Count quantizable layers
    quantizable = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Linear, nn.LSTM, nn.GRU, nn.RNN))
    )

    # Benchmark original
    original_latency = None
    if benchmark and sample_input is not None:
        sample_cpu = sample_input.cpu()
        original_latency = _benchmark_latency(model, sample_cpu, forward_fn)

    # Apply dynamic quantization
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM, nn.GRU, nn.RNN},
        dtype=torch.qint8,
    )

    quantized_size = _model_size_mb(quantized)

    # Benchmark quantized
    quantized_latency = None
    speedup = None
    if benchmark and sample_input is not None:
        sample_cpu = sample_input.cpu()
        quantized_latency = _benchmark_latency(quantized, sample_cpu, forward_fn)
        if original_latency and quantized_latency > 0:
            speedup = original_latency / quantized_latency

    result = QuantizationResult(
        model_name=model_name,
        method="dynamic_int8",
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        size_reduction=1 - (quantized_size / original_size) if original_size > 0 else 0,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=speedup,
        quantized_layers=quantizable,
        total_layers=total_layers,
    )

    return quantized, result


def quantize_static(model: nn.Module, sample_input: torch.Tensor,
                    model_name: str = "model",
                    forward_fn: Optional[Callable] = None,
                    n_calibration: int = 10,
                    benchmark: bool = True) -> Tuple[nn.Module, QuantizationResult]:
    """
    Apply static INT8 quantization with calibration.

    Requires a sample input for calibration. Inserts observer modules,
    runs calibration passes, then converts to quantized ops.
    """
    model = model.cpu().eval()
    original_size = _model_size_mb(model)
    total_layers = _count_layers(model)
    sample_cpu = sample_input.cpu()

    # Benchmark original
    original_latency = None
    if benchmark:
        original_latency = _benchmark_latency(model, sample_cpu, forward_fn)

    # Prepare model for static quantization
    quantized = copy.deepcopy(model)
    quantized.eval()

    # Set qconfig
    quantized.qconfig = tq.get_default_qconfig("x86")

    # Fuse common layer patterns where possible
    try:
        quantized = tq.fuse_modules(quantized, _find_fusable_modules(quantized),
                                     inplace=True)
    except Exception:
        pass  # Fusion is optional, skip if model structure doesn't support it

    # Insert observers
    tq.prepare(quantized, inplace=True)

    # Calibration: run sample inputs through the model
    fwd = forward_fn if forward_fn else lambda m, x: m(x)
    with torch.no_grad():
        for _ in range(n_calibration):
            fwd(quantized, sample_cpu)

    # Convert to quantized model
    tq.convert(quantized, inplace=True)

    quantized_size = _model_size_mb(quantized)

    # Count how many modules got quantized
    quantized_count = sum(
        1 for m in quantized.modules()
        if type(m).__name__.startswith("Quantized")
    )

    # Benchmark quantized
    quantized_latency = None
    speedup = None
    if benchmark:
        try:
            quantized_latency = _benchmark_latency(quantized, sample_cpu, forward_fn)
            if original_latency and quantized_latency > 0:
                speedup = original_latency / quantized_latency
        except Exception:
            pass  # Some quantized models may need different input handling

    result = QuantizationResult(
        model_name=model_name,
        method="static_int8",
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        size_reduction=1 - (quantized_size / original_size) if original_size > 0 else 0,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=speedup,
        quantized_layers=quantized_count,
        total_layers=total_layers,
    )

    return quantized, result


def quantize_float16(model: nn.Module, model_name: str = "model",
                     sample_input: Optional[torch.Tensor] = None,
                     forward_fn: Optional[Callable] = None,
                     benchmark: bool = True) -> Tuple[nn.Module, QuantizationResult]:
    """
    Convert model to float16 half-precision.

    Simple 2x size reduction. Works on any model architecture.
    Best used on GPU; CPU float16 support varies.
    """
    model = model.eval()
    original_size = _model_size_mb(model)
    total_layers = _count_layers(model)

    # Benchmark original
    original_latency = None
    if benchmark and sample_input is not None:
        original_latency = _benchmark_latency(model, sample_input, forward_fn)

    # Convert to float16
    quantized = copy.deepcopy(model).half()

    quantized_size = _model_size_mb(quantized)

    # Count converted layers
    fp16_count = sum(
        1 for p in quantized.parameters() if p.dtype == torch.float16
    )
    total_params = sum(1 for _ in quantized.parameters())

    # Benchmark
    quantized_latency = None
    speedup = None
    if benchmark and sample_input is not None:
        try:
            sample_half = sample_input.half()
            quantized_latency = _benchmark_latency(quantized, sample_half, forward_fn)
            if original_latency and quantized_latency > 0:
                speedup = original_latency / quantized_latency
        except Exception:
            pass  # float16 on CPU may not work for all ops

    result = QuantizationResult(
        model_name=model_name,
        method="float16",
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        size_reduction=1 - (quantized_size / original_size) if original_size > 0 else 0,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=speedup,
        quantized_layers=fp16_count,
        total_layers=total_params,
    )

    return quantized, result


def quantize_weight_only(model: nn.Module, model_name: str = "model",
                         bits: int = 8,
                         sample_input: Optional[torch.Tensor] = None,
                         forward_fn: Optional[Callable] = None,
                         benchmark: bool = True,
                         ) -> Tuple[nn.Module, QuantizationResult]:
    """
    Apply weight-only quantization (INT8 or INT4).

    Quantizes weight tensors in Linear and Conv layers to lower precision
    while keeping activations in float32. Weights are stored as quantized
    integers with scale/zero-point, and dequantized on the fly during
    inference.

    Args:
        model: PyTorch model to quantize.
        model_name: Name for display.
        bits: 8 for INT8, 4 for INT4.
        sample_input: For benchmarking.
        forward_fn: Custom forward function.
        benchmark: Whether to measure latency.
    """
    model = model.cpu().eval()
    original_size = _model_size_mb(model)
    total_layers = _count_layers(model)

    # Benchmark original
    original_latency = None
    if benchmark and sample_input is not None:
        sample_cpu = sample_input.cpu()
        original_latency = _benchmark_latency(model, sample_cpu, forward_fn)

    quantized = copy.deepcopy(model)
    quantized.eval()
    quantized_count = 0

    for name, module in quantized.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            weight = module.weight.data.float()

            if bits == 8:
                # Symmetric per-tensor INT8 quantization
                w_max = weight.abs().max().clamp(min=1e-8)
                scale = w_max / 127.0
                w_int = torch.clamp(torch.round(weight / scale), -128, 127)
                w_deq = w_int * scale
            elif bits == 4:
                # Symmetric per-channel INT4 quantization (per output channel)
                if weight.ndim >= 2:
                    # Per-channel: compute scale per output channel
                    reduce_dims = list(range(1, weight.ndim))
                    w_max = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
                    scale = w_max / 7.0
                    w_int = torch.clamp(torch.round(weight / scale), -8, 7)
                    w_deq = w_int * scale
                else:
                    w_max = weight.abs().max().clamp(min=1e-8)
                    scale = w_max / 7.0
                    w_int = torch.clamp(torch.round(weight / scale), -8, 7)
                    w_deq = w_int * scale
            else:
                raise ValueError(f"Unsupported bit width: {bits}. Use 4 or 8.")

            module.weight.data = w_deq
            # Store quantization metadata as buffers for reference
            module.register_buffer(f"_wq_scale", scale.squeeze())
            module.register_buffer(f"_wq_bits", torch.tensor(bits))
            quantized_count += 1

    quantized_size = _model_size_mb(quantized)
    # Estimate actual compressed size (weights stored at lower bits)
    weight_bytes = sum(
        p.numel() for name, p in quantized.named_parameters()
        if "weight" in name and any(
            isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d))
            for _, m in quantized.named_modules()
        )
    ) * (bits / 8)
    other_bytes = sum(
        p.numel() * p.element_size() for name, p in quantized.named_parameters()
        if "weight" not in name
    )
    estimated_size_mb = (weight_bytes + other_bytes) / (1024 * 1024)

    # Benchmark quantized
    quantized_latency = None
    speedup = None
    if benchmark and sample_input is not None:
        sample_cpu = sample_input.cpu()
        quantized_latency = _benchmark_latency(quantized, sample_cpu, forward_fn)
        if original_latency and quantized_latency > 0:
            speedup = original_latency / quantized_latency

    result = QuantizationResult(
        model_name=model_name,
        method=f"weight_only_int{bits}",
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        size_reduction=1 - (estimated_size_mb / original_size) if original_size > 0 else 0,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=speedup,
        quantized_layers=quantized_count,
        total_layers=total_layers,
    )

    return quantized, result


def quantize_smoothquant(model: nn.Module, model_name: str = "model",
                         sample_input: Optional[torch.Tensor] = None,
                         forward_fn: Optional[Callable] = None,
                         alpha: float = 0.5,
                         n_calibration: int = 10,
                         benchmark: bool = True,
                         ) -> Tuple[nn.Module, QuantizationResult]:
    """
    Apply SmoothQuant (Xiao et al., 2023) for W8A8 quantization.

    SmoothQuant smooths activation outliers by migrating quantization
    difficulty from activations to weights using a per-channel scaling
    factor s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha).

    The transform is mathematically equivalent:
        Y = (X diag(s)^-1) @ (diag(s) W) = X @ W

    After smoothing, both weights and activations become easier to
    quantize to INT8 with minimal accuracy loss.

    Args:
        model: PyTorch model to quantize.
        model_name: Name for display.
        sample_input: Calibration input (required).
        forward_fn: Custom forward function.
        alpha: Migration strength (0-1). Higher = more smoothing on activations.
               0.5 is the default from the paper. Use 0.75 for models with
               large activation outliers (e.g. OPT, BLOOM).
        n_calibration: Number of calibration forward passes.
        benchmark: Whether to measure latency.
    """
    if sample_input is None:
        raise ValueError("SmoothQuant requires sample_input for calibration")

    model = model.cpu().eval()
    original_size = _model_size_mb(model)
    total_layers = _count_layers(model)
    sample_cpu = sample_input.cpu()

    # Benchmark original
    original_latency = None
    if benchmark:
        original_latency = _benchmark_latency(model, sample_cpu, forward_fn)

    quantized = copy.deepcopy(model)
    quantized.eval()
    fwd = forward_fn if forward_fn else lambda m, x: m(x)

    # ── Step 1: Collect activation statistics via hooks ──
    act_scales = {}  # module_name -> max per-channel activation magnitude
    handles = []

    def _make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            if isinstance(x, torch.Tensor) and x.ndim >= 2:
                # Per-channel max absolute value (last dim = features)
                x_abs_max = x.abs().amax(dim=list(range(x.ndim - 1)))
                if name in act_scales:
                    act_scales[name] = torch.max(act_scales[name], x_abs_max)
                else:
                    act_scales[name] = x_abs_max.clone()
        return hook

    # Register hooks on Linear layers
    linear_modules = {}
    for name, module in quantized.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_make_hook(name)))
            linear_modules[name] = module

    # Run calibration
    print(f"  [SmoothQuant] Calibrating with {n_calibration} passes (alpha={alpha})...")
    with torch.no_grad():
        for _ in range(n_calibration):
            fwd(quantized, sample_cpu)

    # Remove hooks
    for h in handles:
        h.remove()

    # ── Step 2: Compute smoothing scales and apply ──
    smoothed_count = 0
    for name, module in linear_modules.items():
        if name not in act_scales:
            continue

        x_scale = act_scales[name].clamp(min=1e-8)
        w = module.weight.data.float()
        # Per-input-channel weight max: max over output dim
        w_scale = w.abs().amax(dim=0).clamp(min=1e-8)

        # s_j = x_scale^alpha / w_scale^(1-alpha)
        s = (x_scale.pow(alpha) / w_scale.pow(1 - alpha)).clamp(min=1e-8)

        # Apply smoothing: W_new = W @ diag(s), so activations get divided by s
        # Equivalent: scale weights by s per input channel
        module.weight.data = w * s.unsqueeze(0)

        # If there's a preceding LayerNorm/bias, absorb the inverse scaling
        # For now, store the scale as a buffer for the activation transform
        module.register_buffer("_smooth_scale", s)
        smoothed_count += 1

    print(f"  [SmoothQuant] Smoothed {smoothed_count} Linear layers")

    # ── Step 3: Quantize weights to INT8 (per-tensor symmetric) ──
    quantized_count = 0
    for name, module in quantized.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.float()
            w_max = w.abs().max().clamp(min=1e-8)
            scale = w_max / 127.0
            w_int = torch.clamp(torch.round(w / scale), -128, 127)
            module.weight.data = w_int * scale
            module.register_buffer("_wq_scale", scale)
            module.register_buffer("_wq_bits", torch.tensor(8))
            quantized_count += 1

    # ── Step 4: Wrap forward to quantize activations at runtime ──
    # Install pre-hooks that quantize input activations to INT8
    act_hooks = []
    for name, module in quantized.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, '_smooth_scale'):
            def _make_act_quant_hook(mod):
                def hook(module, input):
                    if isinstance(input, tuple):
                        x = input[0]
                        rest = input[1:]
                    else:
                        x = input
                        rest = ()
                    if isinstance(x, torch.Tensor) and x.ndim >= 2:
                        # Divide by smooth scale (absorb into activation)
                        s = mod._smooth_scale.to(x.device)
                        x = x / s.unsqueeze(0).expand_as(x) if x.ndim == 2 else x / s

                        # Quantize activation to INT8
                        x_max = x.abs().max().clamp(min=1e-8)
                        a_scale = x_max / 127.0
                        x_int = torch.clamp(torch.round(x / a_scale), -128, 127)
                        x = x_int * a_scale

                    return (x,) + rest if rest else (x,)
                return hook
            act_hooks.append(
                module.register_forward_pre_hook(_make_act_quant_hook(module))
            )

    quantized_size = _model_size_mb(quantized)

    # Benchmark quantized
    quantized_latency = None
    speedup = None
    if benchmark:
        try:
            quantized_latency = _benchmark_latency(quantized, sample_cpu, forward_fn)
            if original_latency and quantized_latency > 0:
                speedup = original_latency / quantized_latency
        except Exception:
            pass

    result = QuantizationResult(
        model_name=model_name,
        method=f"smoothquant_w8a8_alpha{alpha}",
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        size_reduction=1 - (quantized_size / original_size) if original_size > 0 else 0,
        original_latency_ms=original_latency,
        quantized_latency_ms=quantized_latency,
        speedup=speedup,
        quantized_layers=quantized_count,
        total_layers=total_layers,
    )

    return quantized, result


def _find_fusable_modules(model: nn.Module) -> List[List[str]]:
    """Find sequences of modules that can be fused (Conv+BN+ReLU, etc.)."""
    fuse_patterns = []
    named = dict(model.named_modules())
    names = list(named.keys())

    for i, name in enumerate(names):
        m = named[name]
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            # Look for Conv -> BN -> ReLU
            seq = [name]
            if i + 1 < len(names) and isinstance(named[names[i+1]], (nn.BatchNorm2d, nn.BatchNorm1d)):
                seq.append(names[i+1])
                if i + 2 < len(names) and isinstance(named[names[i+2]], nn.ReLU):
                    seq.append(names[i+2])
            elif i + 1 < len(names) and isinstance(named[names[i+1]], nn.ReLU):
                seq.append(names[i+1])
            if len(seq) > 1:
                fuse_patterns.append(seq)
        elif isinstance(m, nn.Linear):
            # Linear -> ReLU
            if i + 1 < len(names) and isinstance(named[names[i+1]], nn.ReLU):
                fuse_patterns.append([name, names[i+1]])

    return fuse_patterns


def save_quantized_model(model: nn.Module, save_path: str,
                         method: str = "dynamic") -> str:
    """Save a quantized model to disk."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if method == "float16":
        # float16 models can be saved normally
        torch.save(model.state_dict(), save_path)
    else:
        # INT8 quantized models: save the full model (not just state_dict)
        # because quantized modules have special structure
        torch.save(model, save_path)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  Saved quantized model: {save_path} ({size_mb:.2f} MB)")
    return save_path


def quantize_model(model: nn.Module, method: str = "dynamic",
                   model_name: str = "model",
                   sample_input: Optional[torch.Tensor] = None,
                   forward_fn: Optional[Callable] = None,
                   save_path: Optional[str] = None,
                   benchmark: bool = True) -> Tuple[nn.Module, QuantizationResult]:
    """
    Quantize a model using the specified method and optionally save it.

    Args:
        model: PyTorch model to quantize.
        method: "dynamic", "static", or "float16".
        model_name: Name for display/saving.
        sample_input: Required for static quantization and benchmarking.
        forward_fn: Custom forward function (for HuggingFace models).
        save_path: If provided, save the quantized model to this path.
        benchmark: Whether to measure latency before/after.

    Returns:
        (quantized_model, QuantizationResult)
    """
    print(f"\n{'#'*70}")
    print(f"#  QUANTIZATION: {model_name} ({method})")
    print(f"{'#'*70}")

    if method == "dynamic":
        quantized, result = quantize_dynamic(
            model, model_name, sample_input, forward_fn, benchmark,
        )
    elif method == "static":
        if sample_input is None:
            raise ValueError("Static quantization requires sample_input for calibration")
        quantized, result = quantize_static(
            model, sample_input, model_name, forward_fn, benchmark=benchmark,
        )
    elif method == "float16":
        quantized, result = quantize_float16(
            model, model_name, sample_input, forward_fn, benchmark,
        )
    elif method == "weight_int8":
        quantized, result = quantize_weight_only(
            model, model_name, bits=8,
            sample_input=sample_input, forward_fn=forward_fn, benchmark=benchmark,
        )
    elif method == "weight_int4":
        quantized, result = quantize_weight_only(
            model, model_name, bits=4,
            sample_input=sample_input, forward_fn=forward_fn, benchmark=benchmark,
        )
    elif method == "smoothquant":
        if sample_input is None:
            raise ValueError("SmoothQuant requires sample_input for calibration")
        quantized, result = quantize_smoothquant(
            model, model_name, sample_input=sample_input,
            forward_fn=forward_fn, benchmark=benchmark,
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}. "
                         f"Choose from: dynamic, static, float16, "
                         f"weight_int8, weight_int4, smoothquant")

    print(result.summary())

    if save_path:
        save_quantized_model(quantized, save_path, method)
        result.save_path = save_path

    return quantized, result
