# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NYU final project: a PyTorch model profiling and compression framework. It profiles neural network architectures (GPT-2, ResNet-18, ViT, CLIP) using static analysis and graph export, then generates compression plans with per-layer quantization assignment, pruning candidates, token merging estimates, and KV cache quantization priority.

The core hypothesis: Transformers with uniform parameter distributions are better suited for quantization, while CNNs with uneven distributions benefit more from structured pruning. ViT (hybrid) and VLM (CLIP) serve as test cases where the framework automatically determines the best compression strategy.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# ── Static Profiling ──
python main.py --model gpt2           # Profile GPT-2
python main.py --model vit            # Profile ViT
python main.py --model vlm            # Profile CLIP
python main.py --model resnet         # Profile ResNet-18
python main.py --model all            # Profile all models

# ── Graph Analysis (export + FLOPs + KV cache + compression plan) ──
python main.py --model gpt2 --graph
python main.py --model vit --graph
python main.py --model vlm --graph
python main.py --model gpt2 --graph --save-graph   # Also save graph files/CSVs
python main.py --model gpt2 --graph --full-export  # Use torch.compile (CPU-intensive, GPU-accelerated)

# ── Apply Plan-Based Mixed-Precision Quantization ──
python main.py --model gpt2 --graph --apply-quantization
python main.py --model vit --graph --apply-quantization

# ── Uniform Quantization ──
python main.py --model gpt2 --quantize weight_int8
python main.py --model gpt2 --quantize weight_int4
python main.py --model gpt2 --quantize dynamic
python main.py --model gpt2 --quantize smoothquant
python main.py --model gpt2 --quantize float16

# ── Token Merging (ViT/VLM only) ──
python main.py --model vit --token-merging                    # Analyze strategies
python main.py --model vit --apply-tome                       # Apply and save model
python main.py --model vit --apply-tome --merge-ratio 0.7     # Custom ratio
python main.py --model vlm --apply-tome --merge-strategy kmeans

# ── Dynamic Profiling (torch.profiler) ──
python main.py --model gpt2 --dynamic-profile                 # CPU profiling
python main.py --model gpt2 --dynamic-profile --device cuda   # GPU profiling
python main.py --model gpt2 --dynamic-profile --save-trace    # Save Chrome trace
python main.py --model gpt2 --dynamic-profile --profile-runs 10

# ── Local Models (SafeTensors, state dicts, .pt files) ──
python main.py --local ./small --arch gpt2
python main.py --local ./small --arch gpt2 --graph --apply-quantization
python main.py --local ./weights.pt --arch resnet18 --quantize weight_int4
python main.py --local ./vit.safetensors --arch google/vit-base-patch16-224 --apply-tome
python main.py --local ./small --arch gpt2 --dynamic-profile --device cuda

# ── Custom Options ──
python main.py --model gpt2 --graph --context-lengths 512,2048,16384
python main.py --model gpt2 --quantize weight_int8 --save-quantized ./gpt2_int8.pt
python main.py --model vit --graph --output-dir ./my_results

# Results are saved to ./profiling_results/ as JSON
```

## Architecture

The pipeline flows: **models.py** (load) → **static_profiler.py** (analyze weights) → **graph_export.py** (export graph, detect compressible ops) → **static_profiler.py** (graph-based FLOPs/KV cache/compression plan) → **quantize.py** / **token_merging.py** (apply compression) → **main.py** (orchestrate).

- **`main.py`** — CLI entry point. Orchestrates profiling, graph analysis, quantization, and token merging. Handles local model loading (SafeTensors, state dicts, full models via `--local` + `--arch`).

- **`static_profiler.py`** — Two profiling modes:
  - `StaticProfiler.profile()` — weight-only analysis: parameter counts, layer types, depth.
  - `StaticProfiler.profile_from_graph()` — graph-based: per-node FLOPs/params/activation memory, arithmetic intensity, roofline placement, KV cache budget, per-block aggregation, and `CompressionPlan` (Pareto ranking, quant assignment, pruning candidates, token merging estimates, KV quant priority).

- **`graph_export.py`** — Exports model graph via torch.export → torch.fx → torch.compile → module-walk fallback. `torch.compile` is available via `--full-export` but skipped by default (CPU-intensive, uses GPU if available). Detects compressible ops, builds dependency DAG, saves operator CSVs and compressibility reports.

- **`quantize.py`** — Six quantization methods: `dynamic` (W8A8 dynamic), `static` (W8A8 calibrated), `float16`, `weight_int8`, `weight_int4`, `smoothquant` (W8A8 with activation smoothing). The `apply_quant_plan()` in main.py applies mixed-precision per-layer quantization from the compression plan.

- **`token_merging.py`** — Three token merging strategies for ViT/VLM: bipartite soft matching (ToMe), k-means clustering, average pooling. `TokenMergingWrapper` applies merging at inference via hooks. `TokenMergingProfiler` compares strategies.

- **`dynamic_profile.py`** — Uses `torch.profiler` for runtime analysis: per-operator CPU/CUDA latency, memory allocation per op, tensor shapes, call stacks, FLOPs per op. Supports Chrome trace export (`--save-trace`) for visualization in `chrome://tracing`.

- **`models.py`** — Model loaders returning `(model, sample_input, forward_fn)` tuples for GPT-2 (all sizes), ResNet-18, ViT-Base, and CLIP. HuggingFace models download to local `./model/` directory.

## Key Patterns

- Each model loader returns a 3-tuple `(model, sample_input, forward_fn)` — `forward_fn` is `None` for standard models, a callable for HuggingFace models.
- Profile results use dataclasses with `.to_dict()` for JSON serialization and `.summary()` for formatted console output.
- `--graph` is the unified flag for graph export + graph-based profiling + compression plan. Add `--save-graph` to persist graph files.
- `--apply-quantization` requires `--graph` (needs the compression plan). `--quantize` works standalone.
- `--dynamic-profile` skips static profiling by default (runs only torch.profiler). Static profiling still runs if combined with `--graph`, `--quantize`, or `--apply-quantization`.
- For local models: SafeTensors and state dicts require `--arch` to specify the model architecture. Full `nn.Module` `.pt` files load directly.
- HuggingFace models cache to `./model/` via `HF_HOME` env var set in main.py.
- GPT-2 uses HuggingFace `Conv1D` (not `nn.Linear`) — the profiler detects this automatically.
- `--graph` defaults to fast module-walk (no torch.compile). Add `--full-export` for full torch.compile graph capture (uses GPU if available, but compiler is still CPU-bound).
