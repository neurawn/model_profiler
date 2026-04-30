# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NYU final project: a PyTorch model profiling and compression recommendation framework. It profiles neural network architectures (GPT-2, ResNet-18, ViT, and blackbox models) using static analysis and runtime benchmarking, then recommends optimal compression methods (quantization, pruning, distillation, low-rank factorization).

The core hypothesis being tested: Transformers with uniform parameter distributions are better suited for quantization, while CNNs with uneven distributions benefit more from structured pruning. ViT (hybrid) serves as the test case where the framework must automatically determine the best method.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (all models)
python main.py

# Profile a single model
python main.py --model gpt2
python main.py --model resnet
python main.py --model vit
python main.py --model blackbox

# Static analysis only (skip runtime benchmarking)
python main.py --static-only

# Specify device and target
python main.py --device cuda --target-device mobile

# Results are saved to ./profiling_results/ as JSON
```

## Architecture

The pipeline flows: **models.py** (load) → **static_profiler.py** (analyze weights) → **dynamic_profiler.py** (benchmark runtime) → **compression_recommender.py** (recommend) → **main.py** (orchestrate).

- **`static_profiler.py`** — `StaticProfiler` analyzes model architecture without inference: parameter distributions, layer type composition, sparsity, weight entropy, rank estimates, and compression friendliness scores. Produces `StaticProfile` dataclass.

- **`dynamic_profiler.py`** — `DynamicProfiler` benchmarks runtime: latency (with warmup), peak memory, FLOPs (via fvcore or manual estimation), per-layer timing via hooks, and throughput. Uses `tracemalloc` on CPU, CUDA memory APIs on GPU. Produces `DynamicProfile` dataclass.

- **`compression_recommender.py`** — `CompressionRecommender` takes static+dynamic profiles, extracts a feature vector, classifies architecture (transformer/cnn/hybrid/rnn/unknown), and scores 7 compression methods using weighted feature matching. Currently commented out in `main.py` pipeline.

- **`models.py`** — Model loaders returning `(model, sample_input, forward_fn)` tuples. HuggingFace models (GPT-2, ViT) need custom `forward_fn` because they return dicts. `BlackboxModel` wraps any `nn.Module` to simulate unknown architecture profiling.

- **`main.py`** — CLI entry point with argparse. Orchestrates the pipeline, runs comparative analysis validating the project's hypotheses, and saves JSON results.

## Key Patterns

- Each model loader returns a 3-tuple `(model, sample_input, forward_fn)` — `forward_fn` is `None` for standard models, a callable for HuggingFace models.
- Profile results use dataclasses with `.to_dict()` for JSON serialization and `.summary()` for formatted console output.
- The compression recommender is currently commented out in `main.py:67-78` and `main.py:103-128` — the `compare_results` and `save_results` functions reference it but will error if called without it.
- Imports in `main.py` use `sys.path` manipulation for direct script execution; the package also has `__init__.py` for proper module imports.
