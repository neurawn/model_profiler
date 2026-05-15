"""
ToMe Benchmark
==============
Evaluates token-merged ViT models on ImageNet-1k validation set,
replicating the evaluation methodology from Bolya et al. (2023)
"Token Merging: Your ViT But Faster" (arXiv:2210.09461).

Metrics:
  - Top-1 / Top-5 accuracy on ImageNet-1k val (50k images)
  - Throughput (images/sec) at optimal batch size
  - Average latency (ms/image)

If ImageNet is not available locally, downloads a 1k-sample subset
from HuggingFace (imagenet-1k) for quick benchmarking.

Usage:
    python tome_benchmark.py --models-dir ./all_token_merging_models
    python tome_benchmark.py --models-dir ./all_token_merging_models --imagenet-dir /data/imagenet/val
    python tome_benchmark.py --models-dir ./all_token_merging_models --num-samples 1000
"""

import os
import sys
import time
import argparse
import glob
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_DIR


# ImageNette class folder names → ImageNet-1k class indices
_IMAGENETTE_MAP = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _download_imagenette(transform) -> Optional[torch.utils.data.Dataset]:
    """Download ImageNette (10-class ImageNet subset) as a tarball fallback."""
    import tarfile
    import urllib.request
    from torchvision.datasets import ImageFolder

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    data_dir = os.path.join(MODEL_DIR, "imagenette2-320")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(val_dir):
        tgz_path = os.path.join(MODEL_DIR, "imagenette2-320.tgz")
        if not os.path.exists(tgz_path):
            print(f"  Downloading ImageNette (320px, ~100MB)...")
            try:
                urllib.request.urlretrieve(url, tgz_path)
            except Exception as e:
                print(f"  Download failed: {e}")
                return None

        print(f"  Extracting ImageNette...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(MODEL_DIR)

    if not os.path.isdir(val_dir):
        print(f"  ImageNette extraction failed.")
        return None

    # ImageFolder maps folder names alphabetically to 0..9,
    # but we need real ImageNet class indices for ViT evaluation.
    dataset = ImageFolder(val_dir, transform=transform)

    # Build mapping from ImageFolder's class_to_idx to ImageNet indices
    folder_to_imagenet = {}
    for folder_name, folder_idx in dataset.class_to_idx.items():
        if folder_name in _IMAGENETTE_MAP:
            folder_to_imagenet[folder_idx] = _IMAGENETTE_MAP[folder_name]

    # Remap labels
    dataset.samples = [
        (path, folder_to_imagenet.get(label, label))
        for path, label in dataset.samples
    ]
    dataset.targets = [s[1] for s in dataset.samples]

    print(f"  Loaded ImageNette val ({len(dataset)} images, 10 classes)")
    print(f"  NOTE: Accuracy reflects 10 ImageNet classes only.")
    print(f"        Relative comparisons between models are still valid.")
    return dataset


@dataclass
class BenchmarkResult:
    name: str
    strategy: str
    ratio: float
    top1: float
    top5: float
    throughput: float  # images/sec
    latency_ms: float  # ms per image
    num_samples: int


def load_imagenet_val(imagenet_dir: Optional[str], num_samples: int,
                      image_size: int = 224) -> DataLoader:
    """Load ImageNet validation set, or a HuggingFace subset if unavailable."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if imagenet_dir and os.path.isdir(imagenet_dir):
        from torchvision.datasets import ImageFolder
        print(f"  Loading ImageNet val from: {imagenet_dir}")
        dataset = ImageFolder(imagenet_dir, transform=transform)
    else:
        # Try full ImageNet from HuggingFace (requires auth), then fall back
        # to ImageNette downloaded as a tarball (10-class, ~4k val images).
        dataset = None

        try:
            from datasets import load_dataset
            print(f"  Trying HuggingFace ImageNet-1k (requires auth)...")
            hf_ds = load_dataset(
                "ILSVRC/imagenet-1k", split="validation",
                cache_dir=MODEL_DIR,
            )

            class HFImageNetDataset(torch.utils.data.Dataset):
                def __init__(self, hf_dataset, tfm):
                    self.ds = hf_dataset
                    self.transform = tfm
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, idx):
                    item = self.ds[idx]
                    image = item["image"].convert("RGB")
                    return self.transform(image), item["label"]

            dataset = HFImageNetDataset(hf_ds, transform)
            print(f"  Loaded ImageNet-1k val ({len(dataset)} images)")
        except Exception:
            pass

        if dataset is None:
            dataset = _download_imagenette(transform)

        if dataset is None:
            print(f"\n  Could not load any dataset.")
            print(f"  Options:")
            print(f"    1. pip install datasets && huggingface-cli login")
            print(f"    2. --imagenet-dir /path/to/imagenet/val")
            sys.exit(1)

    if num_samples > 0 and num_samples < len(dataset):
        # Deterministic subset spread across classes
        step = len(dataset) // num_samples
        indices = list(range(0, len(dataset), step))[:num_samples]
        dataset = Subset(dataset, indices)

    print(f"  Dataset size: {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return loader


def evaluate_accuracy(model, dataloader, device, forward_fn=None
                      ) -> Tuple[float, float]:
    """Compute top-1 and top-5 accuracy."""
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if forward_fn:
                output = forward_fn(model, images)
            else:
                output = model(images)

            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output

            _, pred1 = logits.topk(1, dim=1)
            _, pred5 = logits.topk(5, dim=1)

            correct1 += (pred1.squeeze(1) == labels).sum().item()
            correct5 += (labels.unsqueeze(1) == pred5).any(dim=1).sum().item()
            total += labels.size(0)

    return correct1 / total * 100, correct5 / total * 100


def measure_throughput(model, device, image_size=224,
                       warmup=10, iterations=50, forward_fn=None
                       ) -> Tuple[float, float]:
    """
    Measure throughput (images/sec) and latency (ms/image).

    Follows the ToMe paper: find optimal batch size, fp32, measure on GPU.
    """
    model.eval()

    # Find max batch size that fits in memory
    if device == "cuda":
        batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    else:
        batch_sizes = [32, 16, 8, 4, 2, 1]

    best_throughput = 0
    best_latency = float("inf")

    for bs in batch_sizes:
        try:
            dummy = torch.randn(bs, 3, image_size, image_size, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    if forward_fn:
                        forward_fn(model, dummy)
                    else:
                        model(dummy)

            # Time
            if device == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iterations):
                        if forward_fn:
                            forward_fn(model, dummy)
                        else:
                            model(dummy)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(iterations):
                        if forward_fn:
                            forward_fn(model, dummy)
                        else:
                            model(dummy)
                elapsed = time.perf_counter() - start

            throughput = (bs * iterations) / elapsed
            latency = elapsed / iterations * 1000 / bs  # ms per image

            if throughput > best_throughput:
                best_throughput = throughput
                best_latency = latency

            del dummy
            if device == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError:
            # OOM — try smaller batch
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

    return best_throughput, best_latency


def parse_model_filename(filename: str) -> Tuple[str, float]:
    """Extract strategy and ratio from filename like vit_base_tome_bipartite_r0.5.pt"""
    m = re.search(r"tome_(\w+)_r([\d.]+)\.pt$", filename)
    if m:
        return m.group(1), float(m.group(2))
    return "unknown", 0.0


def run_benchmark(args):
    from transformers import ViTForImageClassification
    from token_merging import apply_token_merging

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*78}")
    print(f"  ToMe BENCHMARK — ImageNet-1k Evaluation")
    print(f"  Device: {device} | Samples: {args.num_samples or 'all'}")
    print(f"{'='*78}")

    # ── 1. Load dataset ──
    print(f"\n[1/4] Loading dataset...")
    dataloader = load_imagenet_val(args.imagenet_dir, args.num_samples)

    # ── 2. Load baseline model ──
    print(f"\n[2/4] Loading baseline ViT-Base...")
    baseline_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", cache_dir=MODEL_DIR,
    ).to(device)

    def vit_forward(model, x):
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            return model(pixel_values=x)
        return model(x)

    # ── 3. Evaluate baseline ──
    print(f"\n[3/4] Evaluating baseline...")
    base_top1, base_top5 = evaluate_accuracy(
        baseline_model, dataloader, device, forward_fn=vit_forward,
    )
    base_throughput, base_latency = measure_throughput(
        baseline_model, device, forward_fn=vit_forward,
    )

    baseline = BenchmarkResult(
        name="ViT-Base (baseline)",
        strategy="none", ratio=1.0,
        top1=base_top1, top5=base_top5,
        throughput=base_throughput, latency_ms=base_latency,
        num_samples=len(dataloader.dataset),
    )
    print(f"  Baseline: top1={base_top1:.2f}%, throughput={base_throughput:.1f} im/s")

    # ── 4. Evaluate all token-merged models ──
    print(f"\n[4/4] Evaluating token-merged models...")
    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*.pt")))
    if not model_files:
        print(f"  No .pt files found in {args.models_dir}")
        return

    results: List[BenchmarkResult] = [baseline]

    for i, filepath in enumerate(model_files):
        filename = os.path.basename(filepath)
        strategy, ratio = parse_model_filename(filename)

        print(f"\n  [{i+1}/{len(model_files)}] {filename}")
        print(f"    Strategy: {strategy}, Keep ratio: {ratio:.0%}")

        # Load weights into fresh model
        checkpoint = torch.load(filepath, map_location="cpu")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", cache_dir=MODEL_DIR,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Apply token merging
        merge_config = checkpoint["merge_config"]
        wrapped = apply_token_merging(model, **merge_config)
        wrapped = wrapped.to(device)

        # Accuracy
        top1, top5 = evaluate_accuracy(
            wrapped, dataloader, device, forward_fn=vit_forward,
        )

        # Throughput
        throughput, latency = measure_throughput(
            wrapped, device, forward_fn=vit_forward,
        )

        results.append(BenchmarkResult(
            name=filename.replace(".pt", ""),
            strategy=strategy, ratio=ratio,
            top1=top1, top5=top5,
            throughput=throughput, latency_ms=latency,
            num_samples=len(dataloader.dataset),
        ))
        print(f"    top1={top1:.2f}%, throughput={throughput:.1f} im/s")

        # Cleanup
        del wrapped, model, checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Print results table ──
    print_results_table(results, baseline)

    # ── Save CSV ──
    save_csv(results, args.output)


def print_results_table(results: List[BenchmarkResult], baseline: BenchmarkResult):
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print(f"  ToMe BENCHMARK RESULTS — ImageNet-1k ({results[0].num_samples} samples)")
    print(f"{'='*100}")

    hdr = (f"  {'Model':<40} {'Strategy':<12} {'Ratio':>5} "
           f"{'Top-1':>7} {'Δ Top-1':>8} {'Top-5':>7} "
           f"{'im/s':>7} {'Speedup':>8} {'Lat(ms)':>8}")
    sep = f"  {'─'*38}  {'─'*10}  {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*8}"
    print(f"\n{hdr}")
    print(sep)

    for r in results:
        delta_top1 = r.top1 - baseline.top1
        speedup = r.throughput / baseline.throughput if baseline.throughput > 0 else 0

        delta_str = f"{delta_top1:+.2f}%" if r.strategy != "none" else "—"
        speedup_str = f"{speedup:.2f}x" if r.strategy != "none" else "1.00x"
        ratio_str = f"{r.ratio:.0%}" if r.strategy != "none" else "—"

        print(f"  {r.name:<40} {r.strategy:<12} {ratio_str:>5} "
              f"{r.top1:>6.2f}% {delta_str:>8} {r.top5:>6.2f}% "
              f"{r.throughput:>7.1f} {speedup_str:>8} {r.latency_ms:>7.2f}")

        # Separator between strategies
        if r != results[-1]:
            next_r = results[results.index(r) + 1]
            if next_r.strategy != r.strategy:
                print(sep)

    print(f"{'='*100}\n")


def save_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("model,strategy,ratio,top1,top5,throughput_ips,latency_ms,speedup,delta_top1,num_samples\n")
        baseline = results[0]
        for r in results:
            speedup = r.throughput / baseline.throughput if baseline.throughput > 0 else 0
            delta = r.top1 - baseline.top1
            f.write(f"{r.name},{r.strategy},{r.ratio:.2f},{r.top1:.2f},{r.top5:.2f},"
                    f"{r.throughput:.1f},{r.latency_ms:.2f},{speedup:.3f},{delta:.2f},"
                    f"{r.num_samples}\n")
    print(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token-merged ViT models on ImageNet-1k"
    )
    parser.add_argument("--models-dir", type=str,
                        default="./all_token_merging_models",
                        help="Directory containing saved .pt models")
    parser.add_argument("--imagenet-dir", type=str, default=None,
                        help="Path to ImageNet validation set (e.g. /data/imagenet/val). "
                             "If not provided, downloads from HuggingFace.")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all, default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device for evaluation")
    parser.add_argument("--output", type=str,
                        default="./profiling_results/tome_benchmark.csv",
                        help="Path to save CSV results")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
