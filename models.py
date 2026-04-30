"""
Model Loaders Module
====================
Provides functions to load each of the four model types for profiling:
1. Transformer (GPT-2) — text generation
2. CNN (ResNet-18) — image classification
3. ViT (Vision Transformer) — image classification (test set for framework)
4. Blackbox — wraps any nn.Module with no architecture knowledge

Also provides sample input generators for each model type.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable

# Local cache directory for all pretrained models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")


# ════════════════════════════════════════════════════════════════
# 1. TRANSFORMER: GPT-2
# ════════════════════════════════════════════════════════════════

def load_gpt2(model_size: str = "gpt2") -> Tuple[nn.Module, torch.Tensor, Callable]:
    """
    Load GPT-2 from HuggingFace transformers.

    Args:
        model_size: "gpt2" (117M), "gpt2-medium" (345M), "gpt2-large" (774M)

    Returns:
        (model, sample_input, forward_fn)
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"  Loading {model_size}...")
    model = GPT2LMHeadModel.from_pretrained(model_size, cache_dir=MODEL_DIR, token=HF_TOKEN)
    tokenizer = GPT2Tokenizer.from_pretrained(model_size, cache_dir=MODEL_DIR, token=HF_TOKEN)

    # Create sample input
    sample_text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer(sample_text, return_tensors="pt")
    input_ids = tokens["input_ids"]

    # Custom forward function (HuggingFace models return dicts)
    def forward_fn(m, x):
        return m(x, labels=x)  # Include labels for loss computation

    print(f"  ✓ {model_size} loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, input_ids, forward_fn


def gpt2_input_fn(batch_size: int, seq_len: int = 64) -> torch.Tensor:
    """Generate random token IDs for GPT-2 batch scaling tests."""
    return torch.randint(0, 50257, (batch_size, seq_len))


# ════════════════════════════════════════════════════════════════
# 2. CNN: ResNet-18
# ════════════════════════════════════════════════════════════════

def load_resnet18(pretrained: bool = True) -> Tuple[nn.Module, torch.Tensor, None]:
    """
    Load ResNet-18 from torchvision.

    Returns:
        (model, sample_input, None)  — None means use default forward
    """
    from torchvision.models import resnet18, ResNet18_Weights

    print("  Loading ResNet-18...")
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    # Standard ImageNet input: batch=1, 3 channels, 224x224
    sample_input = torch.randn(1, 3, 224, 224)

    print(f"  ✓ ResNet-18 loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, sample_input, None


def resnet_input_fn(batch_size: int) -> torch.Tensor:
    """Generate random images for ResNet batch scaling tests."""
    return torch.randn(batch_size, 3, 224, 224)


# ════════════════════════════════════════════════════════════════
# 3. VISION TRANSFORMER: ViT-Base
# ════════════════════════════════════════════════════════════════

def load_vit(model_name: str = "google/vit-base-patch16-224") -> Tuple[nn.Module, torch.Tensor, Callable]:
    """
    Load Vision Transformer from HuggingFace.

    This serves as the "test set" for the framework — the recommender
    should determine whether quantization or pruning is better for ViT
    (a hybrid architecture with both conv patch embedding and attention).

    Returns:
        (model, sample_input, forward_fn)
    """
    from transformers import ViTForImageClassification

    print(f"  Loading ViT ({model_name})...")
    model = ViTForImageClassification.from_pretrained(model_name, cache_dir=MODEL_DIR, token=HF_TOKEN)

    # ViT expects pixel_values
    sample_input = torch.randn(1, 3, 224, 224)

    def forward_fn(m, x):
        return m(pixel_values=x)

    print(f"  ✓ ViT loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, sample_input, forward_fn


def vit_input_fn(batch_size: int) -> torch.Tensor:
    """Generate random images for ViT batch scaling tests."""
    return torch.randn(batch_size, 3, 224, 224)


# ════════════════════════════════════════════════════════════════
# 4. BLACKBOX MODEL WRAPPER
# ════════════════════════════════════════════════════════════════

class BlackboxModel(nn.Module):
    """
    Wraps any nn.Module to simulate a "blackbox" scenario.
    The profiler should work without knowing the original architecture.

    This demonstrates that the framework can analyze unknown models
    by examining the parameter structure and runtime behavior.
    """

    def __init__(self, model: nn.Module, name: str = "blackbox"):
        super().__init__()
        self.wrapped = model
        self._name = name
        # Deliberately hide the class name
        self.__class__.__name__ = "BlackboxModel"

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    @property
    def name(self):
        return self._name


def load_blackbox_model(model: Optional[nn.Module] = None,
                        sample_input: Optional[torch.Tensor] = None,
                        forward_fn: Optional[Callable] = None,
                        name: str = "blackbox") -> Tuple[nn.Module, torch.Tensor, Optional[Callable]]:
    """
    Wrap an existing model as a blackbox, or create a custom unknown model.

    If no model is provided, creates a moderately complex custom architecture
    that mixes different layer types — the framework should still be able
    to profile and recommend compression for it.
    """
    if model is not None:
        print(f"  Wrapping existing model as blackbox...")
        wrapped = BlackboxModel(model, name=name)
        if sample_input is None:
            sample_input = torch.randn(1, 3, 224, 224)  # default
        return wrapped, sample_input, forward_fn

    # Create a custom "unknown" architecture
    print("  Creating custom blackbox model...")

    class CustomUnknownNet(nn.Module):
        """
        A deliberate mix of conv, attention-like, and linear layers.
        The profiler should handle this without prior knowledge.
        """
        def __init__(self):
            super().__init__()
            # Conv backbone
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            # Linear head with a pseudo-attention mechanism
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 7 * 7, 512)
            self.query = nn.Linear(512, 128)
            self.key = nn.Linear(512, 128)
            self.value = nn.Linear(512, 128)
            self.out_proj = nn.Linear(128, 256)
            self.classifier = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.features(x)
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))

            # Simple self-attention-like operation
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            attn = torch.softmax(q * k / (128 ** 0.5), dim=-1)
            x = attn * v
            x = self.out_proj(x)
            x = self.dropout(x)
            return self.classifier(x)

    model = BlackboxModel(CustomUnknownNet(), name="custom_blackbox")
    sample_input = torch.randn(1, 3, 224, 224)

    print(f"  ✓ Blackbox model created ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, sample_input, None


# ════════════════════════════════════════════════════════════════
# UTILITY: Load all models
# ════════════════════════════════════════════════════════════════

def load_all_models() -> dict:
    """
    Load all four model types and return a dict of:
    {name: (model, sample_input, forward_fn)}
    """
    models = {}

    print("\n[1/4] Loading Transformer (GPT-2)...")
    models["GPT-2"] = load_gpt2()

    print("\n[2/4] Loading CNN (ResNet-18)...")
    models["ResNet-18"] = load_resnet18()

    print("\n[3/4] Loading ViT (Vision Transformer)...")
    models["ViT-Base"] = load_vit()

    print("\n[4/4] Loading Blackbox Model...")
    models["Blackbox"] = load_blackbox_model()

    print("\n✓ All models loaded.\n")
    return models
