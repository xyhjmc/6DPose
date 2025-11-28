"""Utility helpers to summarize model parameters and FLOPs."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


def _get_profiler_activities(device: torch.device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> int:
    """Estimate forward FLOPs using PyTorch profiler.

    Args:
        model: Model to profile.
        input_shape: Input tensor shape (N, C, H, W).
        device: Device where the model is located.

    Returns:
        Estimated number of floating point operations for a single forward pass.
    """
    was_training = model.training
    model.eval()

    dummy_input = torch.randn(input_shape, device=device)
    flops = 0

    activities = _get_profiler_activities(device)

    with torch.no_grad(), profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        model(dummy_input)

    for evt in prof.key_averages():
        if evt.flops is not None:
            flops += int(evt.flops)

    if was_training:
        model.train()

    return flops


def summarize_model_stats(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> Dict[str, float]:
    """Collect parameter count and FLOPs statistics for a model.

    Args:
        model: Model to summarize.
        input_shape: Input tensor shape (N, C, H, W) used for FLOPs estimation.
        device: Device where the model resides.

    Returns:
        A dictionary containing parameter count, parameter count in millions,
        FLOPs, and FLOPs in GFLOPs.
    """
    params = count_parameters(model)
    flops = estimate_flops(model, input_shape, device)
    return {
        "param_count": float(params),
        "param_millions": params / 1e6,
        "flops": float(flops),
        "gflops": flops / 1e9,
    }
