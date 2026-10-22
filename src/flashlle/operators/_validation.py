"""Input checks shared by complex64 CUDA pointwise operators."""

from __future__ import annotations

import torch


def check_complex64_cuda(field: torch.Tensor, label: str) -> None:
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if field.device.type != "cuda":
        raise ValueError(f"{label} must reside on CUDA")
    if field.dtype != torch.complex64:
        raise TypeError(f"{label} must have dtype torch.complex64")
    if field.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional field")
    if not field.is_contiguous():
        raise ValueError(f"{label} must be contiguous")
