"""PyTorch CUDA baseline for the dispersive NLSE with self-steepening.

This module mirrors :mod:`nlse_numpy_baseline` with ordinary PyTorch tensor
operations. It deliberately does not use Triton operators or CUDA Graphs, so
it can serve as a reference for the optimized GPU solver.
"""

from __future__ import annotations

import numpy as np
import torch

from ._validation import check_input
from .params import NLSEParams, angular_frequencies, linear_coefficients


def pytorch_nlse_step(
    field: torch.Tensor,
    linear: torch.Tensor,
    w: torch.Tensor,
    h: float,
    omega0: float,
) -> torch.Tensor:
    """Run one ordinary PyTorch NLSE step on an already prepared GPU field."""
    field = torch.fft.ifft(torch.fft.fft(field) * linear)
    if h == 0:
        return field

    field.mul_(torch.exp(1j * h * field.abs().square()))
    if omega0 == float("inf"):
        return field

    cubic = field.abs().square() * field
    derivative = torch.fft.ifft(
        (1j * w.to(torch.float32) / omega0) * torch.fft.fft(cubic)
    )
    field.sub_(h * derivative)
    return field


def simulate_pytorch(a0: np.ndarray, params: NLSEParams, steps: int) -> np.ndarray:
    """Propagate on the current CUDA device and return a new NumPy field.

    The splitting and update order are the same as in ``simulate_numpy``:
    linear propagation, exact Kerr rotation, and explicit-Euler
    self-steepening. The GPU baseline uses ``complex64`` to match the CUDA
    solver's field precision.
    """
    field = check_input(a0, params.n_samples, steps)
    if steps == 0 or params.dz == 0:
        return np.array(field, dtype=np.complex64, copy=True)

    device = torch.device("cuda", torch.cuda.current_device())
    field = torch.as_tensor(field, dtype=torch.complex64, device=device).contiguous()
    linear = torch.as_tensor(
        linear_coefficients(params), dtype=torch.complex64, device=device
    )
    w = torch.as_tensor(
        angular_frequencies(params), dtype=torch.float32, device=device
    )
    h = float(params.gamma0 * params.dz)

    with torch.no_grad():
        for _ in range(steps):
            field = pytorch_nlse_step(field, linear, w, h, params.omega0)

    return field.cpu().numpy().copy()
