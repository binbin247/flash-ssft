"""Ordinary PyTorch GPU baselines with the production operators' interfaces.

These functions deliberately use PyTorch tensor operations, without Triton or
custom kernels. Pointwise operators update their input; a full step returns a
new angular-domain tensor and leaves its original input unchanged.
"""

from __future__ import annotations

import torch


def torch_kerr_(angular: torch.Tensor, h: float) -> torch.Tensor:
    """Apply the Kerr phase in place using ordinary complex64 tensor operations."""
    angular.mul_(torch.exp(1j * h * angular.abs().square()))
    return angular


def torch_kerr_raman_(angular: torch.Tensor, h: float, raman: float) -> torch.Tensor:
    """Apply Kerr–Raman rotation with ordinary PyTorch operations and real FFTs."""
    if raman == 0:
        return torch_kerr_(angular, h)
    intensity = angular.abs().square()
    spectrum = torch.fft.rfft(intensity)
    wavenumbers = torch.arange(spectrum.numel(), dtype=torch.float32, device=angular.device)
    if angular.numel() % 2 == 0:
        wavenumbers[-1].zero_()
    spectrum.mul_(1j * wavenumbers)
    derivative = torch.fft.irfft(spectrum, n=angular.numel())
    angular.mul_(torch.exp(1j * h * (intensity - raman * derivative)))
    return angular


def torch_linear_and_pump_(
    spectrum: torch.Tensor, linear: torch.Tensor, pump: complex
) -> torch.Tensor:
    """Apply precomputed linear factors and add the zero-mode pump in place."""
    spectrum.mul_(linear)
    spectrum[0].add_(pump)
    return spectrum


def torch_nlse_step(
    field: torch.Tensor, linear: torch.Tensor, w: torch.Tensor, h: float, omega0: float
) -> torch.Tensor:
    """Full dispersion/loss -> Kerr -> Euler shock step, with h=gamma0*dz."""
    field = torch.fft.ifft(torch.fft.fft(field) * linear)
    if h == 0:
        return field
    torch_kerr_(field, h)
    if omega0 != float("inf"):
        cubic = field.abs().square() * field
        derivative = torch.fft.ifft((1j * w.to(torch.float32) / omega0) * torch.fft.fft(cubic))
        field.sub_(h * derivative)
    return field
