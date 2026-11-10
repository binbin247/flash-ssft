"""Ordinary PyTorch references used by the GPU operator tests."""

from __future__ import annotations

import torch


def torch_kerr_(angular: torch.Tensor, h: float) -> torch.Tensor:
    """Apply the Kerr phase in place using ordinary PyTorch operations."""
    angular.mul_(torch.exp(1j * h * angular.abs().square()))
    return angular


def torch_kerr_raman_(angular: torch.Tensor, h: float, raman: float) -> torch.Tensor:
    """Apply a Kerr–Raman rotation using ordinary PyTorch operations."""
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
    """Apply linear factors and add a zero-mode pump in place."""
    spectrum.mul_(linear)
    spectrum[0].add_(pump)
    return spectrum


def torch_nlse_step(
    field: torch.Tensor, linear: torch.Tensor, w: torch.Tensor, h: float, omega0: float
) -> torch.Tensor:
    """Run one ordinary PyTorch NLSE step."""
    field = torch.fft.ifft(torch.fft.fft(field) * linear)
    if h == 0:
        return field
    torch_kerr_(field, h)
    if omega0 != float("inf"):
        cubic = field.abs().square() * field
        derivative = torch.fft.ifft(
            (1j * w.to(torch.float32) / omega0) * torch.fft.fft(cubic)
        )
        field.sub_(h * derivative)
    return field
