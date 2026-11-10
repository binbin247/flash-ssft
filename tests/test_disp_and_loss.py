"""Check NLSE linear propagation against independent Fourier-mode solutions."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest
import torch

from flashlle.operators import DispParams, disp_and_loss_, make_linear


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton operator checks")
    return torch.device("cuda")


@pytest.mark.parametrize("n", [1, 255, 256, 257])
@pytest.mark.parametrize(
    "dz,alpha,beta2,beta3,beta4",
    [
        (0.0, 0.4, -0.3, 0.2, 0.1),
        (0.2, 0.4, 0.0, 0.0, 0.0),
        (0.2, 0.0, -0.3, 0.0, 0.0),
        (0.2, 0.0, 0.0, 0.2, 0.0),
        (0.2, 0.0, 0.0, 0.0, 0.1),
        (0.2, 0.4, -0.3, 0.2, 0.1),
    ],
    ids=["zero-step", "loss", "beta2", "beta3", "beta4", "combined"],
)
def test_disp_and_loss_fourier_modes(n, dz, alpha, beta2, beta3, beta4, cuda):
    dt = 0.5
    time = np.arange(n) * dt
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    params = DispParams(beta2=beta2, beta3=beta3, beta4=beta4)
    factors = make_linear(torch.tensor(omega, device=cuda), dz, alpha, params)
    assert factors.dtype == torch.complex64
    assert factors.is_contiguous()

    seed = np.zeros(n, dtype=np.complex128)
    expected = np.zeros(n, dtype=np.complex128)
    # Include DC and both frequency signs to catch odd-order sign mistakes.
    for mode, amplitude in [(0, 0.6 + 0.2j), (1, -0.3 + 0.4j), (-2, 0.2 - 0.1j)]:
        frequency = omega[mode % n]
        wave = amplitude * np.exp(1j * frequency * time)
        # Substitute each mode directly into the user's differential equation.
        generator = -alpha / 2 - sum(
            1j**(m - 1) * beta / factorial(m) * (1j * frequency)**m
            for m, beta in enumerate((beta2, beta3, beta4), start=2)
        )
        seed += wave
        expected += wave * np.exp(dz * generator)

    spectrum = torch.fft.fft(torch.tensor(seed, dtype=torch.complex64, device=cuda))
    original_factors = factors.clone()
    assert disp_and_loss_(spectrum, factors) is spectrum
    actual = torch.fft.ifft(spectrum).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-7)
    torch.testing.assert_close(factors, original_factors, rtol=0, atol=0)


def test_disp_and_loss_rejects_mismatched_shapes(cuda):
    spectrum = torch.ones(257, dtype=torch.complex64, device=cuda)
    factors = torch.ones(256, dtype=torch.complex64, device=cuda)
    with pytest.raises(ValueError, match="matching shape and device"):
        disp_and_loss_(spectrum, factors)


def test_disp_and_loss_empty(cuda):
    spectrum = torch.empty(0, dtype=torch.complex64, device=cuda)
    assert disp_and_loss_(spectrum, spectrum.clone()) is spectrum


@pytest.mark.parametrize("dz", [0.0, 0.2])
def test_make_linear_loss_only(dz):
    w = torch.tensor([0.0, 2.0, -2.0])
    linear = make_linear(w, dz, alpha=0.4, disp_params=DispParams())
    assert linear.device == w.device
    assert linear.dtype == torch.complex64
    torch.testing.assert_close(linear, torch.full_like(linear, np.exp(-0.4*dz/2)))


def test_make_linear_optical_frequencies():
    """Large frequencies and small physical beta values must not overflow."""
    w = torch.tensor([0.0, 1e13, -1e13], dtype=torch.float32)
    params = DispParams(beta2=-2e-26, beta3=1e-39, beta4=1e-52)
    actual = make_linear(w, dz=0.2, alpha=0.4, disp_params=params)
    omega = w.numpy().astype(np.float64)
    generator = -0.4/2 - sum(
        1j**(m - 1) * beta / factorial(m) * (1j * omega)**m
        for m, beta in enumerate((params.beta2, params.beta3, params.beta4), start=2)
    )
    np.testing.assert_allclose(actual.numpy(), np.exp(0.2 * generator), rtol=1e-6)
