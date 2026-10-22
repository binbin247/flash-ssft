"""NumPy complex128 reference for the normalized single-field LLE."""

from __future__ import annotations

import numpy as np

from ._validation import check_input
from .gpu import LLEParams, linear_coefficients


def simulate_numpy(psi0: np.ndarray, params: LLEParams, steps: int) -> np.ndarray:
    """Evolve with linear propagation, an N*F*h pump increment and Kerr in complex128."""
    field = check_input(psi0, params.n_modes, steps)
    linear, pump = linear_coefficients(params)
    spectrum = np.fft.fft(np.asarray(field, dtype=np.complex128))
    for _ in range(steps):
        spectrum *= linear
        spectrum[0] += pump
        angular = np.fft.ifft(spectrum)
        angular *= np.exp(1j * params.h * np.abs(angular) ** 2)
        spectrum = np.fft.fft(angular)
    return np.fft.ifft(spectrum)
