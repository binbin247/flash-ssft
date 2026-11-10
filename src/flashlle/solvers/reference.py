"""NumPy complex128 reference for the dispersive NLSE with self-steepening."""

from __future__ import annotations

import numpy as np

from ._validation import check_input
from .params import NLSEParams, angular_frequencies, linear_coefficients


def simulate_numpy(a0: np.ndarray, params: NLSEParams, steps: int) -> np.ndarray:
    """Propagate to z=steps*dz with first-order linear -> Kerr -> shock splitting.

    Kerr is exact; self-steepening uses an explicit Euler update of the rotated
    field. The reference implements the same integration scheme as the GPU
    solver, so agreement alone does not establish step-size convergence.
    """
    field = np.array(check_input(a0, params.n_samples, steps), dtype=np.complex128, copy=True)
    if steps == 0 or params.dz == 0:
        return field
    linear = linear_coefficients(params)
    w = angular_frequencies(params)
    h = params.gamma0 * params.dz
    for _ in range(steps):
        field = np.fft.ifft(np.fft.fft(field) * linear)
        if h != 0:
            field *= np.exp(1j * h * np.abs(field)**2)
            if np.isfinite(params.omega0):
                cubic = np.abs(field)**2 * field
                derivative = np.fft.ifft((1j * w / params.omega0) * np.fft.fft(cubic))
                field -= h * derivative
    return field
