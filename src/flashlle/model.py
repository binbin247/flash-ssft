"""Initial field for the single-field Kerr LLE."""

from __future__ import annotations

import numpy as np

from .gpu import LLEParams


def initial_field(params: LLEParams) -> np.ndarray:
    """Soliton-like seed used by the reference LLE experiment."""
    if params.n_modes < 2:
        raise ValueError("n_modes must be at least 2")
    if params.d2 <= 0 or params.detuning_alpha <= 0:
        raise ValueError("the soliton-like seed requires d2 > 0 and alpha > 0")
    theta = np.linspace(-np.pi, np.pi, params.n_modes, endpoint=False)
    cw = params.pump_f / (1.0 + 1j * params.detuning_alpha)
    width = np.sqrt(2.0 * params.detuning_alpha / params.d2)
    pulse = np.sqrt(2.0 * params.detuning_alpha) / np.cosh(width * theta)
    return np.asarray(cw + pulse, dtype=np.complex128)
