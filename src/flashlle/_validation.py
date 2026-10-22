"""Shared checks for host LLE inputs."""

from numbers import Integral

import numpy as np


def check_input(psi0: np.ndarray, n_modes: int, steps: int) -> np.ndarray:
    """Validate the field shape, complex dtype and nonnegative step count."""
    field = np.asarray(psi0)
    if field.shape != (n_modes,):
        raise ValueError(f"psi0 must have shape ({n_modes},), got {field.shape}")
    if not np.issubdtype(field.dtype, np.complexfloating):
        raise TypeError("psi0 must contain complex values")
    if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    return field
