"""Propagate a normalized NLSE pulse with third-order dispersion and shock."""

from __future__ import annotations

import argparse

import numpy as np

from flashlle.solvers import NLSEParams, simulate_numpy, time_grid


def example_params(n_samples: int = 1024, beta3: float = 0.05) -> NLSEParams:
    """Keep a fixed time window when comparing different sampling resolutions."""
    return NLSEParams(n_samples=n_samples, dt=20.0 / n_samples, dz=0.001,
                      beta2=-1.0, beta3=beta3, gamma0=1.0, omega0=20.0)


def initial_field(params: NLSEParams) -> np.ndarray:
    """Unit-amplitude sech pulse; a launch condition, not an assumed solution."""
    return np.asarray(1.0 / np.cosh(time_grid(params)), dtype=np.complex128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "gpu", "compare"), default="numpy")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 0:
        parser.error("samples must be positive and steps must be nonnegative")
    params = example_params(args.samples)
    seed = initial_field(params)
    if args.backend in ("numpy", "compare"):
        final = cpu = simulate_numpy(seed, params, steps=args.steps)
    if args.backend in ("gpu", "compare"):
        from flashlle.solvers.nlse_gpu_opt import simulate_gpu

        final = gpu = simulate_gpu(seed, params, steps=args.steps, graph_chunk=16)
    if not np.isfinite(final).all():
        raise FloatingPointError("Non-finite field; reduce dz and check resolution")
    print(f"NLSE beta3={params.beta3}, N={params.n_samples}, z={args.steps * params.dz:g}, "
          f"energy={params.dt * np.sum(np.abs(final)**2):.8g}")
    if args.backend == "compare":
        relative_error = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
        print(f"GPU/NumPy relative field error={relative_error:.3e}")


if __name__ == "__main__":
    main()
