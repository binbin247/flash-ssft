"""Compare CPU and GPU LLE results with third-order dispersion."""

from __future__ import annotations

import numpy as np

from flashlle import LLEParams, initial_field, simulate_gpu, simulate_numpy


def main() -> None:
    params = LLEParams(n_modes=1024, d3=1e-4)
    seed = initial_field(params)
    cpu = simulate_numpy(seed, params, steps=1000)
    gpu = simulate_gpu(seed, params, steps=1000, graph_chunk=16)
    relative_error = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    print(f"TOD d3={params.d3}, N={params.n_modes}, relative field error={relative_error:.3e}")


if __name__ == "__main__":
    main()
