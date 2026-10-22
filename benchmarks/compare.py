"""Host-to-host benchmark of the composable GPU LLE solver against NumPy.

From the project root::

    python benchmarks/compare.py --modes 1024 2048 4096 \
        --steps 1000 10000 100000 --repeats 3

The timed call receives a host initial field and returns the final host field.
Initialization, Triton compilation, and CUDA Graph capture are represented by
``first_call_seconds``; warmed repetitions determine ``median_seconds``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flashlle import LLEParams, initial_field, simulate_gpu, simulate_numpy


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


def _accuracy(field: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    field = np.asarray(field, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    power = float(np.mean(np.abs(field) ** 2))
    reference_power = float(np.mean(np.abs(reference) ** 2))
    spectrum = np.abs(np.fft.fft(field)) ** 2
    reference_spectrum = np.abs(np.fft.fft(reference)) ** 2
    return {
        "field_relative_l2": _relative_l2(field, reference),
        "mean_power": power,
        "mean_power_relative_error": abs(power - reference_power) / max(reference_power, 1e-30),
        "power_spectrum_relative_l2": _relative_l2(spectrum, reference_spectrum),
    }


def _timed_call(
    solver: Callable[[np.ndarray, LLEParams, int], np.ndarray],
    seed: np.ndarray,
    params: LLEParams,
    steps: int,
    *,
    gpu: bool,
) -> tuple[float, np.ndarray]:
    gc.collect()
    if gpu:
        import torch

        torch.cuda.synchronize()
    start = time.perf_counter()
    final = np.asarray(solver(seed, params, steps))
    if gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if final.shape != (params.n_modes,) or not np.issubdtype(final.dtype, np.complexfloating):
        raise ValueError(f"Unexpected solver result: shape={final.shape}, dtype={final.dtype}")
    if not np.isfinite(final).all():
        raise FloatingPointError("Solver returned a non-finite field")
    return elapsed, final


def _software_hardware(include_gpu: bool) -> dict[str, object]:
    info: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for package in ("torch", "triton"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = None
    if include_gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU benchmark requested, but CUDA is unavailable")
        info["gpu"] = torch.cuda.get_device_name()
        info["gpu_compute_capability"] = list(torch.cuda.get_device_capability())
        info["torch_cuda"] = torch.version.cuda
    return info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--d3", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--extra-warmups", type=int, default=0,
                        help="Additional calls after the recorded first call")
    parser.add_argument("--graph-chunk", type=int, default=16)
    parser.add_argument("--backends", nargs="+", choices=("numpy", "gpu"),
                        default=["numpy", "gpu"])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "compare.json")
    args = parser.parse_args()
    if any(n <= 0 for n in args.modes):
        parser.error("--modes must contain positive integers")
    if any(s < 0 for s in args.steps):
        parser.error("--steps must contain nonnegative integers")
    if args.repeats <= 0 or args.extra_warmups < 0 or args.graph_chunk <= 0:
        parser.error("--repeats and --graph-chunk must be positive; --extra-warmups nonnegative")
    return args


def main() -> None:
    args = _parse_args()
    include_gpu = "gpu" in args.backends
    metadata = _software_hardware(include_gpu)
    rows: list[dict[str, object]] = []

    for n_modes in args.modes:
        params = LLEParams(n_modes=n_modes, d3=args.d3)
        seed = initial_field(params)
        for steps in args.steps:
            final_by_backend: dict[str, np.ndarray] = {}
            current_rows: dict[str, dict[str, object]] = {}
            for backend in args.backends:
                gpu = backend == "gpu"
                if gpu:
                    solver = lambda x, p, s: simulate_gpu(
                        x, p, s, graph_chunk=args.graph_chunk
                    )
                else:
                    solver = simulate_numpy

                first_call, final = _timed_call(solver, seed, params, steps, gpu=gpu)
                for _ in range(args.extra_warmups):
                    _timed_call(solver, seed, params, steps, gpu=gpu)
                samples = []
                for _ in range(args.repeats):
                    elapsed, final = _timed_call(solver, seed, params, steps, gpu=gpu)
                    samples.append(elapsed)

                final_by_backend[backend] = final
                row: dict[str, object] = {
                    "n_modes": n_modes,
                    "steps": steps,
                    "backend": backend,
                    "first_call_seconds": first_call,
                    "repeat_seconds": samples,
                    "median_seconds": median(samples),
                }
                rows.append(row)
                current_rows[backend] = row

            cpu = final_by_backend.get("numpy")
            gpu_field = final_by_backend.get("gpu")
            if cpu is not None and gpu_field is not None:
                cpu_row = current_rows["numpy"]
                gpu_row = current_rows["gpu"]
                gpu_row["speedup_vs_numpy"] = cpu_row["median_seconds"] / gpu_row["median_seconds"]
                gpu_row.update(_accuracy(gpu_field, cpu))

            for row in current_rows.values():
                speedup = row.get("speedup_vs_numpy")
                speedup_text = f"{speedup:.2f}x" if speedup is not None else "-"
                print(f"N={n_modes:4d} steps={steps:6d} {row['backend']:5s} "
                      f"first={row['first_call_seconds']:.4f}s "
                      f"median={row['median_seconds']:.4f}s speedup={speedup_text}",
                      flush=True)

    parameters = asdict(LLEParams(d3=args.d3))
    parameters.pop("n_modes")
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "protocol": {
            "input_and_output": "host angular-domain complex field",
            "splitting": "first-order Lie; full linear/pump step followed by full Kerr step",
            "gpu_linear_evolution": "precomputed exp(lambda*h), then zero-mode pump=NFh (first-order approximation); followed by IFFT and Kerr",
            "gpu_state_domain": "angular",
            "first_call": "recorded separately; can include initialization, compilation, graph capture",
            "warm_repeats": args.repeats,
            "extra_warmups": args.extra_warmups,
            "graph_chunk": args.graph_chunk,
            "graph_execution": "positive steps use chunk=min(graph_chunk, steps); zero steps return a copy without graph capture",
            "gpu_dtype": "complex64",
            "numpy_dtype": "complex128",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
