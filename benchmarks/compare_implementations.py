"""Compare Torch/Triton LLE operators with and without CUDA Graphs on one GPU.

All four variants use complex64 fields, the same precomputed coefficients and
cuFFT transforms. Timings include input reset and an independent output clone;
coefficient preparation, host transfers, warmup and graph capture are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import triton

from _torch_reference import torch_lle_step
from flashlle import LLEParams, initial_field, linear_coefficients
from flashlle.gpu import CudaGraphStepper, lle_step


VARIANTS = ("torch_eager", "triton_eager", "torch_graph", "triton_graph")
COMPARISONS = {
    "triton_eager_vs_torch_eager": ("triton_eager", "torch_eager"),
    "triton_graph_vs_torch_graph": ("triton_graph", "torch_graph"),
    "torch_graph_vs_torch_eager": ("torch_graph", "torch_eager"),
    "triton_graph_vs_triton_eager": ("triton_graph", "triton_eager"),
}


def run_eager(
    initial_state: torch.Tensor,
    steps: int,
    *,
    state: torch.Tensor,
    step: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Match CudaGraphStepper.run's input copy and independent output clone."""
    state.copy_(initial_state)
    angular = state
    for _ in range(steps):
        angular = step(angular)
    return angular.clone()


def timed_run(operation, seed: torch.Tensor, steps: int) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize(seed.device)
    start = time.perf_counter()
    output = operation(seed, steps)
    torch.cuda.synchronize(seed.device)
    return time.perf_counter() - start, output


def accuracy(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if (actual.shape != reference.shape or actual.dtype != np.complex64
            or reference.dtype != np.complex64):
        raise AssertionError("Unexpected output shape or dtype")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite LLE output")
    actual128 = actual.astype(np.complex128)
    reference128 = reference.astype(np.complex128)
    delta = actual128 - reference128
    scale = max(float(np.linalg.norm(reference128)), 1e-30)
    return {
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "field_relative_l2": float(np.linalg.norm(delta) / scale),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def source_hashes() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src" / "flashlle").rglob("*.py"))
    script = Path(__file__).resolve()
    paths.extend([script, script.with_name("_torch_reference.py")])
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def save_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--graph-chunk", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-3,
                        help="maximum field relative L2 error between Torch and Triton")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "compare_implementations.json")
    args = parser.parse_args()
    if min(args.modes) < 2 or min(args.steps) < 1:
        parser.error("modes must be >= 2 and steps must be positive")
    if args.graph_chunk < 1 or args.repeats < 1:
        parser.error("graph-chunk and repeats must be positive")
    if not np.isfinite(args.relative_l2_tolerance) or args.relative_l2_tolerance <= 0:
        parser.error("relative-l2-tolerance must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    parameters = asdict(LLEParams())
    parameters.pop("n_modes")
    result = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": properties.name,
            "device_index": device.index,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_gpu_memory_bytes": properties.total_memory,
            "source_sha256": source_hashes(),
        },
        "parameters": parameters,
        "protocol": {
            "modes": args.modes,
            "steps": args.steps,
            "variants": list(VARIANTS),
            "repeats": args.repeats,
            "graph_chunk": args.graph_chunk,
            "state_domain": "angular",
            "dtype": "complex64",
            "input_and_output": "GPU resident; identical seed and coefficients for every variant",
            "splitting": "FFT -> linear/pump -> IFFT -> Kerr, first-order Lie splitting",
            "linear_evolution": "precomputed exp(lambda*h), then zero-mode pump=N*F*h",
            "torch_operators": "ordinary PyTorch pointwise operations, without torch.compile",
            "triton_operators": "production fused linear_and_pump_ and kerr_",
            "timing": "perf_counter with device synchronization before and after each run",
            "included": "input state.copy_, all LLE steps, output.clone, final synchronization",
            "excluded": "coefficient preparation, host transfers, persistent input-state allocation, warmup, graph capture, accuracy checks",
            "graph_execution": "production CudaGraphStepper; effective chunk=min(graph_chunk, steps); eager remainder",
            "graph_boundary_copy": "captured copy into persistent state after every graph chunk is included",
            "warmup_runs_per_variant_per_case": 2,
            "ordering": "rotate all four variants across repetitions and cases; record each order",
            "accuracy": "Torch/Triton comparisons at each execution mode; graph/eager comparisons within each implementation",
            "torch_triton_relative_l2_tolerance": args.relative_l2_tolerance,
            "graph_eager_relative_l2_tolerance": 1e-5,
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for n_modes in args.modes:
        params = LLEParams(n_modes=n_modes)
        seed_gpu = torch.as_tensor(initial_field(params), dtype=torch.complex64, device=device)
        linear, pump = linear_coefficients(params)
        linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
        torch_step = partial(torch_lle_step, linear=linear_gpu, pump=pump, h=params.h)
        triton_step = partial(lle_step, linear=linear_gpu, pump=pump, h=params.h)

        for steps in args.steps:
            chunk = min(args.graph_chunk, steps)
            torch_runner = CudaGraphStepper(seed_gpu, torch_step, chunk)
            triton_runner = CudaGraphStepper(seed_gpu, triton_step, chunk)
            operations = {
                "torch_eager": partial(run_eager, state=seed_gpu.clone(), step=torch_step),
                "triton_eager": partial(run_eager, state=seed_gpu.clone(), step=triton_step),
                "torch_graph": torch_runner.run,
                "triton_graph": triton_runner.run,
            }
            for operation in operations.values():
                for _ in range(2):
                    operation(seed_gpu, steps)
            torch.cuda.synchronize(device)

            samples = {variant: [] for variant in VARIANTS}
            errors = {comparison: [] for comparison in COMPARISONS}
            orders = []
            failures = []
            for repeat in range(args.repeats):
                offset = (len(result["rows"]) + repeat) % len(VARIANTS)
                order = VARIANTS[offset:] + VARIANTS[:offset]
                orders.append(list(order))
                outputs = {}
                for variant in order:
                    elapsed, outputs[variant] = timed_run(operations[variant], seed_gpu, steps)
                    samples[variant].append(elapsed)
                host_outputs = {variant: output.cpu().numpy() for variant, output in outputs.items()}
                for comparison, (actual, reference) in COMPARISONS.items():
                    error = accuracy(host_outputs[actual], host_outputs[reference])
                    errors[comparison].append(error)
                    tolerance = (args.relative_l2_tolerance if comparison.startswith("triton_")
                                 and "vs_torch" in comparison else 1e-5)
                    if error["field_relative_l2"] > tolerance:
                        failures.append(f"repeat={repeat} {comparison}: {error['field_relative_l2']:.6g} > {tolerance}")
                del outputs, host_outputs

            timings = {
                variant: {
                    "repeat_seconds": values,
                    "median_seconds": median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                }
                for variant, values in samples.items()
            }
            medians = {variant: timing["median_seconds"] for variant, timing in timings.items()}
            result["rows"].append({
                "n_modes": n_modes,
                "steps": steps,
                "effective_graph_chunk": chunk,
                "graph_replays": steps // chunk,
                "eager_remainder_steps": steps % chunk,
                "execution_orders": orders,
                "timings": timings,
                "accuracy": errors,
                "validation_passed": not failures,
                "validation_failures": failures,
                "speedup": {
                    "triton_vs_torch_eager": medians["torch_eager"] / medians["triton_eager"],
                    "triton_vs_torch_graph": medians["torch_graph"] / medians["triton_graph"],
                    "graph_vs_eager_torch": medians["torch_eager"] / medians["torch_graph"],
                    "graph_vs_eager_triton": medians["triton_eager"] / medians["triton_graph"],
                    "triton_graph_vs_torch_eager": medians["torch_eager"] / medians["triton_graph"],
                },
            })
            save_result(args.output, result)
            if failures:
                raise AssertionError(f"N={n_modes}, steps={steps}: " + "; ".join(failures))
            timings_text = " ".join(f"{name}={medians[name] * 1000:.3f}ms" for name in VARIANTS)
            print(f"N={n_modes} steps={steps} {timings_text}", flush=True)
            del operations, operation, torch_runner, triton_runner

    result["metadata"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_result(args.output, result)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
