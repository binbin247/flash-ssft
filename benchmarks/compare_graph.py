"""Compare a benchmark-only eager loop with the cached or fresh GPU solver.

Run from the project root:
    python benchmarks/compare_graph.py --repeats 5

CUDA/Triton are warmed before measurements. For positive steps, "graph_fresh"
includes graph construction, but is not a cold process or compiler measurement.
All timings include host input/output transfers and CUDA synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import triton

from flashlle import LLEParams, initial_field
from flashlle.gpu import clear_graph_cache, linear_coefficients, lle_step, simulate_gpu
from flashlle._validation import check_input


def _simulate_eager(seed, params, steps):
    """Run the same LLE steps without CUDA Graphs for this benchmark only."""
    field = check_input(seed, params.n_modes, steps)
    device = torch.device("cuda", torch.cuda.current_device())
    angular = torch.as_tensor(field, dtype=torch.complex64, device=device).contiguous()
    if steps == 0:
        return angular.cpu().numpy().copy()
    linear, pump = linear_coefficients(params)
    linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
    for _ in range(steps):
        angular = lle_step(angular, linear_gpu, pump, params.h)
    return angular.cpu().numpy().copy()


def timed(seed, params, steps, *, chunk, eager_baseline=False):
    torch.cuda.synchronize()
    start = time.perf_counter()
    if eager_baseline:
        final = _simulate_eager(seed, params, steps)
    else:
        final = simulate_gpu(seed, params, steps, graph_chunk=chunk)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if final.shape != seed.shape or final.dtype != np.complex64:
        raise ValueError("Unexpected output shape or dtype")
    if not np.isfinite(final).all():
        raise FloatingPointError("Non-finite solver output")
    return elapsed, final


def accuracy(actual, reference):
    delta = actual.astype(np.complex128) - reference.astype(np.complex128)
    scale = max(float(np.linalg.norm(reference.astype(np.complex128))), 1e-30)
    return {
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "field_relative_l2": float(np.linalg.norm(delta) / scale),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def snapshot():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,"
             "utilization.gpu,memory.used,clocks.sm,pstate", "--format=csv"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_hashes(*benchmark_paths: str) -> dict[str, str]:
    """Record all solver modules, including operators added as separate files."""
    paths = sorted((PROJECT_ROOT / "src" / "flashlle").rglob("*.py"))
    paths.extend(PROJECT_ROOT / name for name in benchmark_paths)
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[15, 16, 17, 100, 1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-chunk", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "compare_graph.json")
    args = parser.parse_args()
    if any(n < 2 for n in args.modes) or any(s < 0 for s in args.steps):
        parser.error("modes must be >= 2 and steps must be >= 0")
    if args.repeats < 1 or args.graph_chunk < 1:
        parser.error("repeats and graph-chunk must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA")

    variants = ["no_graph", "graph_cached", "graph_fresh"]
    params_dict = asdict(LLEParams())
    params_dict.pop("n_modes")
    result = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_snapshot_start": snapshot(),
            "source_sha256": source_hashes("benchmarks/compare_graph.py"),
        },
        "parameters": params_dict,
        "protocol": {
            "input_and_output": "host angular-domain complex field",
            "timing": "perf_counter with CUDA synchronization before and after",
            "splitting": "first-order Lie; full linear/pump step followed by full Kerr step",
            "gpu_linear_evolution": "precomputed exp(lambda*h), then zero-mode pump=NFh (first-order approximation); followed by IFFT and Kerr",
            "state_domain": "angular",
            "gpu_dtype": "complex64",
            "graph_chunk": args.graph_chunk,
            "repeats": args.repeats,
            "warmup_steps_per_mode": max(1000, args.graph_chunk),
            "no_graph": "benchmark-only eager loop; CUDA, cuFFT and Triton already warm",
            "graph_cached": "public simulate_gpu; previously captured graph at min(graph_chunk, steps)",
            "graph_fresh": "clear graph cache before timing; positive-step call includes capture",
            "order": "rotate variants each repetition to reduce ordering bias",
            "accuracy": "compare both graph outputs with no_graph each repetition",
            "note": "positive-step solver calls use graphs with chunk=min(graph_chunk, steps); zero steps copy the input without capture",
        },
        "warmup_seconds": {},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for n_modes in args.modes:
        clear_graph_cache()
        params = LLEParams(n_modes=n_modes)
        seed = initial_field(params)
        warmup_steps = max(1000, args.graph_chunk)
        warmup = {}
        for eager_baseline in (True, False):
            elapsed, _ = timed(seed, params, warmup_steps,
                               eager_baseline=eager_baseline, chunk=args.graph_chunk)
            warmup["no_graph" if eager_baseline else "graph"] = elapsed
        result["warmup_seconds"][str(n_modes)] = warmup

        for steps in args.steps:
            samples = {variant: [] for variant in variants}
            errors = {variant: [] for variant in variants if variant != "no_graph"}
            orders = []
            for repeat in range(args.repeats):
                offset = repeat % len(variants)
                order = variants[offset:] + variants[:offset]
                orders.append(order)
                outputs = {}
                for variant in order:
                    if variant == "graph_fresh":
                        # Cache release itself is outside the measured interval.
                        clear_graph_cache()
                    elif variant == "graph_cached":
                        # Ensure the cache exists even for arbitrary CLI step order.
                        simulate_gpu(seed, params, min(args.graph_chunk, steps),
                                     graph_chunk=args.graph_chunk)
                    elapsed, outputs[variant] = timed(
                        seed, params, steps, eager_baseline=variant == "no_graph",
                        chunk=args.graph_chunk,
                    )
                    samples[variant].append(elapsed)
                for variant in errors:
                    error = accuracy(outputs[variant], outputs["no_graph"])
                    errors[variant].append(error)
                    if error["field_relative_l2"] > 1e-5:
                        raise AssertionError(f"Graph output mismatch: {error}")

            summary = {
                variant: {
                    "repeat_seconds": values,
                    "median_seconds": median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                }
                for variant, values in samples.items()
            }
            baseline = summary["no_graph"]["median_seconds"]
            for variant in errors:
                summary[variant]["speedup_vs_no_graph"] = (
                    baseline / summary[variant]["median_seconds"]
                )
                summary[variant]["accuracy"] = {
                    "all_bitwise_equal": all(e["bitwise_equal"] for e in errors[variant]),
                    "max_field_relative_l2": max(e["field_relative_l2"] for e in errors[variant]),
                    "max_absolute_error": max(e["max_absolute_error"] for e in errors[variant]),
                }
            result["rows"].append({
                "n_modes": n_modes,
                "steps": steps,
                "graph_active": steps > 0,
                "effective_graph_chunk": min(args.graph_chunk, steps),
                "execution_orders": orders,
                **summary,
            })
            save()
            cached = summary["graph_cached"]
            fresh = summary["graph_fresh"]
            print(
                f"N={n_modes:5d} steps={steps:7d} "
                f"no_graph={baseline * 1000:10.3f}ms "
                f"cached={cached['median_seconds'] * 1000:10.3f}ms "
                f"fresh={fresh['median_seconds'] * 1000:10.3f}ms "
                f"speedup={cached['speedup_vs_no_graph']:.2f}x "
                f"equal={cached['accuracy']['all_bitwise_equal']}",
                flush=True,
            )
    result["metadata"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["gpu_snapshot_end"] = snapshot()
    save()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
