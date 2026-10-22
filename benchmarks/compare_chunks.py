"""Sweep CUDA Graph chunk sizes using the actual public solver.

CUDA/cuFFT/Triton are warmed first. Setup timing measures CudaGraphStepper
construction (including its two warmup steps), with no graph replay. Cached
solver timing includes host input/output transfers and synchronization.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median

from compare_graph import PROJECT_ROOT, accuracy, snapshot, source_hashes, timed

import numpy as np
import torch
import triton

from flashlle import LLEParams, initial_field, linear_coefficients
from flashlle.gpu import CudaGraphStepper, clear_graph_cache, lle_step, simulate_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--chunks", nargs="+", type=int,
                        default=[1, 4, 16, 32, 64, 100, 128, 256])
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "compare_chunks.json")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.modes) < 2 or min(args.chunks) < 1 or min(args.steps) < 1:
        parser.error("repeats, chunks and steps must be positive; modes must be >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    result = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "numpy": np.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_snapshot_start": snapshot(),
            "source_sha256": source_hashes(
                "benchmarks/compare_chunks.py", "benchmarks/compare_graph.py"
            ),
        },
        "protocol": {
            "repeats": args.repeats,
            "chunks": args.chunks,
            "steps": args.steps,
            "timing": "perf_counter with CUDA synchronization before and after",
            "splitting": "first-order Lie; full linear/pump step followed by full Kerr step",
            "gpu_linear_evolution": "precomputed exp(lambda*h), then zero-mode pump=NFh (first-order approximation); followed by IFFT and Kerr",
            "state_domain": "angular",
            "setup": "CudaGraphStepper constructor at the requested chunk, independent of run steps; no replay; libraries already warm",
            "cached": "public simulate_gpu call with a cached graph; host-to-host",
            "memory": "PyTorch reserved-memory increase during constructor, after empty_cache; excludes allocations outside PyTorch allocator",
            "accuracy_reference": "same solver with graph_chunk=16",
            "ordering": "deterministically shuffled chunks each repeat",
            "graph_activation": "all positive-step calls; effective chunk=min(chunk, steps)",
            "remainder": "steps % effective_chunk eager steps are included",
        },
        "orders": [],
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    rng = random.Random(20261020)
    for n_modes in args.modes:
        clear_graph_cache()
        params = LLEParams(n_modes=n_modes)
        seed = initial_field(params)
        simulate_gpu(seed, params, 1000)
        references = {
            steps: simulate_gpu(seed, params, steps, graph_chunk=16)
            for steps in args.steps
        }
        clear_graph_cache()
        device = torch.device("cuda", torch.cuda.current_device())
        seed_gpu = torch.as_tensor(seed, dtype=torch.complex64, device=device)
        linear, pump = linear_coefficients(params)
        linear_gpu = torch.as_tensor(linear, dtype=torch.complex64, device=device)
        step = partial(lle_step, linear=linear_gpu, pump=pump, h=params.h)
        rows = {}
        for chunk in args.chunks:
            row = {
                "n_modes": n_modes,
                "chunk": chunk,
                "setup_seconds": [],
                "graph_reserved_bytes": [],
                "runs": [
                    {"steps": steps, "graph_active": True,
                     "effective_graph_chunk": min(chunk, steps),
                     "eager_remainder_steps": steps % min(chunk, steps),
                     "repeat_seconds": [], "exact_output_matches": [],
                     "relative_l2": []}
                    for steps in args.steps
                ],
            }
            rows[chunk] = row
            result["rows"].append(row)

        for repeat in range(args.repeats):
            order = list(args.chunks)
            rng.shuffle(order)
            result["orders"].append({"n_modes": n_modes, "repeat": repeat, "chunks": order})
            for chunk in order:
                clear_graph_cache()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                reserved_before = torch.cuda.memory_reserved()
                start = time.perf_counter()
                runner = CudaGraphStepper(seed_gpu, step, chunk)
                torch.cuda.synchronize()
                setup = time.perf_counter() - start
                reserved = torch.cuda.memory_reserved() - reserved_before
                rows[chunk]["setup_seconds"].append(setup)
                rows[chunk]["graph_reserved_bytes"].append(reserved)
                del runner
                gc.collect()
                torch.cuda.empty_cache()

                warmed_chunks = set()
                for run in rows[chunk]["runs"]:
                    # Warm each effective chunk, including shortened graphs.
                    effective_chunk = run["effective_graph_chunk"]
                    if effective_chunk not in warmed_chunks:
                        simulate_gpu(seed, params, effective_chunk, graph_chunk=chunk)
                        warmed_chunks.add(effective_chunk)
                    elapsed, final = timed(seed, params, run["steps"],
                                           chunk=chunk)
                    error = accuracy(final, references[run["steps"]])
                    run["repeat_seconds"].append(elapsed)
                    run["exact_output_matches"].append(error["bitwise_equal"])
                    run["relative_l2"].append(error["field_relative_l2"])
                    if error["field_relative_l2"] > 1e-5:
                        raise AssertionError(error)
                save()
            print(f"N={n_modes} repeat={repeat + 1}/{args.repeats} finished", flush=True)

        for row in rows.values():
            row["setup_median_seconds"] = median(row["setup_seconds"])
            row["graph_reserved_median_bytes"] = median(row["graph_reserved_bytes"])
            for run in row["runs"]:
                run["median_seconds"] = median(run["repeat_seconds"])
                run["min_seconds"] = min(run["repeat_seconds"])
                run["max_seconds"] = max(run["repeat_seconds"])
            timings = " ".join(
                f"{run['steps']}steps={run['median_seconds'] * 1000:.2f}ms"
                for run in row["runs"]
            )
            print(
                f"N={n_modes} chunk={row['chunk']:3d} "
                f"setup={row['setup_median_seconds'] * 1000:.2f}ms "
                f"reserved={row['graph_reserved_median_bytes'] / 2**20:.1f}MiB "
                f"{timings}", flush=True,
            )
        del seed_gpu, step, linear_gpu
        save()

    clear_graph_cache()
    result["metadata"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["gpu_snapshot_end"] = snapshot()
    save()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
