"""Compare ordinary PyTorch and Triton operators on resident complex64 CUDA data.

Example: python benchmarks/benchmark_operators.py --operators kerr linear_and_pump
    --sizes 128 257 --repeats 2 --chunk 4 --calls 16 --output /tmp/operators

--output is a prefix: /tmp/operators.json, .png and .svg are written. The default
prefix includes a timestamp, leaving historical benchmark results untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median

import matplotlib
import numpy as np
import torch
import triton

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashlle.operators import kerr_, linear_and_pump_
from _torch_reference import torch_kerr_, torch_linear_and_pump_

VARIANTS = ("torch_eager", "triton_eager", "torch_graph", "triton_graph")
LABELS = ("PyTorch eager", "Triton eager", "PyTorch CUDA Graph", "Triton CUDA Graph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", nargs="+", choices=("kerr", "linear_and_pump"),
                        default=["kerr", "linear_and_pump"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[2**p for p in range(7, 21)])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--calls", type=int, default=None,
                        help="Total calls per repetition; must be divisible by chunk. "
                             "Default adapts to N (64-1024 with the default chunk).")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--h", type=float, default=0.005)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"benchmark_operators_{stamp}",
                        help="Output prefix, without .json/.png/.svg")
    args = parser.parse_args(argv)
    if any(n < 1 for n in args.sizes) or min(args.repeats, args.chunk, args.warmup) < 1:
        parser.error("sizes, repeats, chunk and warmup must be positive")
    if args.calls is not None and (args.calls < 1 or args.calls % args.chunk):
        parser.error("calls must be positive and divisible by chunk")
    if not np.isfinite(args.h) or args.h < 0:
        parser.error("h must be finite and nonnegative")
    args.operators = list(dict.fromkeys(args.operators))
    args.sizes = sorted(set(args.sizes))
    return args


def metadata():
    paths = sorted((ROOT / "src" / "flashlle").rglob("*.py"))
    paths.extend(ROOT / "benchmarks" / name for name in (
        "_torch_reference.py", "benchmark_operators.py", "benchmark_kerr.py"
    ))
    device = torch.cuda.current_device()
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "triton": triton.__version__,
        "torch_cuda": torch.version.cuda, "device_index": device,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
    }


def formula(operator, seed, linear, pump, h, calls):
    """Closed-form complex128 check, independent of GPU kernels and Python loops."""
    field = seed.astype(np.complex128)
    if operator == "kerr":
        return field * np.exp(1j * h * calls * np.abs(field) ** 2)
    factors = linear.astype(np.complex128)
    evolved = factors**calls
    field *= evolved
    pump_sum = calls if factors[0] == 1 else (evolved[0] - 1) / (factors[0] - 1)
    field[0] += pump * pump_sum
    return field


def capture(operation, state, seed, chunk):
    """Capture already-warmed in-place operations; preparation is never timed."""
    state.copy_(seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(chunk):
            operation()
    state.copy_(seed)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(operation, state, seed, invocations):
    state.copy_(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(invocations):
        operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def benchmark(operator, seed, args):
    n = seed.size
    calls = args.calls or max(args.chunk, min(1024, (1 << 22) // n) // args.chunk * args.chunk)
    linear = np.exp((-0.2 - 0.7j) * (1 + np.arange(n) / n) * args.h).astype(np.complex64)
    pump = (0.125 - 0.375j) * args.h
    seed_gpu = torch.as_tensor(seed, device="cuda")
    linear_gpu = torch.as_tensor(linear, device="cuda")
    states = [seed_gpu.clone() for _ in VARIANTS]
    if operator == "kerr":
        functions = (torch_kerr_, kerr_, torch_kerr_, kerr_)
        operations = [partial(fn, state, args.h) for fn, state in zip(functions, states)]
    else:
        functions = (torch_linear_and_pump_, linear_and_pump_) * 2
        operations = [partial(fn, state, linear_gpu, pump) for fn, state in zip(functions, states)]

    single = formula(operator, seed, linear, pump, args.h, 1)
    for operation, state in zip(operations[:2], states[:2]):
        assert operation() is state
        np.testing.assert_allclose(state.cpu().numpy(), single, rtol=2e-6, atol=2e-7)

    # Both backends warm on a separate stream before either graph is captured.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for operation, state in zip(operations, states):
            state.copy_(seed_gpu)
            for _ in range(args.warmup):
                operation()
    torch.cuda.current_stream().wait_stream(stream)
    graphs = [capture(operations[i], states[i], seed_gpu, args.chunk) for i in (2, 3)]
    timed_operations = operations[:2] + [graph.replay for graph in graphs]
    samples = {variant: [] for variant in VARIANTS}
    for repeat in range(args.repeats):
        for offset in range(len(VARIANTS)):
            i = (repeat + offset) % len(VARIANTS)
            invocations = calls // args.chunk if i >= 2 else calls
            seconds = measure(timed_operations[i], states[i], seed_gpu, invocations)
            samples[VARIANTS[i]].append(seconds * 1e6 / calls)

    for eager, graphed in ((0, 2), (1, 3)):
        torch.testing.assert_close(states[eager], states[graphed], rtol=1e-6, atol=1e-7)
    expected = formula(operator, seed, linear, pump, args.h, calls)
    errors = {}
    for variant, state in zip(VARIANTS, states):
        actual = state.cpu().numpy().astype(np.complex128)
        relative_l2 = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
        if not np.isfinite(actual).all() or relative_l2 > 1e-3:
            raise AssertionError(f"{operator}, N={n}, {variant}: formula relative L2={relative_l2}")
        errors[variant] = relative_l2
    return {
        "operator": operator, "N": n, "calls_per_repeat": calls,
        "graph_replays_per_repeat": calls // args.chunk,
        "samples_us_per_call": samples,
        "median_us_per_call": {variant: median(values) for variant, values in samples.items()},
        "formula_relative_l2": errors,
    }


def plot(rows, args, gpu):
    fig, axes = plt.subplots(1, len(args.operators), figsize=(7 * len(args.operators), 5), squeeze=False)
    for ax, operator in zip(axes[0], args.operators):
        selected = [row for row in rows if row["operator"] == operator]
        sizes = [row["N"] for row in selected]
        for variant, label in zip(VARIANTS, LABELS):
            ax.plot(sizes, [row["median_us_per_call"][variant] for row in selected], marker="o", label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes], rotation=45, ha="right")
        ax.set(xlabel="Array size N", ylabel="Time per operator call (µs)", title=operator)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"PyTorch vs Triton · {gpu}")
    fig.text(0.5, 0.01, f"complex64 · median of {args.repeats} repeats · {args.chunk} calls per graph replay\n"
             "Resident GPU data; transfers, reset, warmup and capture excluded; host dispatch included",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for suffix in (".png", ".svg"):
        fig.savefig(str(args.output) + suffix, dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    result = {
        "metadata": metadata(), "complete": False,
        "protocol": {
            "dtype": "complex64", "device_residency": "GPU for every timed variant",
            "timing": "perf_counter with CUDA synchronization before and after",
            "excluded": ["transfers", "input reset", "coefficient preparation", "compilation", "warmup", "graph capture"],
            "included": "host dispatch and GPU execution of the same total operator count",
            "order": "rotate all four variants each repetition",
            "h": args.h, "repeats": args.repeats, "graph_chunk": args.chunk,
            "warmup_calls_per_variant": args.warmup, "seed": 20261020,
            "linear_coefficients": "exp((-0.2-0.7j)*(1+index/N)*h)",
            "pump": "(0.125-0.375j)*h, added to the zero mode",
            "single_call_check": {"numpy_dtype": "complex128", "rtol": 2e-6, "atol": 2e-7},
            "repeated_formula_relative_l2_limit": 1e-3,
            "graph_vs_own_eager_check": {"rtol": 1e-6, "atol": 1e-7},
        },
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261020)
    for n in args.sizes:
        seed = (0.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
        for operator in args.operators:
            row = benchmark(operator, seed, args)
            result["rows"].append(row)
            timings = " ".join(f"{name}={value:.3f}µs" for name, value in row["median_us_per_call"].items())
            print(f"{operator} N={n} calls={row['calls_per_repeat']} {timings}", flush=True)
            Path(str(args.output) + ".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["complete"] = True
    result["metadata"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(str(args.output) + ".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result["rows"], args, result["metadata"]["gpu"])
    print(f"Saved {args.output}.json / .png / .svg", flush=True)


if __name__ == "__main__":
    main()
