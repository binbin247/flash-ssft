"""Run the NLSE correctness and GPU tuning experiments.

The script intentionally keeps the experiment protocol in one place. Run it
from the repository root, or pass ``--output-dir`` explicitly. GPU phases use
the current CUDA device; the server wrapper binds host GPU 6 as CUDA device 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from examples.run_tod import example_params, initial_field
from flashlle.solvers import (
    NLSEParams,
    angular_frequencies,
    linear_coefficients,
    simulate_numpy,
)
from flashlle.solvers.nlse_gpu_opt import CudaGraphStepper, nlse_step, simulate_gpu
from flashlle.solvers.nlse_pytorch_baseline import pytorch_nlse_step, simulate_pytorch


DEFAULT_SAMPLES = (2**10, 2**11, 2**12, 2**13)
DEFAULT_TAIL_SAMPLES = (257, 1025, 4097)
DEFAULT_BLOCKS = (64, 128, 256, 512)
DEFAULT_WARPS = (1, 2, 4, 8)
DEFAULT_CHUNKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_STEPS = 4096
DEFAULT_REPEATS = 7
DEFAULT_WARMUPS = 2


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_hashes() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "examples").rglob("*.py"))
    paths += [Path(__file__), PROJECT_ROOT / "pyproject.toml"]
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.exists()
    }


def metadata() -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    info: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_index": device.index,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "source_sha256": source_hashes(),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # pragma: no cover - environment reporting only
        info["triton_error"] = f"{type(exc).__name__}: {exc}"
    return info


def make_params(n_samples: int, dz: float = 0.001) -> NLSEParams:
    params = example_params(n_samples)
    if dz == params.dz:
        return params
    return NLSEParams(
        n_samples=params.n_samples,
        dt=params.dt,
        dz=dz,
        alpha=params.alpha,
        beta2=params.beta2,
        beta3=params.beta3,
        beta4=params.beta4,
        gamma0=params.gamma0,
        omega0=params.omega0,
        beta_higher=params.beta_higher,
    )


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual128 = np.asarray(actual, dtype=np.complex128)
    reference128 = np.asarray(reference, dtype=np.complex128)
    return float(np.linalg.norm(actual128 - reference128)
                 / max(float(np.linalg.norm(reference128)), 1e-30))


def accuracy(actual: np.ndarray, reference: np.ndarray, dt: float) -> dict[str, Any]:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    delta = actual.astype(np.complex128) - reference.astype(np.complex128)
    reference128 = reference.astype(np.complex128)
    ref_norm = max(float(np.linalg.norm(reference128)), 1e-30)
    power = np.abs(actual) ** 2
    ref_power = np.abs(reference) ** 2
    spectrum = np.abs(np.fft.fft(actual)) ** 2
    ref_spectrum = np.abs(np.fft.fft(reference)) ** 2
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "finite": bool(np.isfinite(actual).all()),
        "relative_l2": float(np.linalg.norm(delta) / ref_norm),
        "max_absolute_error": float(np.max(np.abs(delta))) if actual.size else 0.0,
        "mean_power_relative_error": float(
            abs(float(np.mean(power)) - float(np.mean(ref_power)))
            / max(float(np.mean(ref_power)), 1e-30)
        ),
        "energy_relative_error": float(
            abs(float(dt * np.sum(power)) - float(dt * np.sum(ref_power)))
            / max(float(dt * np.sum(ref_power)), 1e-30)
        ),
        "spectrum_relative_l2": float(
            np.linalg.norm(spectrum.astype(np.float64) - ref_spectrum.astype(np.float64))
            / max(float(np.linalg.norm(ref_spectrum)), 1e-30)
        ),
    }


def _cuda_time(operation: Callable[[], Any]) -> tuple[float, Any]:
    """Time an operation with CUDA events, including its output clone/copy."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0, output


def _wall_time(operation: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    output = operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start, output


def _run_eager(
    initial: torch.Tensor,
    steps: int,
    step: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> torch.Tensor:
    state.copy_(initial)
    current = state
    for _ in range(steps):
        current = step(current)
    return current.clone()


def _gpu_context(params: NLSEParams, seed: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = torch.device("cuda", torch.cuda.current_device())
    seed_gpu = torch.as_tensor(seed, dtype=torch.complex64, device=device).contiguous()
    linear_gpu = torch.as_tensor(linear_coefficients(params), dtype=torch.complex64, device=device)
    w_gpu = torch.as_tensor(angular_frequencies(params), dtype=torch.float64, device=device)
    return seed_gpu, linear_gpu, w_gpu, float(params.gamma0 * params.dz)


def _make_step(
    params: NLSEParams,
    linear: torch.Tensor,
    w: torch.Tensor,
    backend: str,
    block: int = 256,
    num_warps: int = 4,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if backend == "pytorch":
        return partial(
            pytorch_nlse_step,
            linear=linear,
            w=w,
            h=params.gamma0 * params.dz,
            omega0=params.omega0,
        )
    if backend == "triton":
        return partial(
            nlse_step,
            linear=linear,
            w=w,
            h=params.gamma0 * params.dz,
            omega0=params.omega0,
            block=block,
            num_warps=num_warps,
        )
    raise ValueError(f"Unknown backend: {backend}")


def _warmup(
    operation: Callable[[], Any], count: int
) -> None:
    for _ in range(count):
        operation()
    torch.cuda.synchronize()


def _stats(times: list[float], steps: int, n_samples: int) -> dict[str, float]:
    value = median(times)
    return {
        "repeat_seconds": times,
        "steady_median_seconds": value,
        "steady_min_seconds": min(times),
        "steady_max_seconds": max(times),
        "steady_steps_per_second": steps / value,
        "steady_samples_steps_per_second": n_samples * steps / value,
    }


def _memory() -> dict[str, int]:
    return {
        "memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _base_record(params: NLSEParams, n_samples: int, steps: int) -> dict[str, Any]:
    return {
        "n_samples": n_samples,
        "steps": steps,
        "dt": params.dt,
        "dz": params.dz,
        "target_z": steps * params.dz,
        "parameters": asdict(params),
        "dtype": "complex64",
    }


def run_correctness(
    samples: Iterable[int],
    tail_samples: Iterable[int],
    output_dir: Path,
    *,
    block: int,
    num_warps: int,
    graph_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_cache: dict[tuple[int, int], np.ndarray] = {}
    cases = [(n, s) for n in samples for s in (0, 1, 16, 4096)]
    cases += [(n, s) for n in tail_samples for s in (1, 16, 4096)]
    for n_samples, steps in cases:
        params = make_params(n_samples)
        seed = initial_field(params)
        original = seed.copy()
        key = (n_samples, steps)
        if key not in reference_cache:
            print(f"[correctness] NumPy reference N={n_samples} steps={steps}", flush=True)
            reference_cache[key] = simulate_numpy(seed, params, steps)
        reference = reference_cache[key]

        torch.cuda.reset_peak_memory_stats()
        pytorch_start = time.perf_counter()
        pytorch_output = simulate_pytorch(seed, params, steps)
        torch.cuda.synchronize()
        pytorch_seconds = time.perf_counter() - pytorch_start

        torch.cuda.reset_peak_memory_stats()
        optimized_start = time.perf_counter()
        optimized_output = simulate_gpu(
            seed, params, steps, graph_chunk=graph_chunk,
            block=block, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        optimized_seconds = time.perf_counter() - optimized_start

        assert np.array_equal(seed, original), "solver modified host input"
        row = _base_record(params, n_samples, steps)
        row.update({
            "phase": "correctness",
            "block": block,
            "num_warps": num_warps,
            "graph_chunk": graph_chunk,
            "pytorch_seconds": pytorch_seconds,
            "optimized_seconds": optimized_seconds,
            "pytorch_accuracy": accuracy(pytorch_output, reference, params.dt),
            "optimized_accuracy": accuracy(optimized_output, reference, params.dt),
            "optimized_vs_pytorch_relative_l2": relative_l2(optimized_output, pytorch_output),
            "pytorch_input_preserved": bool(np.array_equal(seed, original)),
            "optimized_input_preserved": bool(np.array_equal(seed, original)),
            "memory": _memory(),
        })
        row["passed"] = bool(
            row["pytorch_accuracy"]["finite"]
            and row["optimized_accuracy"]["finite"]
            and row["pytorch_accuracy"]["relative_l2"] <= (1e-3 if steps >= 4096 else 1e-5)
            and row["optimized_accuracy"]["relative_l2"] <= (1e-3 if steps >= 4096 else 1e-5)
        )
        rows.append(row)
        print(
            f"[correctness] N={n_samples} steps={steps} "
            f"torch={row['pytorch_accuracy']['relative_l2']:.3e} "
            f"triton={row['optimized_accuracy']['relative_l2']:.3e} "
            f"passed={row['passed']}",
            flush=True,
        )
    return rows


def _measure_candidate(
    params: NLSEParams,
    seed: np.ndarray,
    steps: int,
    block: int,
    num_warps: int,
    graph_chunk: int,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    seed_gpu, linear_gpu, w_gpu, _ = _gpu_context(params, seed)
    torch_step = _make_step(params, linear_gpu, w_gpu, "pytorch")
    triton_step = _make_step(params, linear_gpu, w_gpu, "triton", block, num_warps)
    torch_state = seed_gpu.clone()
    triton_state = seed_gpu.clone()

    with torch.no_grad():
        torch_warmup = lambda: _run_eager(seed_gpu, steps, torch_step, torch_state)
        triton_warmup = lambda: _run_eager(seed_gpu, steps, triton_step, triton_state)
        _warmup(torch_warmup, warmups)

        torch_times = [_cuda_time(torch_warmup)[0] for _ in range(repeats)]

        compile_start = time.perf_counter()
        _cuda_time(lambda: triton_step(seed_gpu.clone()))
        compile_seconds = time.perf_counter() - compile_start
        capture_start = time.perf_counter()
        runner = CudaGraphStepper(seed_gpu, triton_step, graph_chunk)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - capture_start

        verify_steps = min(graph_chunk, steps)
        expected = _run_eager(seed_gpu, verify_steps, torch_step, torch_state)
        actual = runner.run(seed_gpu, verify_steps)
        verification_relative_l2 = relative_l2(
            actual.detach().cpu().numpy(), expected.detach().cpu().numpy()
        )
        if not np.isfinite(actual.detach().cpu().numpy()).all() or verification_relative_l2 > 1e-3:
            raise ValueError(
                f"candidate correctness failed: relative_l2={verification_relative_l2:.6g}"
            )

        graph_operation = lambda: runner.run(seed_gpu, steps)
        _warmup(graph_operation, warmups)
        graph_times = [_cuda_time(graph_operation)[0] for _ in range(repeats)]
        triton_times = [_cuda_time(triton_warmup)[0] for _ in range(repeats)]
        torch.cuda.synchronize()

    row: dict[str, Any] = {
        "phase": "kernel_search",
        "n_samples": params.n_samples,
        "steps": steps,
        "block": block,
        "num_warps": num_warps,
        "graph_chunk": graph_chunk,
        "compile_seconds": compile_seconds,
        "capture_seconds": capture_seconds,
        "verification_steps": verify_steps,
        "verification_relative_l2": verification_relative_l2,
        "full_chunks": steps // graph_chunk,
        "remainder_steps": steps % graph_chunk,
        "pytorch_eager": _stats(torch_times, steps, params.n_samples),
        "triton_eager": _stats(triton_times, steps, params.n_samples),
        "triton_graph": _stats(graph_times, steps, params.n_samples),
        "memory": _memory(),
    }
    del runner, torch_state, triton_state, seed_gpu, linear_gpu, w_gpu
    torch.cuda.empty_cache()
    return row


def run_kernel_search(
    samples: Iterable[int],
    output_dir: Path,
    *,
    blocks: Iterable[int],
    warps: Iterable[int],
    steps: int,
    graph_chunk: int,
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_samples in samples:
        params = make_params(n_samples)
        seed = initial_field(params)
        for block in blocks:
            for num_warps in warps:
                print(
                    f"[kernel-search] N={n_samples} block={block} warps={num_warps}",
                    flush=True,
                )
                try:
                    row = _measure_candidate(
                        params, seed, steps, block, num_warps, graph_chunk,
                        repeats, warmups,
                    )
                    row["status"] = "ok"
                except Exception as exc:
                    row = {
                        "phase": "kernel_search",
                        "n_samples": n_samples,
                        "steps": steps,
                        "block": block,
                        "num_warps": num_warps,
                        "graph_chunk": graph_chunk,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    print(f"[kernel-search] failed: {row['error']}", flush=True)
                rows.append(row)
    return rows


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def select_best_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["block"], row["num_warps"])
        grouped.setdefault(key, []).append(row["triton_graph"]["steady_median_seconds"])
    if not grouped:
        raise RuntimeError("No successful kernel configuration")
    best = min(grouped.items(), key=lambda item: geometric_mean(item[1]))
    return {
        "block": best[0][0],
        "num_warps": best[0][1],
        "geometric_mean_seconds": geometric_mean(best[1]),
        "sample_count": len(best[1]),
    }


def _measure_chunk(
    params: NLSEParams,
    seed: np.ndarray,
    steps: int,
    block: int,
    num_warps: int,
    chunk: int,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    seed_gpu, linear_gpu, w_gpu, _ = _gpu_context(params, seed)
    triton_step = _make_step(params, linear_gpu, w_gpu, "triton", block, num_warps)
    compile_start = time.perf_counter()
    _cuda_time(lambda: triton_step(seed_gpu.clone()))
    compile_seconds = time.perf_counter() - compile_start
    capture_start = time.perf_counter()
    runner = CudaGraphStepper(seed_gpu, triton_step, chunk)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_start
    operation = lambda: runner.run(seed_gpu, steps)
    _warmup(operation, warmups)
    times = [_cuda_time(operation)[0] for _ in range(repeats)]
    result = {
        "phase": "chunk_search",
        "n_samples": params.n_samples,
        "steps": steps,
        "block": block,
        "num_warps": num_warps,
        "graph_chunk": chunk,
        "compile_seconds": compile_seconds,
        "capture_seconds": capture_seconds,
        "full_chunks": steps // chunk,
        "remainder_steps": steps % chunk,
        "triton_graph": _stats(times, steps, params.n_samples),
        "memory": _memory(),
        "status": "ok",
    }
    del runner, seed_gpu, linear_gpu, w_gpu
    torch.cuda.empty_cache()
    return result


def run_chunk_search(
    samples: Iterable[int],
    output_dir: Path,
    *,
    block: int,
    num_warps: int,
    chunks: Iterable[int],
    steps_list: Iterable[int],
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_samples in samples:
        params = make_params(n_samples)
        seed = initial_field(params)
        for steps in steps_list:
            for chunk in chunks:
                print(
                    f"[chunk-search] N={n_samples} steps={steps} chunk={chunk}",
                    flush=True,
                )
                try:
                    rows.append(_measure_chunk(
                        params, seed, steps, block, num_warps, chunk, repeats, warmups,
                    ))
                except Exception as exc:
                    row = {
                        "phase": "chunk_search",
                        "n_samples": n_samples,
                        "steps": steps,
                        "block": block,
                        "num_warps": num_warps,
                        "graph_chunk": chunk,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    rows.append(row)
                    print(f"[chunk-search] failed: {row['error']}", flush=True)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
    return rows


def run_final(
    samples: Iterable[int],
    output_dir: Path,
    *,
    block: int,
    num_warps: int,
    graph_chunk: int,
    steps: int,
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_samples in samples:
        params = make_params(n_samples)
        seed = initial_field(params)
        torch.cuda.reset_peak_memory_stats()
        numpy_start = time.perf_counter()
        numpy_output = simulate_numpy(seed, params, steps)
        numpy_seconds = time.perf_counter() - numpy_start

        torch.cuda.reset_peak_memory_stats()
        pytorch_start = time.perf_counter()
        pytorch_output = simulate_pytorch(seed, params, steps)
        torch.cuda.synchronize()
        pytorch_seconds = time.perf_counter() - pytorch_start

        simulate_gpu_clear = __import__(
            "flashlle.solvers.nlse_gpu_opt", fromlist=["clear_graph_cache"]
        ).clear_graph_cache
        simulate_gpu_clear()
        torch.cuda.reset_peak_memory_stats()
        first_start = time.perf_counter()
        optimized_first = simulate_gpu(
            seed, params, steps, graph_chunk=graph_chunk,
            block=block, num_warps=num_warps,
        )
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - first_start
        steady_operation = lambda: simulate_gpu(
            seed, params, steps, graph_chunk=graph_chunk,
            block=block, num_warps=num_warps,
        )
        _warmup(steady_operation, warmups)
        steady_times = [_wall_time(steady_operation)[0] for _ in range(repeats)]
        row = _base_record(params, n_samples, steps)
        row.update({
            "phase": "final",
            "block": block,
            "num_warps": num_warps,
            "graph_chunk": graph_chunk,
            "numpy_seconds": numpy_seconds,
            "pytorch_end_to_end": pytorch_seconds,
            "optimized_first_call_seconds": first_seconds,
            "optimized_steady": _stats(steady_times, steps, n_samples),
            "pytorch_accuracy": accuracy(pytorch_output, numpy_output, params.dt),
            "optimized_accuracy": accuracy(optimized_first, numpy_output, params.dt),
            "optimized_vs_pytorch_relative_l2": relative_l2(optimized_first, pytorch_output),
            "pytorch_speedup_reference": numpy_seconds / pytorch_seconds,
            "optimized_speedup_pytorch": pytorch_seconds / median(steady_times),
            "optimized_speedup_eager_reference": first_seconds / median(steady_times),
            "memory": _memory(),
        })
        rows.append(row)
        print(
            f"[final] N={n_samples} numpy={numpy_seconds:.3f}s "
            f"pytorch={pytorch_seconds:.3f}s optimized={median(steady_times):.3f}s "
            f"speedup={row['optimized_speedup_pytorch']:.2f}x",
            flush=True,
        )
    return rows


def convergence_check(samples: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for n_samples in samples:
        outputs = []
        for dz, steps in ((0.001, 4096), (0.0005, 8192), (0.00025, 16384)):
            params = make_params(n_samples, dz)
            seed = initial_field(params)
            start = time.perf_counter()
            output = simulate_numpy(seed, params, steps)
            elapsed = time.perf_counter() - start
            outputs.append((dz, steps, output, elapsed))
        reference = outputs[-1][2]
        for dz, steps, output, elapsed in outputs:
            rows.append({
                "phase": "convergence",
                "n_samples": n_samples,
                "dz": dz,
                "steps": steps,
                "target_z": dz * steps,
                "seconds": elapsed,
                "relative_l2_to_smallest_dz": relative_l2(output, reference),
                "energy": float(make_params(n_samples, dz).dt * np.sum(np.abs(output) ** 2)),
                "finite": bool(np.isfinite(output).all()),
            })
    return rows


def write_results(output_dir: Path, name: str, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")

    rows = payload.get("rows", [])
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            flat[key] = json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list)) else value
        flat_rows.append(flat)
    if flat_rows:
        keys = sorted({key for row in flat_rows for key in row})
        with (output_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)

    summary = [f"# {name}", "", f"Rows: {len(rows)}", ""]
    if payload.get("selection"):
        summary += ["## Selection", "", "```json", json.dumps(payload["selection"], indent=2), "```", ""]
    summary += ["## Rows", "", "```json", json.dumps(rows, indent=2, default=_json_default), "```", ""]
    (output_dir / f"{name}.md").write_text("\n".join(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("correctness", "kernel", "chunk", "final", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "NLSE_bench" / "results")
    parser.add_argument("--samples", nargs="+", type=int, default=list(DEFAULT_SAMPLES))
    parser.add_argument("--tail-samples", nargs="+", type=int, default=list(DEFAULT_TAIL_SAMPLES))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--blocks", nargs="+", type=int, default=list(DEFAULT_BLOCKS))
    parser.add_argument("--warps", nargs="+", type=int, default=list(DEFAULT_WARPS))
    parser.add_argument("--chunks", nargs="+", type=int, default=list(DEFAULT_CHUNKS))
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--graph-chunk", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.steps < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("steps and repeats must be positive; warmups must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(metadata(), indent=2, default=_json_default), flush=True)

    if args.phase in ("correctness", "all"):
        rows = run_correctness(
            args.samples, args.tail_samples, args.output_dir,
            block=args.block, num_warps=args.num_warps, graph_chunk=args.graph_chunk,
        )
        write_results(args.output_dir, "correctness", {"metadata": metadata(), "rows": rows})

    selection: dict[str, Any] | None = None
    if args.phase in ("kernel", "all"):
        rows = run_kernel_search(
            args.samples, args.output_dir,
            blocks=args.blocks, warps=args.warps, steps=args.steps,
            graph_chunk=args.graph_chunk, repeats=args.repeats, warmups=args.warmups,
        )
        selection = select_best_config(rows)
        write_results(args.output_dir, "kernel_search", {
            "metadata": metadata(), "rows": rows, "selection": selection,
        })

    if args.phase in ("chunk", "all"):
        if selection is None:
            kernel_path = args.output_dir / "kernel_search.json"
            if not kernel_path.exists():
                raise RuntimeError("chunk phase requires kernel_search.json or --phase all")
            selection = json.loads(kernel_path.read_text())["selection"]
        rows = run_chunk_search(
            args.samples, args.output_dir,
            block=selection["block"], num_warps=selection["num_warps"],
            chunks=args.chunks, steps_list=(args.steps, args.steps + 1),
            repeats=args.repeats, warmups=args.warmups,
        )
        write_results(args.output_dir, "chunk_search", {
            "metadata": metadata(), "rows": rows, "selection": selection,
        })

    if args.phase in ("final", "all"):
        if selection is None:
            kernel_path = args.output_dir / "kernel_search.json"
            selection = json.loads(kernel_path.read_text())["selection"]
        chunk = args.graph_chunk
        chunk_path = args.output_dir / "chunk_search.json"
        if chunk_path.exists():
            chunk_rows = json.loads(chunk_path.read_text())["rows"]
            eligible = [
                row for row in chunk_rows
                if row.get("status") == "ok" and row.get("steps") == args.steps
            ]
            if eligible:
                chunk = min(eligible, key=lambda row: row["triton_graph"]["steady_median_seconds"])["graph_chunk"]
        rows = run_final(
            args.samples, args.output_dir,
            block=selection["block"], num_warps=selection["num_warps"],
            graph_chunk=chunk, steps=args.steps,
            repeats=args.repeats, warmups=args.warmups,
        )
        write_results(args.output_dir, "final", {
            "metadata": metadata(), "rows": rows,
            "selection": {**selection, "graph_chunk": chunk},
        })

    if args.phase == "all":
        rows = convergence_check((4096,))
        write_results(args.output_dir, "convergence", {"metadata": metadata(), "rows": rows})


if __name__ == "__main__":
    main()
