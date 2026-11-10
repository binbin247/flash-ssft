"""CPU solver usage must not depend on an installed GPU stack."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_cpu_solver_and_example_work_without_gpu_dependencies():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import importlib.abc
            import sys

            class BlockGPUImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'triton'}:
                        raise AssertionError(f'CPU usage tried to import {fullname}')

            sys.meta_path.insert(0, BlockGPUImports())

            import flashlle
            assert not any(name.startswith('flashlle.') for name in sys.modules)

            import numpy as np
            from flashlle.solvers import NLSEParams, linear_coefficients, simulate_numpy
            from examples.run_tod import initial_field

            params = NLSEParams(n_samples=128, dt=0.1)
            seed = initial_field(params)
            original = seed.copy()
            linear = linear_coefficients(params)
            final = simulate_numpy(seed, params, steps=3)
            assert linear.shape == final.shape == seed.shape
            assert final.dtype == np.complex128
            assert np.isfinite(final).all()
            assert not np.shares_memory(seed, final)
            np.testing.assert_array_equal(seed, original)
            assert not {'torch', 'triton', 'flashlle.operators', 'flashlle.solvers.nlse'} & sys.modules.keys()
        """)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
