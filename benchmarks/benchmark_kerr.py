"""Compatibility entry point for the Kerr-only PyTorch/Triton GPU comparison.

Prefer ``python benchmarks/benchmark_operators.py --operators kerr``. All other
command-line arguments are forwarded; --output is a JSON/PNG/SVG file prefix.
"""

import sys

from benchmark_operators import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--operators", "kerr"])
