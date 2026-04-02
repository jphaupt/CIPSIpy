"""Simple end-to-end benchmark for CIPSI.

Run with:
    pytest benchmarks/test_cipsi_performance.py --benchmark-only

Tip:
    Add --benchmark-autosave to keep historical benchmark runs.

This first-pass benchmark only times whole-program execution.
For optimization work later, add fine-grained stage benchmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    """Allow running benchmarks without requiring a prior editable install."""
    src_path = _repo_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from cipsipy.cipsi import CIPSISolver


def _fcidump(system: str) -> str:
    return str(_repo_root() / "assets" / system / "FCIDUMP")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "system,max_dets,max_iterations",
    [
        pytest.param("H3+/3-21g", 10, 1, id="h3p_321g"),
        pytest.param("Li/sto-3g", 10, 1, id="li_sto3g"),
        pytest.param("LiH/6-31gstar", 12, 1, id="lih_631gstar", marks=pytest.mark.slow),
    ],
)
def test_cipsi_end_to_end(benchmark, system: str, max_dets: int, max_iterations: int):
    """Benchmark a short full CIPSI run with bounded determinant growth."""

    def _run() -> None:
        solver = CIPSISolver(fcidump_filename=_fcidump(system))
        solver.run_cipsi(max_iterations=max_iterations, max_dets=max_dets)

    benchmark(_run)
