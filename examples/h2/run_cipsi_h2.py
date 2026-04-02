"""
Run a full CIPSI calculation for H2/STO-3G and print final energies.

Usage:
    python examples/h2/run_cipsi_h2.py
"""

from pathlib import Path
import os
import sys

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax


def pick_device(gpu_id=0):
    try:
        device = jax.devices("gpu")[gpu_id]
    except (RuntimeError, IndexError):
        device = jax.devices("cpu")[0]

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Selected device: {device}")
    return device

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _repo_root()
    fcidump_path = root / "assets" / "H2" / "sto-3g" / "FCIDUMP"
    ref_path = root / "assets" / "H2" / "sto-3g" / "e_gs.txt"

    device = pick_device(gpu_id=0)

    with open(ref_path, "r", encoding="utf-8") as f:
        e_ref = float(f.read().strip())

    with jax.default_device(device):
        from cipsipy.cipsi import CIPSISolver
        solver = CIPSISolver(fcidump_filename=str(fcidump_path))
        e_var, e_est = solver.run_cipsi()

    print("=" * 72)
    print(f"E_var(return):   {float(e_var): .12f} a.u.")
    print(f"E_est(return):   {float(e_est): .12f} a.u.")
    print(f"E_FCI(ref):      {e_ref: .12f} a.u.")
    print(f"|Delta E_var|:   {abs(float(e_var) - e_ref): .3e} a.u.")
    print(f"|Delta E_est|:   {abs(float(e_est) - e_ref): .3e} a.u.")


if __name__ == "__main__":
    main()
