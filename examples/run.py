from pathlib import Path
import os
import sys

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.cipsi import CIPSISolver


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _repo_root()
    fcidump_path = root / "assets" / "H3+" / "3-21g" / "FCIDUMP"
    ref_path = root / "assets" / "H3+" / "3-21g" / "e_gs.txt"

    with open(ref_path, "r", encoding="utf-8") as f:
        e_ref = float(f.read().strip())

    solver = CIPSISolver(fcidump_filename=str(fcidump_path))
    e_cipsi = float(solver.run_cipsi())

    # Recompute final variational/PT2 components for reporting.
    e_var_el, _ = solver._diagonalise_variational_space()
    da_ext, db_ext, eps_ext = solver.run_unfiltered_selection(e_var_el)
    contribs = solver._aggregate_external_contributions(da_ext, db_ext, eps_ext)

    e_var = e_var_el + solver.ham.e_nuc
    e_pt2 = float(sum(contribs.values()))
    e_est = e_var + e_pt2

    print("=" * 72)
    print(f"E_CIPSI(return): {e_cipsi: .12f} a.u.")
    print(f"E_FCI(ref):      {e_ref: .12f} a.u.")
    print(f"|Delta|:         {abs(e_cipsi - e_ref): .3e} a.u.")


if __name__ == "__main__":
    main()
