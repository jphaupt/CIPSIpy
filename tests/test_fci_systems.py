"""
FCI validation tests for multiple quantum chemistry systems.

Validates Hamiltonian matrix elements and ground state energies against
PySCF reference data for various molecular systems.

Two-electron integrals use chemist's notation: (pq|rs) = eri[p,q,r,s]
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.fcidump import read_fcidump
from cipsipy.hamiltonian import hamiltonian_element_spin


# ============================================================================
# Test Systems Configuration
# ============================================================================

# List of systems to test. Comment out lines to skip specific systems.
TEST_SYSTEMS = [
    "H2/sto-3g",
    "HeH+/sto-3g",
    "H3+/3-21g",
    "Li/sto-3g",
    "LiH/6-31gstar",
]


# ============================================================================
# Helper Functions
# ============================================================================


def generate_determinants(n_orbitals, n_alpha, n_beta):
    """
    Generate determinants in lexicographic order matching PySCF convention.

    Returns:
        List of (det_alpha, det_beta) tuples with integer bitstrings.
    """
    def make_strings_lexicographic(n_orbitals, n_electrons):
        strings = []
        for occupied_orbitals in combinations(range(n_orbitals), n_electrons):
            det = 0
            for orb in occupied_orbitals:
                det |= 1 << orb
            strings.append(det)
        strings.sort()
        return strings

    alpha_strings = make_strings_lexicographic(n_orbitals, n_alpha)
    beta_strings = make_strings_lexicographic(n_orbitals, n_beta)

    determinants = []
    for det_alpha in alpha_strings:
        for det_beta in beta_strings:
            determinants.append((det_alpha, det_beta))

    return determinants


def build_hamiltonian_matrix(determinants, n_orbitals, h_core, eri):
    """Build Hamiltonian matrix element-wise (without nuclear repulsion)."""
    n_det = len(determinants)
    H = np.zeros((n_det, n_det))

    for i, (det_i_alpha, det_i_beta) in enumerate(determinants):
        for j, (det_j_alpha, det_j_beta) in enumerate(determinants):
            H[i, j] = hamiltonian_element_spin(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta,
                n_orbitals, h_core, eri
            )

    return H


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def assets_dir():
    """Path to assets directory."""
    test_dir = Path(__file__).parent
    return test_dir.parent / "assets"


@pytest.fixture(scope="module", params=TEST_SYSTEMS)
def system_data(request, assets_dir):
    """
    Load and cache all data for a test system.

    Returns:
        Dict with system name, integrals, determinants, and matrices.
    """
    system_name = request.param
    system_path = assets_dir / system_name.replace("/", os.sep)

    # Load FCIDUMP
    fcidump_path = system_path / "FCIDUMP"
    n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(str(fcidump_path))

    # Determine electron configuration
    n_alpha = (n_elec + spin) // 2
    n_beta = (n_elec - spin) // 2

    # Generate determinants
    determinants = generate_determinants(n_orb, n_alpha, n_beta)
    n_det = len(determinants)

    # Build Hamiltonian matrix (cached for all tests)
    H = build_hamiltonian_matrix(determinants, n_orb, h_core, eri)
    H_with_enuc = H + np.eye(n_det) * e_nuc

    # Load reference data
    H_ref = np.loadtxt(system_path / "fci_matrix.txt")
    if H_ref.ndim == 1:
        H_ref = H_ref.reshape(n_det, n_det)

    with open(system_path / "e_gs.txt") as f:
        e_gs_ref = float(f.read().strip())

    return {
        "name": system_name,
        "n_orb": n_orb,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "n_det": n_det,
        "e_nuc": e_nuc,
        "H": H_with_enuc,
        "H_ref": H_ref,
        "e_gs_ref": e_gs_ref,
    }


# ============================================================================
# Tests
# ============================================================================


class TestFCISystems:
    """FCI validation tests."""

    def test_matrix_elements(self, system_data):
        """Validate Hamiltonian matrix elements against reference."""
        name = system_data["name"]
        H = system_data["H"]
        H_ref = system_data["H_ref"]
        n_det = system_data["n_det"]

        print(f"\n{'='*70}")
        print(f"Matrix Elements: {name}")
        print(f"{'='*70}")
        print(f"  Orbitals: {system_data['n_orb']}")
        print(f"  Electrons: ({system_data['n_alpha']}α, {system_data['n_beta']}β)")
        print(f"  Determinants: {n_det}")

        max_diff = np.max(np.abs(H - H_ref))
        diag_diff = np.max(np.abs(np.diag(H) - np.diag(H_ref)))

        mask = ~np.eye(n_det, dtype=bool)
        offdiag_diff = np.max(np.abs(H[mask] - H_ref[mask])) if mask.any() else 0.0

        print(f"\n  Max difference: {max_diff:.2e}")
        print(f"  Diagonal:       {diag_diff:.2e}")
        print(f"  Off-diagonal:   {offdiag_diff:.2e}")

        tolerance = 1e-12
        assert max_diff < tolerance, (
            f"Matrix elements differ by {max_diff:.2e} > {tolerance:.2e}"
        )

        print(f"  ✓ Pass (tolerance: {tolerance:.2e})")

    def test_ground_state_energy(self, system_data):
        """Validate ground state energy from diagonalization."""
        name = system_data["name"]
        H = system_data["H"]
        e_gs_ref = system_data["e_gs_ref"]

        print(f"\n{'='*70}")
        print(f"Ground State Energy: {name}")
        print(f"{'='*70}")

        eigenvalues = jnp.linalg.eigh(H)[0]
        e_gs_computed = float(eigenvalues[0])

        print(f"  Reference: {e_gs_ref:.12f} a.u.")
        print(f"  Computed:  {e_gs_computed:.12f} a.u.")
        print(f"  Δ:         {abs(e_gs_computed - e_gs_ref):.2e} a.u.")

        tolerance = 1e-12
        energy_diff = abs(e_gs_computed - e_gs_ref)
        assert energy_diff < tolerance, (
            f"Energy differs by {energy_diff:.2e} > {tolerance:.2e}"
        )

        print(f"  ✓ Pass (tolerance: {tolerance:.2e})")

    def test_hermiticity(self, system_data):
        """Validate Hamiltonian is Hermitian."""
        name = system_data["name"]
        H = system_data["H"]

        hermiticity_error = np.max(np.abs(H - H.T))

        print(f"\n{name} - Hermiticity: {hermiticity_error:.2e}")

        assert hermiticity_error < 1e-12, (
            f"Non-Hermitian: error {hermiticity_error:.2e}"
        )

        print(f"  ✓ Pass")


class TestSystemCoverage:
    """Verify expected systems are available."""

    def test_all_systems_available(self, assets_dir):
        """Check that all configured systems exist."""
        print(f"\nConfigured systems ({len(TEST_SYSTEMS)}):")
        missing = []

        for system_name in TEST_SYSTEMS:
            system_path = assets_dir / system_name.replace("/", os.sep)
            fcidump = system_path / "FCIDUMP"
            matrix = system_path / "fci_matrix.txt"
            energy = system_path / "e_gs.txt"

            if fcidump.exists() and matrix.exists() and energy.exists():
                print(f"  ✓ {system_name}")
            else:
                print(f"  ✗ {system_name} (missing files)")
                missing.append(system_name)

        assert not missing, f"Missing systems: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
