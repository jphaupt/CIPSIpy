"""
Comprehensive FCI tests for multiple quantum chemistry systems.

Tests validate the Hamiltonian implementation by:
1. Reading FCIDUMP files and reference FCI matrices from assets
2. Building Hamiltonian matrices element-wise
3. Comparing against reference matrices
4. Diagonalizing and comparing ground state energies

All required data is automatically discovered from the assets directory.
Each system directory must contain:
- FCIDUMP: Integral file
- fci_matrix.txt: Reference Hamiltonian matrix (with e_nuc on diagonal)
- e_gs.txt: Reference ground state energy
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.fcidump import read_fcidump
from cipsipy.hamiltonian import hamiltonian_element_spin


def discover_test_systems(assets_dir):
    """
    Discover all test systems in the assets directory.

    Returns:
        List of tuples: (system_path, relative_path, fcidump_path, matrix_path, e_gs_path)
    """
    systems = []

    for system_dir in assets_dir.iterdir():
        if not system_dir.is_dir() or system_dir.name.startswith(".") or system_dir.name == "__pycache__":
            continue

        # Check for files directly in system dir (e.g., assets/H2/sto-3g/)
        for subdir in system_dir.iterdir():
            if not subdir.is_dir():
                continue

            fcidump_path = subdir / "FCIDUMP"
            matrix_path = subdir / "fci_matrix.txt"
            e_gs_path = subdir / "e_gs.txt"

            # All required files must exist
            if fcidump_path.exists() and matrix_path.exists() and e_gs_path.exists():
                # Create relative path for display (e.g., "H2/sto-3g")
                relative_path = f"{system_dir.name}/{subdir.name}"
                systems.append((str(subdir), relative_path, str(fcidump_path),
                               str(matrix_path), str(e_gs_path)))

    return sorted(systems, key=lambda x: x[1])


def generate_determinants(n_orbitals, n_alpha, n_beta):
    """
    Generate all possible determinants for given number of electrons and orbitals.

    Lexicographic order of bit strings.
    For each alpha string (in lex order), iterate through all beta strings (in lex order).

    Args:
        n_orbitals: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons

    Returns:
        List of (det_alpha, det_beta) tuples where each determinant is an integer
        with bits representing occupied orbitals, ordered to match PySCF's convention.
    """
    def make_strings_lexicographic(n_orbitals, n_electrons):
        """Generate all bit strings with n_electrons bits set, in lexicographic order."""
        strings = []
        for occupied_orbitals in combinations(range(n_orbitals), n_electrons):
            det = 0
            for orb in occupied_orbitals:
                det |= 1 << orb
            strings.append(det)
        # Sort by integer value (lexicographic ordering of bit string)
        strings.sort()
        return strings

    # Generate strings in lexicographic order
    alpha_strings = make_strings_lexicographic(n_orbitals, n_alpha)
    beta_strings = make_strings_lexicographic(n_orbitals, n_beta)

    # outer loop alpha, inner loop beta
    determinants = []
    for det_alpha in alpha_strings:
        for det_beta in beta_strings:
            determinants.append((det_alpha, det_beta))

    return determinants


def build_hamiltonian_matrix(determinants, n_orbitals, h_core, eri):
    """
    Build full Hamiltonian matrix element-wise.

    Args:
        determinants: List of (det_alpha, det_beta) tuples
        n_orbitals: Number of spatial orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Hamiltonian matrix (without e_nuc on diagonal)
    """
    n_det = len(determinants)
    H = np.zeros((n_det, n_det))

    for i, (det_i_alpha, det_i_beta) in enumerate(determinants):
        for j, (det_j_alpha, det_j_beta) in enumerate(determinants):
            H[i, j] = hamiltonian_element_spin(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta,
                n_orbitals, h_core, eri
            )

    return H


@pytest.fixture(scope="module")
def assets_dir():
    """Return path to assets directory"""
    test_dir = Path(__file__).parent
    return test_dir.parent / "assets"


@pytest.fixture(scope="module")
def test_systems(assets_dir):
    """Discover and return all test systems"""
    return discover_test_systems(assets_dir)


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for all systems"""
    if "system_info" in metafunc.fixturenames:
        assets_dir = Path(metafunc.config.rootdir) / "assets"
        systems = discover_test_systems(assets_dir)
        # Use relative path as test ID
        metafunc.parametrize("system_info", systems, ids=[s[1] for s in systems])


class TestFCISystems:
    """Test FCI implementation on multiple quantum chemistry systems"""

    def test_matrix_elements(self, system_info):
        """
        Test that computed Hamiltonian matrix elements match reference.

        Validates element-wise construction of the Hamiltonian matrix.
        """
        system_path, relative_path, fcidump_path, matrix_path, e_gs_path = system_info

        print(f"\n{'='*70}")
        print(f"Testing Matrix Elements: {relative_path}")
        print(f"{'='*70}")

        # Read FCIDUMP
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        # Determine number of alpha/beta electrons
        # spin = 2*S = N_alpha - N_beta
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2

        print(f"System: {relative_path}")
        print(f"  Orbitals: {n_orb}")
        print(f"  Electrons: ({n_alpha}α, {n_beta}β)")
        print(f"  Core energy: {e_nuc:.6f} a.u.")

        # Generate determinants
        determinants = generate_determinants(n_orb, n_alpha, n_beta)
        n_det = len(determinants)
        print(f"  Determinants: {n_det}")

        # Build Hamiltonian matrix (without e_nuc)
        H_computed = build_hamiltonian_matrix(determinants, n_orb, h_core, eri)

        # Add nuclear repulsion to diagonal (reference matrices include it)
        H_computed_with_enuc = H_computed + np.eye(n_det) * e_nuc

        # Load reference matrix (includes e_nuc on diagonal)
        H_reference = np.loadtxt(matrix_path)
        if H_reference.ndim == 1:
            H_reference = H_reference.reshape(n_det, n_det)

        # Compare matrices element-wise
        max_diff = np.max(np.abs(H_computed_with_enuc - H_reference))
        print(f"\nMatrix comparison:")
        print(f"  Max absolute difference: {max_diff:.2e}")

        # Check diagonal elements
        diag_diff = np.abs(np.diag(H_computed_with_enuc) - np.diag(H_reference))
        print(f"  Max diagonal difference: {np.max(diag_diff):.2e}")

        # Check off-diagonal elements
        mask = ~np.eye(n_det, dtype=bool)
        offdiag_diff = np.abs(H_computed_with_enuc[mask] - H_reference[mask])
        if len(offdiag_diff) > 0:
            print(f"  Max off-diagonal difference: {np.max(offdiag_diff):.2e}")

        tolerance = 1e-7
        assert max_diff < tolerance, (
            f"Matrix elements differ by {max_diff:.2e}, "
            f"exceeds tolerance {tolerance:.2e}"
        )

        print(f"✓ All matrix elements match within {tolerance:.2e}")

    def test_ground_state_energy(self, system_info):
        """
        Test that diagonalization gives correct ground state energy.

        Validates that the eigenvalue problem is solved correctly.
        """
        system_path, relative_path, fcidump_path, matrix_path, e_gs_path = system_info

        print(f"\n{'='*70}")
        print(f"Testing Energy: {relative_path}")
        print(f"{'='*70}")

        # Read FCIDUMP
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        # Determine number of alpha/beta electrons
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2

        print(f"System: {relative_path}")
        print(f"  Orbitals: {n_orb}, Electrons: ({n_alpha}α, {n_beta}β)")

        # Generate determinants
        determinants = generate_determinants(n_orb, n_alpha, n_beta)

        # Build Hamiltonian matrix (without e_nuc)
        H = build_hamiltonian_matrix(determinants, n_orb, h_core, eri)

        # Add nuclear repulsion to diagonal (as in reference)
        H_full = H + np.eye(len(H)) * e_nuc

        # Diagonalize
        eigenvalues, eigenvectors = jnp.linalg.eigh(H_full)
        computed_energy = float(eigenvalues[0])

        # Read reference energy
        with open(e_gs_path, "r") as f:
            reference_energy = float(f.read().strip())

        print(f"\nGround state energy:")
        print(f"  Reference:  {reference_energy:.12f} a.u.")
        print(f"  Computed:   {computed_energy:.12f} a.u.")
        print(f"  Difference: {abs(computed_energy - reference_energy):.2e} a.u.")

        # Assert energy agreement
        energy_tolerance = 1e-7
        energy_diff = abs(computed_energy - reference_energy)
        assert energy_diff < energy_tolerance, (
            f"Ground state energy differs by {energy_diff:.2e}, "
            f"exceeds tolerance {energy_tolerance:.2e}"
        )

        print(f"✓ Ground state energy matches within {energy_tolerance:.2e}")

    def test_hermiticity(self, system_info):
        """Test that Hamiltonian matrix is Hermitian"""
        system_path, relative_path, fcidump_path, matrix_path, e_gs_path = system_info

        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        # Determine number of alpha/beta electrons
        n_alpha = (n_elec + spin) // 2
        n_beta = (n_elec - spin) // 2

        # Generate determinants
        determinants = generate_determinants(n_orb, n_alpha, n_beta)

        # Build Hamiltonian matrix
        H = build_hamiltonian_matrix(determinants, n_orb, h_core, eri)

        # Check Hermiticity (should be symmetric for real Hamiltonian)
        hermiticity_error = np.max(np.abs(H - H.T))

        print(f"\n{relative_path} - Hermiticity check:")
        print(f"  Max |H - H^T|: {hermiticity_error:.2e}")

        assert hermiticity_error < 1e-12, (
            f"Hamiltonian is not Hermitian: max error {hermiticity_error:.2e}"
        )

        print(f"✓ Hamiltonian is Hermitian within 1e-12")


class TestSystemCoverage:
    """Test that all expected systems are loaded"""

    def test_all_systems_loaded(self, test_systems):
        """Verify all asset systems are found and loaded"""
        system_names = [relative_path for _, relative_path, _, _, _ in test_systems]

        print(f"\nLoaded {len(test_systems)} FCI systems:")
        for name in sorted(system_names):
            print(f"  - {name}")

        # Check we have the expected systems
        expected_systems = ["H2/sto-3g", "HeH+/sto-3g", "H3+/3-21g",
                           "Li/sto-3g", "LiH/6-31gstar"]

        for expected in expected_systems:
            assert expected in system_names, f"Expected system {expected} not found"

        print(f"\n✓ All {len(expected_systems)} expected systems loaded")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
