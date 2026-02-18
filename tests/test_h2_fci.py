"""
Test H2 Full CI using spin-separated determinants.

This test validates the Hamiltonian implementation by:
1. Building the full CI Hamiltonian matrix for H2
2. Diagonalizing it
3. Comparing the ground state energy to expected values
"""

import os
import sys

import jax.numpy as jnp
import pytest

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.fcidump import read_fcidump
from cipsipy.hamiltonian import hamiltonian_element_spin

ENERGY_TOLERANCE = 1e-7


class TestH2FCI:
    """Test Full CI for H2 molecule using spin-separated determinants"""

    def test_h2_fci_energy(self):
        """
        Test H2 FCI calculation with spin-separated determinants.

        H2 with 2 electrons in 2 spatial orbitals (STO-3G basis):
        - Alpha spin: 1 electron, can be in orbital 0 or 1 (2 determinants)
        - Beta spin: 1 electron, can be in orbital 0 or 1 (2 determinants)
        - Total: 2 x 2 = 4 determinants

        Determinants (alpha, beta):
        0: (|0⟩, |0⟩) - both in orbital 0
        1: (|0⟩, |1⟩) - alpha in 0, beta in 1
        2: (|1⟩, |0⟩) - alpha in 1, beta in 0
        3: (|1⟩, |1⟩) - both in orbital 1
        """
        # Read H2 FCIDUMP
        fcidump_path = os.path.join(os.path.dirname(__file__), "../examples/h2/FCIDUMP")
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        print(f"\nH2 System: {n_elec} electrons, {n_orb} spatial orbitals, MS2={spin}")
        print(f"Nuclear repulsion: {e_nuc:.6f} Hartree")

        # Generate all determinants for H2
        # For 2 electrons in 2 spatial orbitals with MS2=0 (singlet)
        # Alpha and beta each have 1 electron
        determinants = [
            (1, 1),  # (|0⟩_α, |0⟩_β) - both in orbital 0
            (1, 2),  # (|0⟩_α, |1⟩_β) - alpha in 0, beta in 1
            (2, 1),  # (|1⟩_α, |0⟩_β) - alpha in 1, beta in 0
            (2, 2),  # (|1⟩_α, |1⟩_β) - both in orbital 1
        ]

        n_det = len(determinants)
        print(f"Number of determinants: {n_det}")

        # Build Hamiltonian matrix
        H_matrix = jnp.zeros((n_det, n_det))

        for i in range(n_det):
            det_i_alpha, det_i_beta = determinants[i]
            for j in range(n_det):
                det_j_alpha, det_j_beta = determinants[j]

                H_matrix = H_matrix.at[i, j].set(
                    hamiltonian_element_spin(
                        det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
                    )
                )

        print("\nHamiltonian matrix (electronic part):")
        print(H_matrix)

        # Diagonalize
        eigenvalues, eigenvectors = jnp.linalg.eigh(H_matrix)

        # Ground state energy (add nuclear repulsion)
        E_ground = eigenvalues[0] + e_nuc

        print("\nEigenvalues (electronic):")
        for i, E in enumerate(eigenvalues):
            print(f"  State {i}: {E:.6f} Hartree")

        print(f"\nGround state energy: {E_ground:.6f} Hartree")
        print(f"(Electronic: {eigenvalues[0]:.6f} + Nuclear: {e_nuc:.6f})")

        print("\nGround state eigenvector:")
        print(eigenvectors[:, 0])

        # Check that ground state energy matches PySCF reference
        # The PySCF FCI result for H2 STO-3G is -1.1372759436 Hartree
        pyscf_reference = -1.1372759436
        assert jnp.isclose(
            E_ground, pyscf_reference, atol=1e-7
        ), f"Ground state energy {E_ground:.6f} differs from PySCF reference {pyscf_reference:.6f}"

        # Check that ground state has correct symmetry
        # For H2 singlet ground state, determinant 0 (both in orbital 0) should dominate
        assert abs(eigenvectors[0, 0]) > 0.9, "Ground state should be dominated by |00⟩"

        print("\n✓ FCI test passed!")
        print(f"  Ground state energy: {E_ground:.6f} Hartree")
        print(f"  Dominant determinant: |α0 β0⟩ with coefficient {eigenvectors[0, 0]:.4f}")

    def test_hamiltonian_matrix_symmetry(self):
        """Test that Hamiltonian matrix is Hermitian"""
        fcidump_path = os.path.join(os.path.dirname(__file__), "../examples/h2/FCIDUMP")
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        determinants = [(1, 1), (1, 2), (2, 1), (2, 2)]
        n_det = len(determinants)

        H_matrix = jnp.zeros((n_det, n_det))
        for i in range(n_det):
            det_i_alpha, det_i_beta = determinants[i]
            for j in range(n_det):
                det_j_alpha, det_j_beta = determinants[j]
                H_matrix = H_matrix.at[i, j].set(
                    hamiltonian_element_spin(
                        det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
                    )
                )

        # Check Hermiticity
        assert jnp.allclose(H_matrix, H_matrix.T), "Hamiltonian matrix should be Hermitian"

    def test_single_determinant_elements(self):
        """Test specific Hamiltonian matrix elements"""
        fcidump_path = os.path.join(os.path.dirname(__file__), "../examples/h2/FCIDUMP")
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        # Test diagonal element for |00⟩
        det_alpha, det_beta = 0b1, 0b1  # Both in orbital 0
        H_00 = hamiltonian_element_spin(
            det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri
        )

        # Manual calculation for |00⟩:
        # E = 2*h[0,0] + (00|00)
        expected_H_00 = 2 * h_core[0, 0] + eri[0, 0, 0, 0]

        print("\nDiagonal element H(|00⟩, |00⟩):")
        print(f"  Calculated: {H_00:.6f}")
        print(f"  Expected:   {expected_H_00:.6f}")

        assert jnp.isclose(
            H_00, expected_H_00, rtol=1e-5
        ), f"Diagonal element mismatch: {H_00} != {expected_H_00}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
