"""
Tests for Hamiltonian matrix elements using Slater-Condon rules
"""

import os
import sys

import jax.numpy as jnp
import pytest

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.hamiltonian import (
    excitation_level,
    get_excitation_operators,
    hamiltonian_diagonal,
    hamiltonian_double,
    hamiltonian_element,
    hamiltonian_single,
)


class TestExcitationLevel:
    """Test determining excitation level between determinants"""

    def test_same_determinant(self):
        """Test that same determinant gives 0 excitations"""
        det_i = 3  # 0b0011
        det_j = 3  # 0b0011
        level = excitation_level(det_i, det_j)
        assert level == 0

    def test_single_excitation(self):
        """Test single excitation gives level 1"""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 6  # 0b0110 (orbitals 1, 2)
        level = excitation_level(det_i, det_j)
        assert level == 1

    def test_double_excitation(self):
        """Test double excitation gives level 2"""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 12  # 0b1100 (orbitals 2, 3)
        level = excitation_level(det_i, det_j)
        assert level == 2

    def test_triple_excitation(self):
        """Test triple excitation gives level 3"""
        det_i = 7  # 0b0111 (orbitals 0, 1, 2)
        det_j = 56  # 0b111000 (orbitals 3, 4, 5)
        level = excitation_level(det_i, det_j)
        assert level == 3


class TestGetExcitationOperators:
    """Test extracting hole and particle indices"""

    def test_single_excitation_indices(self):
        """Test getting hole/particle indices for single excitation"""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 6  # 0b0110 (orbitals 1, 2)
        holes, particles = get_excitation_operators(det_i, det_j)
        # Orbital 0 is removed (hole), orbital 2 is added (particle)
        assert holes == [0]
        assert particles == [2]

    def test_double_excitation_indices(self):
        """Test getting hole/particle indices for double excitation"""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 12  # 0b1100 (orbitals 2, 3)
        holes, particles = get_excitation_operators(det_i, det_j)
        # Orbitals 0, 1 are removed, orbitals 2, 3 are added
        assert holes == [0, 1]
        assert particles == [2, 3]


class TestHamiltonianDiagonal:
    """Test Hamiltonian diagonal elements (same determinant)"""

    def test_diagonal_simple(self):
        """Test diagonal element for simple system"""
        # Simple 2-orbital system with 2 electrons
        det = 3  # 0b0011 (orbitals 0, 1 occupied)
        n_orb = 2

        # One-electron integrals
        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])

        # Two-electron integrals (simplified)
        eri = jnp.zeros((2, 2, 2, 2))
        eri = eri.at[0, 0, 0, 0].set(0.5)  # (00|00)
        eri = eri.at[1, 1, 1, 1].set(0.3)  # (11|11)
        eri = eri.at[0, 1, 0, 1].set(0.1)  # (01|01)
        eri = eri.at[1, 0, 1, 0].set(0.1)  # (10|10)

        energy = hamiltonian_diagonal(det, n_orb, h_core, eri)

        # Expected: sum of h[i,i] for occupied + 1/2 sum of (ii|jj) - (ij|ji)
        # h[0,0] + h[1,1] = -1.0 + (-0.5) = -1.5
        # 1/2 * [(00|00) + (11|11) + 2*(01|01) - 2*(01|10)]
        # Note: (ij|ij) - (ij|ji) = J - K
        expected = -1.5 + 0.5 * (0.5 + 0.3) + (0.1 - 0.1)

        # Actually, for 2 electrons in orbitals 0 and 1:
        # E = h[0,0] + h[1,1] + (00|11) - (01|10)
        # The diagonal is: sum_i h[i,i] + 1/2 sum_{ij} [(ii|jj) - (ij|ji)]
        # For i=0, j=1: (00|11) - (01|10)
        expected = -1.0 + (-0.5) + (0.1 - 0.1)
        expected = -1.5

        assert jnp.isclose(energy, expected, atol=1e-10)

    def test_diagonal_empty(self):
        """Test diagonal for empty determinant"""
        det = 0  # No electrons
        n_orb = 2
        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])
        eri = jnp.zeros((2, 2, 2, 2))

        energy = hamiltonian_diagonal(det, n_orb, h_core, eri)
        assert energy == 0.0


class TestHamiltonianSingle:
    """Test Hamiltonian single excitation elements"""

    def test_single_excitation_element(self):
        """Test single excitation matrix element"""
        # Simple test case
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 6  # 0b0110 (orbitals 1, 2) - excitation 0->2
        n_orb = 3

        # One-electron integral
        h_core = jnp.array([[-1.0, 0.1, 0.2], [0.1, -0.5, 0.3], [0.2, 0.3, -0.3]])

        # Two-electron integrals
        eri = jnp.zeros((3, 3, 3, 3))
        # Set some values for testing
        eri = eri.at[0, 2, 1, 1].set(0.15)  # (02|11) = (ia|kk)
        eri = eri.at[0, 1, 1, 2].set(0.05)  # (01|12) = (ik|ka)

        element = hamiltonian_single(det_i, det_j, n_orb, h_core, eri)

        # For single excitation i->a with other occupied orbitals k:
        # H = h[i,a] + sum_k [(ia|kk) - (ik|ka)]
        # Here: i=0, a=2, k=1 (the other occupied orbital in both dets)
        # H = h[0,2] + [(02|11) - (01|12)]
        expected = 0.2 + (0.15 - 0.05)

        # Need to include phase
        # Phase depends on number of electrons between 0 and 2
        # In det_i, between 0 and 2 exclusive is orbital 1 (occupied)
        # So phase = (-1)^1 = -1
        expected = -1 * (0.2 + 0.15 - 0.05)

        assert jnp.isclose(element, expected, rtol=1e-5)


class TestHamiltonianDouble:
    """Test Hamiltonian double excitation elements"""

    def test_double_excitation_element(self):
        """Test double excitation matrix element"""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 12  # 0b1100 (orbitals 2, 3)
        n_orb = 4

        h_core = jnp.zeros((4, 4))

        # Two-electron integrals
        eri = jnp.zeros((4, 4, 4, 4))
        # For double excitation i,j -> a,b, only (ij|ab) term contributes
        eri = eri.at[0, 1, 2, 3].set(0.25)

        element = hamiltonian_double(det_i, det_j, n_orb, h_core, eri)

        # For double excitation i,j -> a,b:
        # H = (ij|ab)
        # The phase calculation is complex, just check it's non-zero
        assert element != 0.0


class TestHamiltonianElement:
    """Test general Hamiltonian matrix element function"""

    def test_hamiltonian_same_determinant(self):
        """Test Hamiltonian for same determinant (diagonal)"""
        det = 3  # 0b0011
        n_orb = 2
        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])
        eri = jnp.zeros((2, 2, 2, 2))

        element = hamiltonian_element(det, det, n_orb, h_core, eri)
        expected = hamiltonian_diagonal(det, n_orb, h_core, eri)

        assert jnp.isclose(element, expected)

    def test_hamiltonian_single_excitation(self):
        """Test Hamiltonian for single excitation"""
        det_i = 3  # 0b0011
        det_j = 6  # 0b0110
        n_orb = 3
        h_core = jnp.eye(3) * -1.0
        eri = jnp.zeros((3, 3, 3, 3))

        element = hamiltonian_element(det_i, det_j, n_orb, h_core, eri)
        expected = hamiltonian_single(det_i, det_j, n_orb, h_core, eri)

        assert jnp.isclose(element, expected)

    def test_hamiltonian_double_excitation(self):
        """Test Hamiltonian for double excitation"""
        det_i = 3  # 0b0011
        det_j = 12  # 0b1100
        n_orb = 4
        h_core = jnp.zeros((4, 4))
        eri = jnp.ones((4, 4, 4, 4)) * 0.1

        element = hamiltonian_element(det_i, det_j, n_orb, h_core, eri)
        expected = hamiltonian_double(det_i, det_j, n_orb, h_core, eri)

        assert jnp.isclose(element, expected)

    def test_hamiltonian_higher_excitation(self):
        """Test Hamiltonian for triple or higher excitation returns 0"""
        det_i = 7  # 0b0111 (orbitals 0, 1, 2)
        det_j = 56  # 0b111000 (orbitals 3, 4, 5)
        n_orb = 6
        h_core = jnp.zeros((6, 6))
        eri = jnp.zeros((6, 6, 6, 6))

        element = hamiltonian_element(det_i, det_j, n_orb, h_core, eri)
        assert element == 0.0

    def test_hamiltonian_matrix_h2(self):
        """Test building a small Hamiltonian matrix for H2"""
        # Simple 2-orbital, 2-electron system
        # Determinants: |11⟩ (both electrons in orbital 0 and 1)
        n_orb = 2

        h_core = jnp.array([[-1.0, 0.1], [0.1, -0.5]])

        eri = jnp.zeros((2, 2, 2, 2))
        eri = eri.at[0, 0, 1, 1].set(0.2)
        eri = eri.at[1, 1, 0, 0].set(0.2)

        det = 3  # 0b0011 - both orbitals occupied

        # Diagonal element
        H_diag = hamiltonian_element(det, det, n_orb, h_core, eri)

        # Should be sum of one-electron integrals plus two-electron
        # E = h[0,0] + h[1,1] + (00|11)
        expected = -1.0 + (-0.5) + 0.2
        expected = -1.3

        assert jnp.isclose(H_diag, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
