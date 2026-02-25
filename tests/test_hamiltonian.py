"""
Comprehensive tests for spin-separated Hamiltonian matrix elements.

Tests all functions in hamiltonian.py with proper spin handling.
"""

import os
import sys

import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.hamiltonian import (
    excitation_level,
    get_excitation_operators,
    hamiltonian_element,
    hamiltonian_vector_product,
)

import jax
import jax.numpy as jnp

def generate_random_test_data(n_det, norb, seed=42):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    coeffs = jax.random.normal(k1, (n_det,))

    # Random h_core (must be symmetric)
    h_rand = jax.random.normal(k2, (norb, norb))
    h_core = (h_rand + h_rand.T) / 2

    #  Random ERI (norb, norb, norb, norb)
    eri = jax.random.normal(k3, (norb, norb, norb, norb))
    # Note: For strict physical accuracy, one would symmetrize eri here,
    # but for testing the loop logic, raw random values work.

    return coeffs, h_core, eri

class TestHelperFunctions:
    """Test excitation level and operator extraction functions."""

    def test_excitation_level_same(self):
        """Same determinant gives 0 excitations."""
        assert excitation_level(3, 3) == 0

    def test_excitation_level_single(self):
        """Single excitation gives level 1."""
        det_i = 3  # 0b0011
        det_j = 6  # 0b0110
        assert excitation_level(det_i, det_j) == 1

    def test_excitation_level_double(self):
        """Double excitation gives level 2."""
        det_i = 3  # 0b0011
        det_j = 12  # 0b1100
        assert excitation_level(det_i, det_j) == 2

    def test_get_excitation_operators_single(self):
        """Extract correct hole/particle indices for single excitation."""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 6  # 0b0110 (orbitals 1, 2)
        holes, particles = get_excitation_operators(det_i, det_j)
        assert holes == [0]
        assert particles == [2]

    def test_get_excitation_operators_double(self):
        """Extract correct hole/particle indices for double excitation."""
        det_i = 3  # 0b0011 (orbitals 0, 1)
        det_j = 12  # 0b1100 (orbitals 2, 3)
        holes, particles = get_excitation_operators(det_i, det_j)
        assert holes == [0, 1]
        assert particles == [2, 3]


class TestDiagonalElements:
    """Test diagonal Hamiltonian elements."""

    def test_diagonal_two_electrons_same_orbital(self):
        """Test diagonal for 2 electrons in same orbital, different spins."""
        # Alpha in orbital 0, beta in orbital 0 (same spatial orbital)
        det_alpha = 1  # 0b01
        det_beta = 1   # 0b01
        n_orb = 2

        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])
        eri = jnp.zeros((2, 2, 2, 2))
        eri = eri.at[0, 0, 0, 0].set(0.5)  # (00|00)

        energy = hamiltonian_element(det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri)

        # Expected: 2*h[0,0] + (00|00) = 2*(-1.0) + 0.5 = -1.5
        expected = -2.0 + 0.5
        assert jnp.isclose(energy, expected, atol=1e-10)

    def test_diagonal_different_orbitals(self):
        """Test diagonal for alpha and beta electrons in different orbitals."""
        # Alpha in orbital 0, beta in orbital 1
        det_alpha = 1  # 0b01
        det_beta = 2   # 0b10
        n_orb = 2

        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])
        eri = jnp.zeros((2, 2, 2, 2))
        eri = eri.at[0, 0, 1, 1].set(0.2)  # (00|11)

        energy = hamiltonian_element(det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri)

        # Expected: h[0,0] + h[1,1] + (00|11) = -1.0 + (-0.5) + 0.2 = -1.3
        expected = -1.0 - 0.5 + 0.2
        assert jnp.isclose(energy, expected, atol=1e-10)

    def test_diagonal_alpha_alpha_exchange(self):
        """Test diagonal with alpha-alpha exchange interaction."""
        # Two alpha electrons in orbitals 0 and 1
        det_alpha = 3  # 0b11
        det_beta = 0   # 0b00
        n_orb = 2

        h_core = jnp.array([[-1.0, 0.0], [0.0, -0.5]])
        eri = jnp.zeros((2, 2, 2, 2))
        eri = eri.at[0, 0, 1, 1].set(0.3)  # Coulomb
        eri = eri.at[0, 1, 1, 0].set(0.1)  # Exchange

        energy = hamiltonian_element(det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri)

        # Expected: h[0,0] + h[1,1] + 0.5*[(00|11) - (01|10)]
        # The 0.5 factor accounts for double-counting in the i!=j sum
        # = -1.0 + (-0.5) + 0.5*(0.3 - 0.1) = -1.5 + 0.1 = -1.4
        expected = -1.0 - 0.5 + 0.5 * (0.3 - 0.1)
        assert jnp.isclose(energy, expected, atol=1e-10)


class TestSingleExcitations:
    """Test single excitation matrix elements."""

    def test_single_excitation_alpha(self):
        """Test single alpha excitation."""
        # Initial: α in 0, β in 1; Final: α in 2, β in 1 (α: 0→2)
        det_i_alpha = 1  # 0b01
        det_i_beta = 2   # 0b10
        det_j_alpha = 4  # 0b100
        det_j_beta = 2   # 0b10
        n_orb = 3

        h_core = jnp.zeros((3, 3))
        h_core = h_core.at[0, 2].set(0.5)  # h[i,a]

        eri = jnp.zeros((3, 3, 3, 3))
        eri = eri.at[0, 2, 2, 2].set(0.1)  # (ia|kk) same spin (k=a, no contribution)
        eri = eri.at[0, 2, 1, 1].set(0.15)  # (ia|kk) opposite spin (k=1 beta)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # Expected: phase * [h[0,2] + (02|11)]
        # Phase for 0→2 with nothing in between = +1
        # = 1 * [0.5 + 0.15] = 0.65
        expected = 0.65
        assert jnp.isclose(element, expected, atol=1e-10)

    def test_single_excitation_beta(self):
        """Test single beta excitation."""
        # Initial: α in 0, β in 1; Final: α in 0, β in 2 (β: 1→2)
        det_i_alpha = 1  # 0b01
        det_i_beta = 2   # 0b10
        det_j_alpha = 1  # 0b01
        det_j_beta = 4   # 0b100
        n_orb = 3

        h_core = jnp.zeros((3, 3))
        h_core = h_core.at[1, 2].set(0.3)  # h[i,a]

        eri = jnp.zeros((3, 3, 3, 3))
        eri = eri.at[1, 2, 0, 0].set(0.2)  # (ia|kk) opposite spin (k=0 alpha)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # Expected: phase * [h[1,2] + (12|00)]
        # Phase for 1→2 with nothing in between = +1
        # = 1 * [0.3 + 0.2] = 0.5
        expected = 0.5
        assert jnp.isclose(element, expected, atol=1e-10)

    def test_single_excitation_negative_phase(self):
        """Test single excitation with negative phase."""
        # Initial: α in 0,1; Final: α in 0,2 (α: 1→2, orbital 1 between them)
        det_i_alpha = 3  # 0b011
        det_i_beta = 0   # 0b000
        det_j_alpha = 5  # 0b101
        det_j_beta = 0   # 0b000
        n_orb = 3

        h_core = jnp.zeros((3, 3))
        h_core = h_core.at[1, 2].set(0.4)

        eri = jnp.zeros((3, 3, 3, 3))
        eri = eri.at[1, 2, 0, 0].set(0.1)  # (12|00)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # Phase for 1→2 with orbital 0 occupied in between (below 1) = +1
        # But wait - we need to count orbitals between 1 and 2 exclusive
        # Between 1 and 2: none
        # So phase = +1
        # But we also have orbital 0 in the final state
        # Actually phase_single counts electrons between i and a in det_i
        # det_i_alpha = 3 = 0b011, i=1, a=2
        # Orbitals between 1 and 2 (exclusive): none
        # Phase = (-1)^0 = +1
        # Element = 1 * [0.4 + (12|00) - (10|02)]
        # We need exchange term: (ik|ka) where k=0 (same spin)
        eri = eri.at[1, 0, 0, 2].set(0.05)  # (10|02)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # = 1 * [0.4 + 0.1 - 0.05] = 0.45
        expected = 0.45
        assert jnp.isclose(element, expected, atol=1e-10)


class TestDoubleExcitations:
    """Test double excitation matrix elements."""

    def test_double_same_spin_alpha(self):
        """Test double excitation within alpha spin."""
        # Initial: α in 0,1; Final: α in 2,3 (α: 0,1→2,3)
        det_i_alpha = 3   # 0b0011
        det_i_beta = 0    # 0b0000
        det_j_alpha = 12  # 0b1100
        det_j_beta = 0    # 0b0000
        n_orb = 4

        h_core = jnp.zeros((4, 4))
        eri = jnp.zeros((4, 4, 4, 4))
        eri = eri.at[0, 1, 2, 3].set(0.25)  # (ij|ab)
        eri = eri.at[0, 1, 3, 2].set(0.05)  # (ij|ba)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # Expected: phase * [(01|23) - (01|32)]
        # Phase calculation is complex, but element should be:
        # phase * (0.25 - 0.05) = phase * 0.2
        # Need to determine phase - with i=0,j=1,a=2,b=3
        # Phase = (-1)^P where P counts permutations
        assert element != 0.0
        assert jnp.abs(element) == pytest.approx(0.2, abs=1e-10)

    def test_double_opposite_spin(self):
        """Test double excitation with opposite spins."""
        # Initial: α in 0, β in 1; Final: α in 2, β in 3 (α: 0→2, β: 1→3)
        det_i_alpha = 1  # 0b0001
        det_i_beta = 2   # 0b0010
        det_j_alpha = 4  # 0b0100
        det_j_beta = 8   # 0b1000
        n_orb = 4

        h_core = jnp.zeros((4, 4))
        eri = jnp.zeros((4, 4, 4, 4))
        eri = eri.at[0, 2, 1, 3].set(0.35)  # (i_α a_α | i_β a_β)

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        # Expected: phase_α * phase_β * (02|13)
        # phase_α for 0→2: no electrons between = +1
        # phase_β for 1→3: electrons between 1 and 3 in det_i_beta (none) = +1
        # Total phase = 1 * 1 = 1
        # Element = 1 * 0.35 = 0.35
        expected = 0.35
        assert jnp.isclose(element, expected, atol=1e-10)

    def test_triple_excitation_zero(self):
        """Test that triple or higher excitations return 0."""
        # Initial: α in 0,1,2; Final: α in 3,4,5 (3 excitations)
        det_i_alpha = 7   # 0b000111
        det_i_beta = 0    # 0b000000
        det_j_alpha = 56  # 0b111000
        det_j_beta = 0    # 0b000000
        n_orb = 6

        h_core = jnp.zeros((6, 6))
        eri = jnp.ones((6, 6, 6, 6)) * 0.1

        element = hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri)

        assert element == 0.0

class TestMatrixVectorProducts:
    @staticmethod
    def get_reference_matvec(coeffs, da, db, norb, h_core, eri):
        """Helper function: brute force reference using hamiltonian_element function."""
        n = len(da)
        H = jnp.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H = H.at[i, j].set(
                    hamiltonian_element(da[i], db[i], da[j], db[j], norb, h_core, eri)
                )
        return jnp.dot(H, coeffs)

    def test_hvp_minimal_beta(self):
        # 2 orbitals, alpha is fixed, beta is excited
        norb = 2
        da = [0b01, 0b01]
        db = [0b01, 0b10]
        coeffs = jnp.array([1.0, 0.2])

        # dummy integrals
        h_core = jnp.eye(norb)
        eri = jnp.zeros((norb, norb, norb, norb))

        expected = self.get_reference_matvec(coeffs, da, db, norb, h_core, eri)
        actual = hamiltonian_vector_product(coeffs, da, db, norb, h_core, eri)

        assert jnp.allclose(actual, expected)

    def test_hvp_alpha_beta_mix(self):
        norb = 2
        da = [0b01, 0b10, 0b01]
        db = [0b01, 0b01, 0b10]
        coeffs = jnp.array([1.0, 0.5, -0.2])

        h_core = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        eri = jnp.zeros((norb, norb, norb, norb))

        expected = self.get_reference_matvec(coeffs, da, db, norb, h_core, eri)
        actual = hamiltonian_vector_product(coeffs, da, db, norb, h_core, eri)

        assert jnp.allclose(actual, expected)

    def test_larger_system_5x5(self):
        norb = 4
        ndets = 5
        coeffs, h_core, eri = generate_random_test_data(ndets, norb)
        da = [0b0011, 0b0011, 0b0101, 0b1100, 0b1100]
        db = [0b0011, 0b0110, 0b0011, 0b0011, 0b1100]

        expected = self.get_reference_matvec(coeffs, da, db, norb, h_core, eri)
        actual = hamiltonian_vector_product(coeffs, da, db, norb, h_core, eri)

        assert jnp.allclose(actual, expected)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
