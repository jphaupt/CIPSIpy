"""
Tests for determinant operations using bitstring representations
"""

import os
import sys

import jax.numpy as jnp
import pytest

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.determinants import (
    annihilate,
    apply_double_excitation,
    apply_single_excitation,
    count_electrons,
    create,
    create_determinant,
    get_occupied_indices,
    phase_double,
    phase_single,
)


class TestBitstringOperations:
    """Test basic bitstring conversion and counting operations"""

    def test_get_occupied_indices_simple(self):
        """Test conversion of integer to occupation mask"""
        # 3 in binary is 0b11 = [1, 1, 0, 0] for 4 orbitals
        det = 3  # 0b0011
        n_orb = 4
        occ = get_occupied_indices(det, n_orb)
        expected = jnp.array([1, 1, 0, 0])
        assert jnp.array_equal(occ, expected)

    def test_get_occupied_indices_empty(self):
        """Test empty determinant (vacuum state)"""
        det = 0  # 0b0000
        n_orb = 4
        occ = get_occupied_indices(det, n_orb)
        expected = jnp.array([0, 0, 0, 0])
        assert jnp.array_equal(occ, expected)

    def test_get_occupied_indices_full(self):
        """Test fully occupied determinant"""
        det = 15  # 0b1111
        n_orb = 4
        occ = get_occupied_indices(det, n_orb)
        expected = jnp.array([1, 1, 1, 1])
        assert jnp.array_equal(occ, expected)

    def test_get_occupied_indices_sparse(self):
        """Test sparse occupation"""
        det = 5  # 0b0101 (orbitals 0 and 2)
        n_orb = 4
        occ = get_occupied_indices(det, n_orb)
        expected = jnp.array([1, 0, 1, 0])
        assert jnp.array_equal(occ, expected)

    def test_count_electrons(self):
        """Test electron counting"""
        assert count_electrons(0) == 0
        assert count_electrons(1) == 1  # 0b0001
        assert count_electrons(3) == 2  # 0b0011
        assert count_electrons(7) == 3  # 0b0111
        assert count_electrons(15) == 4  # 0b1111
        assert count_electrons(5) == 2  # 0b0101

    def test_create_determinant(self):
        """Test creating determinant from occupation list"""
        # Orbitals 0 and 1 occupied
        occ_indices = [0, 1]
        n_orb = 4
        det = create_determinant(occ_indices, n_orb)
        assert det == 3  # 0b0011

        # Orbitals 0 and 2 occupied
        occ_indices = [0, 2]
        det = create_determinant(occ_indices, n_orb)
        assert det == 5  # 0b0101

    def test_create_determinant_empty(self):
        """Test creating empty determinant"""
        occ_indices = []
        n_orb = 4
        det = create_determinant(occ_indices, n_orb)
        assert det == 0

    def test_roundtrip_conversion(self):
        """Test that create -> get_occupied -> create gives same result"""
        occ_indices = [0, 2, 3]
        n_orb = 5
        det = create_determinant(occ_indices, n_orb)
        occ_mask = get_occupied_indices(det, n_orb)
        # Convert mask back to indices
        recovered_indices = jnp.where(occ_mask)[0]
        assert jnp.array_equal(recovered_indices, jnp.array(occ_indices))


class TestAnnihilationCreation:
    """Test annihilation and creation operators"""

    def test_annihilate_occupied(self):
        """Test annihilating electron from occupied orbital"""
        det = 3  # 0b0011 (orbitals 0 and 1 occupied)
        new_det = annihilate(det, 0)
        assert new_det == 2  # 0b0010 (orbital 1 only)

    def test_annihilate_unoccupied(self):
        """Test annihilating from unoccupied orbital returns 0"""
        det = 3  # 0b0011 (orbitals 0 and 1 occupied)
        new_det = annihilate(det, 2)
        assert new_det == 0  # Invalid operation

    def test_create_unoccupied(self):
        """Test creating electron in unoccupied orbital"""
        det = 3  # 0b0011 (orbitals 0 and 1 occupied)
        new_det = create(det, 2)
        assert new_det == 7  # 0b0111

    def test_create_occupied(self):
        """Test creating in occupied orbital returns 0 (Pauli exclusion)"""
        det = 3  # 0b0011 (orbitals 0 and 1 occupied)
        new_det = create(det, 0)
        assert new_det == 0  # Invalid operation

    def test_create_then_annihilate(self):
        """Test c†c on same orbital"""
        det = 2  # 0b0010
        new_det = create(det, 0)  # 0b0011
        new_det = annihilate(new_det, 0)  # back to 0b0010
        # Note: phase factors would need to be tracked separately
        assert new_det == 2


class TestSingleExcitations:
    """Test single excitation operations"""

    def test_phase_single_simple(self):
        """Test phase for simple single excitation"""
        # For |01⟩ (orbital 0 occupied), excite 0 -> 1
        det = 1  # 0b01
        phase = phase_single(det, 0, 1)
        # Phase is (-1)^(number of electrons between i and a)
        assert phase == 1 or phase == -1

    def test_apply_single_excitation(self):
        """Test applying single excitation"""
        # Start with |01⟩ (orbital 0 occupied)
        det = 1  # 0b0001
        new_det, phase = apply_single_excitation(det, 0, 1)
        # Should give |10⟩ (orbital 1 occupied)
        assert new_det == 2  # 0b0010

    def test_apply_single_excitation_invalid(self):
        """Test invalid single excitation returns zero"""
        # Try to excite from unoccupied orbital
        det = 1  # 0b0001 (orbital 0 occupied)
        new_det, phase = apply_single_excitation(det, 1, 2)
        assert new_det == 0
        assert phase == 0

    def test_single_excitation_h2(self):
        """Test single excitation for H2-like system"""
        # |0011⟩ -> |0110⟩ (excite orbital 0 to orbital 2)
        det = 3  # 0b0011 (orbitals 0 and 1 occupied)
        new_det, phase = apply_single_excitation(det, 0, 2)
        assert new_det == 6  # 0b0110 (orbitals 1 and 2 occupied)


class TestDoubleExcitations:
    """Test double excitation operations"""

    def test_phase_double_simple(self):
        """Test phase for simple double excitation"""
        det = 3  # 0b0011 (orbitals 0, 1 occupied)
        phase = phase_double(det, 0, 1, 2, 3)
        assert phase == 1 or phase == -1

    def test_apply_double_excitation(self):
        """Test applying double excitation"""
        # Start with |0011⟩ (orbitals 0, 1 occupied)
        det = 3  # 0b0011
        new_det, phase = apply_double_excitation(det, 0, 1, 2, 3)
        # Should give |1100⟩ (orbitals 2, 3 occupied)
        assert new_det == 12  # 0b1100

    def test_apply_double_excitation_invalid(self):
        """Test invalid double excitation returns zero"""
        # Try to excite from partially unoccupied orbitals
        det = 1  # 0b0001 (only orbital 0 occupied)
        new_det, phase = apply_double_excitation(det, 0, 1, 2, 3)
        assert new_det == 0
        assert phase == 0

    def test_double_excitation_overlap(self):
        """Test double excitation where holes and particles overlap"""
        # This should be invalid
        det = 3  # 0b0011
        new_det, phase = apply_double_excitation(det, 0, 1, 1, 2)
        # orbital 1 is both a hole and particle - invalid
        assert new_det == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
