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
    generate_double_excited_determinants,
    generate_single_excited_determinants,
    get_occupied_indices,
    phase_double,
    phase_single,
    radix_sort_rec,
    sort_determinants,
    sort_determinants_jax,
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


class TestExcitationGenerators:
    """Test generation of single/double excited determinants."""

    def test_generate_single_excited_determinants(self):
        """Singles count and targets are correct."""
        det = 0b0011  # occupied {0,1}
        n_orb = 4

        singles = generate_single_excited_determinants(det, n_orb)
        targets = set(singles)

        # n_occ * n_virt = 2 * 2 = 4 singles
        assert len(singles) == 4
        assert targets == {0b0110, 0b1010, 0b0101, 0b1001}

    def test_generate_double_excited_determinants(self):
        """Doubles count and target are correct."""
        det = 0b0011  # occupied {0,1}
        n_orb = 4

        doubles = generate_double_excited_determinants(det, n_orb)

        # C(2,2) * C(2,2) = 1 double
        assert len(doubles) == 1
        assert doubles[0] == 0b1100

        det = 0b01011
        n_orb = 5

        doubles = generate_double_excited_determinants(det, n_orb)
        targets = set(doubles)

        assert targets == {0b10110, 0b10101, 0b11100}


class TestDeterminantSorting:
    """Test determinant sorting utilities."""

    def test_radix_sort_rec_returns_sorted_values_and_indices(self):
        """radix_sort_rec sorts values and carries original indices."""
        dets = [6, 3, 1, 4]  # unsorted

        sorted_dets, sorted_idx = radix_sort_rec(dets, i=2)

        assert sorted_dets == [1, 3, 4, 6]
        assert sorted_idx == [2, 1, 3, 0]

    def test_radix_sort_rec_empty_input(self):
        """Empty input is handled without error."""
        sorted_dets, sorted_idx = radix_sort_rec([], i=3)

        assert sorted_dets == []
        assert sorted_idx == []

    def test_sort_determinants_alpha_major_order(self):
        """sort_determinants orders by alpha first, then beta."""
        dets_alpha = [3, 1, 1, 2, 3]
        dets_beta = [2, 3, 1, 0, 1]

        alpha_sorted, beta_sorted = sort_determinants(dets_alpha, dets_beta, norb=3)

        expected_pairs = sorted(zip(dets_alpha, dets_beta), key=lambda x: (x[0], x[1]))
        assert list(zip(alpha_sorted, beta_sorted)) == expected_pairs

    def test_sort_determinants_jax_matches_python_sort(self):
        """JAX sort path gives same alpha-major ordering as Python reference."""
        dets_alpha = jnp.array([3, 1, 1, 2, 3])
        dets_beta = jnp.array([2, 3, 1, 0, 1])

        alpha_sorted, beta_sorted = sort_determinants_jax(dets_alpha, dets_beta, norb=3)

        expected_pairs = sorted(
            zip(dets_alpha.tolist(), dets_beta.tolist()), key=lambda x: (x[0], x[1])
        )
        assert list(zip(alpha_sorted.tolist(), beta_sorted.tolist())) == expected_pairs

    def test_sort_determinants_jax_handles_single_element(self):
        """Single-element inputs are returned unchanged."""
        dets_alpha = jnp.array([5])
        dets_beta = jnp.array([2])

        alpha_sorted, beta_sorted = sort_determinants_jax(dets_alpha, dets_beta, norb=3)

        assert alpha_sorted.tolist() == [5]
        assert beta_sorted.tolist() == [2]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
