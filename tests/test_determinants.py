"""
Tests for determinant operations using bitstring representations
"""
import os
import sys

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import jax.numpy as jnp
import pytest

from cipsipy.determinants import (
    Wavefunction,
    annihilate,
    apply_double_excitation,
    apply_single_excitation,
    construct_A,
    count_electrons,
    create,
    create_determinant,
    find_connected_internal_determinants_beta,
    find_connected_internal_determinants_oppositespin,
    generate_double_excited_determinants,
    generate_single_excited_determinants,
    get_creation_pair,
    get_creation_pairs,
    get_occupied_indices,
    phase_double,
    phase_single,
    radix_sort_rec,
    sort_wavefunction,
    sort_wavefunction_jax,
    get_det_subset_size,
    is_spinorbital_occupied,
    spinorb2spatorb_det,
    spatorb2spinorb_det
)

class TestCreationPair:
    """Test helper that detects two-bit creation pairs between determinants."""

    def test_get_creation_pair_valid_two_creations(self):
        # G has orbitals 0 and 3 occupied, S has 0,1,2,3 occupied.
        pair = get_creation_pair(0b1001, 0b1111, 4)

        assert pair == (1, 2)

    def test_get_creation_pair_none_when_no_difference(self):
        pair = get_creation_pair(0b1010, 0b1010, 4)

        assert pair is None

    def test_get_creation_pair_none_when_single_bit_diff(self):
        pair = get_creation_pair(0b1000, 0b1100, 4)

        assert pair is None

    def test_get_creation_pair_none_when_more_than_two_bits_differ(self):
        pair = get_creation_pair(0b0000, 0b1110, 4)

        assert pair is None

    def test_get_creation_pair_none_when_bits_are_removed_not_created(self):
        # S differs by two bits, but they are occupied in G and empty in S.
        pair = get_creation_pair(0b1110, 0b1000, 4)

        assert pair is None

    def test_get_creation_pair_handles_beta_block_indices(self):
        # norb=4 so beta block starts at index 4.
        # Creations happen at spin-orbitals 5 and 7.
        pair = get_creation_pair(0b00000001, 0b10100001, 4)

        assert pair == (5, 7)


class TestCreationPairs:
    """Case-based tests for all (r, s) pairs yielding <= double excitation to S.

    We use the local shorthand
    A = occ(S) \ occ(G_pq)
    R = occ(G_pq) \ occ(S)
    where occ(det) is the set of occupied spin-orbitals in a determinant.

    TODO proofread these tests more carefully and make sure there are no remaining edge cases
    """

    def test_case_2_added_0_removed_all_virtual_pairs_valid(self):
        # |A|=2, |R|=0. Every virtual pair in G_pq is valid.
        norb = 3  # 6 spin-orbitals
        G_pq = 0b000001
        S = 0b000111

        pairs = get_creation_pairs(G_pq, S, norb)

        virtual = [1, 2, 3, 4, 5]
        expected = [(virtual[i], virtual[j]) for i in range(len(virtual)) for j in range(i + 1, len(virtual))]
        assert pairs == expected

    def test_case_2_added_0_removed_contains_zeroth_order_pair(self):
        # A={1,2}, R={} so the exact pair that reproduces S must be present.
        norb = 3
        G_pq = 0b000001
        S = 0b000111

        pairs = get_creation_pairs(G_pq, S, norb)
        exact = get_creation_pair(G_pq, S, norb)

        assert exact == (1, 2)
        assert exact in pairs

    def test_case_3_added_1_removed_requires_at_least_one_added_orbital(self):
        # Here A={1,2,3} and R={4}, so valid pairs must include at least one
        # orbital from A.
        norb = 4  # 8 spin-orbitals
        G_pq = 0b00010001
        S = 0b00001111

        pairs = get_creation_pairs(G_pq, S, norb)

        A = {1, 2, 3}
        virtual = [1, 2, 3, 5, 6, 7]  # all unoccupied in G_pq
        expected = []
        for i, r in enumerate(virtual):
            for s in virtual[i + 1:]:
                if (r in A) or (s in A):
                    expected.append((r, s))

        assert pairs == expected

    def test_case_4_added_2_removed_requires_both_added_orbitals(self):
        # Here A={1,2,3,4} and R={5,6}, so valid pairs must lie fully in A.
        norb = 4  # 8 spin-orbitals
        G_pq = 0b01100001
        S = 0b00011111

        pairs = get_creation_pairs(G_pq, S, norb)

        A = [1, 2, 3, 4]
        expected = [(A[i], A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
        assert pairs == expected

    def test_more_than_four_differences_returns_empty(self):
        # Here A={1,2,3,4,5} and R={7,8,9}, so the determinants are too far apart.
        norb = 5  # 10 spin-orbitals
        G_pq = 0b1110000001
        S = 0b0000111111

        pairs = get_creation_pairs(G_pq, S, norb)

        assert pairs == []

    def test_result_is_sorted_and_canonical(self):
        norb = 4
        G_pq = 0b00010001
        S = 0b00001111

        pairs = get_creation_pairs(G_pq, S, norb)

        assert pairs == sorted(set(pairs))
        assert all(r < s for r, s in pairs)

class TestDetSubsetSize:
    """Test cutoff-based determinant subset sizing."""

    def test_get_det_subset_size_one_item(self):
        # |c|^2 = [1.0]
        # cumulative = [1.0]
        coeffs = jnp.array([1.0])

        n1, n2 = get_det_subset_size(coeffs, cutoff1=0.5, cutoff2=1.8)

        assert n1 == 1
        assert n2 == 1

    def test_get_det_subset_size_basic_thresholds(self):
        # |c|^2 = [0.64, 0.16, 0.09, 0.04]
        # cumulative = [0.64, 0.80, 0.89, 0.90]
        coeffs = jnp.array([0.8, 0.4, 0.3, 0.2])

        n1, n2 = get_det_subset_size(coeffs, cutoff1=0.75, cutoff2=0.88)

        assert n1 == 2
        assert n2 == 3

    def test_get_det_subset_size_exact_boundary(self):
        # |c|^2 = [0.64, 0.36, 0.0]
        # cumulative = [0.64, 1.00, 1.00]
        coeffs = jnp.array([0.8, 0.6, 0.0])

        n1, n2 = get_det_subset_size(coeffs, cutoff1=0.64, cutoff2=1.0)

        assert n1 == 1
        assert n2 == 2

    def test_get_det_subset_size_complex_coeffs(self):
        # |c|^2 = [0.50, 0.25, 0.04]
        # cumulative = [0.50, 0.75, 0.79]
        coeffs = jnp.array([0.5 + 0.5j, 0.5j, 0.2 + 0.0j])

        n1, n2 = get_det_subset_size(coeffs, cutoff1=0.50, cutoff2=0.74)

        assert n1 == 1
        assert n2 == 2

    def test_get_det_subset_size_cutoff_above_total(self):
        # total |c|^2 = 0.36 + 0.09 = 0.45
        coeffs = jnp.array([0.6, 0.3])

        n1, n2 = get_det_subset_size(coeffs, cutoff1=0.9, cutoff2=1.2)

        assert n1 == len(coeffs)
        assert n2 == len(coeffs)


class TestBitstringOperations:
    """Test basic bitstring conversion and counting operations"""

    def test_spatorb2spinorb_det(self):
        a=0b0011
        b=0b0101
        c=0b01010011
        n=4
        d = spatorb2spinorb_det(a, b, n)
        assert d == c

    def test_spinorb2spatorb_det(self):
        spin_det = 0b01010011
        n = 4

        det_alpha, det_beta = spinorb2spatorb_det(spin_det, n)

        assert det_alpha == 0b0011
        assert det_beta == 0b0101

    def test_spinorb_spatorb_det_roundtrip(self):
        det_alpha = 0b101001
        det_beta = 0b011010
        n = 6

        spin_det = spatorb2spinorb_det(det_alpha, det_beta, n)
        out_alpha, out_beta = spinorb2spatorb_det(spin_det, n)

        assert out_alpha == det_alpha
        assert out_beta == det_beta

    def test_spinorb2spatorb_det_beta_only(self):
        n = 4
        spin_det = 0b10100000

        det_alpha, det_beta = spinorb2spatorb_det(spin_det, n)

        assert det_alpha == 0
        assert det_beta == 0b1010

    def test_is_spinorbital_occupied(self):
        a=0b0011
        b=0b0101
        n=4
        assert is_spinorbital_occupied(a, b, 4, n)
        assert not is_spinorbital_occupied(a, b, 7, n)
        assert is_spinorbital_occupied(a, b, 1, n)
        assert not is_spinorbital_occupied(a, b, 2, n)

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
        coeffs = [0.3, -0.1, 0.7, 0.2, -0.4]
        dets_alpha = [3, 1, 1, 2, 3]
        dets_beta = [2, 3, 1, 0, 1]

        coeffs_sorted, alpha_sorted, beta_sorted, _ = sort_wavefunction(
            coeffs, dets_alpha, dets_beta, norb=5
        )

        expected = sorted(zip(dets_alpha, dets_beta, coeffs), key=lambda x: (x[0], x[1], x[2]))
        expected_pairs = [(a, b) for a, b, _ in expected]
        expected_coeffs = [c for _, _, c in expected]

        assert list(zip(alpha_sorted, beta_sorted)) == expected_pairs
        assert jnp.allclose(
            coeffs_sorted,
            jnp.array(expected_coeffs, dtype=coeffs_sorted.dtype),
            rtol=0.0,
            atol=jnp.finfo(coeffs_sorted.dtype).eps,
        )

    def test_sort_wavefunction_keeps_coefficient_pairing(self):
        """sort_wavefunction reorders coefficients consistently with determinant sort."""
        coeffs = jnp.array([1.0, 2.0, 3.0, 4.0])
        dets_alpha = jnp.array([2, 1, 2, 1])
        dets_beta = jnp.array([0, 3, 1, 2])

        coeffs_sorted, alpha_sorted, beta_sorted, _ = sort_wavefunction(
            coeffs, dets_alpha, dets_beta, norb=4
        )

        expected = sorted(
            zip(dets_alpha.tolist(), dets_beta.tolist(), coeffs.tolist()),
            key=lambda x: (x[0], x[1], x[2]),
        )
        expected_alpha = [a for a, _, _ in expected]
        expected_beta = [b for _, b, _ in expected]
        expected_coeffs = [c for _, _, c in expected]

        assert alpha_sorted.tolist() == expected_alpha
        assert beta_sorted.tolist() == expected_beta
        assert jnp.allclose(
            coeffs_sorted,
            jnp.array(expected_coeffs, dtype=coeffs_sorted.dtype),
            rtol=0.0,
            atol=jnp.finfo(coeffs_sorted.dtype).eps,
        )

    def test_sort_wavefunction_raises_for_empty_wavefunction(self):
        """Empty wavefunction input should raise ValueError."""
        with pytest.raises(ValueError, match="Wavefunction cannot be empty"):
            sort_wavefunction([], [], [], norb=3)

    def test_sort_wavefunction_jax_matches_python_sort(self):
        """JAX sort path gives same alpha-major ordering and pairing as reference."""
        coeffs = jnp.array([0.3, -0.1, 0.7, 0.2, -0.4])
        dets_alpha = jnp.array([3, 1, 1, 2, 3])
        dets_beta = jnp.array([2, 3, 1, 0, 1])

        coeffs_sorted, alpha_sorted, beta_sorted, _ = sort_wavefunction_jax(
            coeffs, dets_alpha, dets_beta, norb=3
        )

        expected = sorted(
            zip(dets_alpha.tolist(), dets_beta.tolist(), coeffs.tolist()),
            key=lambda x: (x[0], x[1]),
        )
        expected_alpha = [a for a, _, _ in expected]
        expected_beta = [b for _, b, _ in expected]
        expected_coeffs = [c for _, _, c in expected]

        assert alpha_sorted.tolist() == expected_alpha
        assert beta_sorted.tolist() == expected_beta
        assert jnp.allclose(
            coeffs_sorted,
            jnp.array(expected_coeffs, dtype=coeffs_sorted.dtype),
            rtol=0.0,
            atol=jnp.finfo(coeffs_sorted.dtype).eps,
        )

    def test_sort_wavefunction_jax_handles_single_element(self):
        """Single-element inputs are returned unchanged."""
        coeffs = jnp.array([1.5])
        dets_alpha = jnp.array([5])
        dets_beta = jnp.array([2])

        coeffs_sorted, alpha_sorted, beta_sorted, _ = sort_wavefunction_jax(
            coeffs, dets_alpha, dets_beta, norb=3
        )

        assert coeffs_sorted.tolist() == [1.5]
        assert alpha_sorted.tolist() == [5]
        assert beta_sorted.tolist() == [2]


class TestConnectedDeterminants:
    """Test helper/index construction and connected determinant search."""

    def test_construct_A_with_repeated_blocks(self):
        """construct_A returns start indices of each unique alpha block plus sentinel."""
        sorted_alpha = [1, 1, 2, 2, 2, 5]

        a_indices = construct_A(sorted_alpha)

        assert a_indices == [0, 2, 5, 6]

    def test_construct_A_all_unique(self):
        """construct_A handles fully unique alpha lists."""
        sorted_alpha = [1, 2, 3]

        a_indices = construct_A(sorted_alpha)

        assert a_indices == [0, 1, 2, 3]

    def test_find_connected_internal_determinants_beta(self):
        """Finds determinant pairs with <=2 beta excitation level in each alpha block."""
        # already alpha-major sorted
        dets_alpha = [0b0001, 0b0001, 0b0001, 0b0010, 0b0010]
        dets_beta = [0b0011, 0b0101, 0b1100, 0b0011, 0b0110]

        A_indices = construct_A(dets_alpha)

        connected_indices = list(
            find_connected_internal_determinants_beta(dets_alpha, dets_beta, A_indices)
        )

        assert connected_indices == [(0, 1), (0, 2), (1, 2), (3, 4)]

    def test_find_connected_internal_determinants_oppositespin(self):
        """Finds opposite-spin connected pairs: alpha single + beta single."""
        # two alpha blocks, single-connected in alpha
        dets_alpha = [0b0011, 0b0011, 0b0101, 0b0101]
        dets_beta = [0b0011, 0b0101, 0b0110, 0b1001]

        A_indices = construct_A(dets_alpha)

        connected_indices = list(
            find_connected_internal_determinants_oppositespin(dets_alpha, dets_beta, A_indices)
        )

        assert connected_indices == [(0, 2), (0, 3), (1, 2), (1, 3)]

    def test_find_connected_internal_determinants_oppositespin_rejects_non_single_alpha(self):
        """Returns no pairs when alpha blocks are not single-connected."""
        # alpha excitation level between 0b0011 and 0b1100 is 2 (not opposite-spin connected)
        dets_alpha = [0b0011, 0b0011, 0b1100, 0b1100]
        dets_beta = [0b0011, 0b0101, 0b0110, 0b1001]

        A_indices = construct_A(dets_alpha)

        connected_indices = list(
            find_connected_internal_determinants_oppositespin(dets_alpha, dets_beta, A_indices)
        )

        assert connected_indices == []


class TestWavefunctionClass:
    def test_wavefunction_length_validation(self):
        with pytest.raises(ValueError, match="same length"):
            Wavefunction(
                coeffs=jnp.array([1.0, 2.0]),
                dets_alpha=jnp.array([1]),
                dets_beta=jnp.array([1, 2]),
                norb=2,
            )

    def test_sorted_and_sorted_jax_match_functional_apis(self):
        coeffs = jnp.array([0.3, -0.1, 0.7, 0.2, -0.4])
        dets_alpha = jnp.array([3, 1, 1, 2, 3])
        dets_beta = jnp.array([2, 3, 1, 0, 1])
        wf = Wavefunction(coeffs=coeffs, dets_alpha=dets_alpha, dets_beta=dets_beta, norb=5)

        func_coeffs, func_alpha, func_beta, func_idx = sort_wavefunction(
            coeffs, dets_alpha, dets_beta, norb=5
        )
        wf_sorted, idx = wf.alpha_sorted()

        assert jnp.allclose(wf_sorted.coeffs, func_coeffs)
        assert jnp.array_equal(wf_sorted.dets_alpha, func_alpha)
        assert jnp.array_equal(wf_sorted.dets_beta, func_beta)
        assert list(idx) == list(func_idx)

        func_coeffs_j, func_alpha_j, func_beta_j, func_idx_j = sort_wavefunction_jax(
            coeffs, dets_alpha, dets_beta, norb=5
        )
        wf_sorted_j, idx_j = wf.alpha_sorted_jax()

        assert jnp.allclose(wf_sorted_j.coeffs, func_coeffs_j)
        assert jnp.array_equal(wf_sorted_j.dets_alpha, func_alpha_j)
        assert jnp.array_equal(wf_sorted_j.dets_beta, func_beta_j)
        assert jnp.array_equal(idx_j, func_idx_j)

    def test_find_connected_internal_determinants_oppositespin_rejects_non_single_beta(self):
        """Returns no pairs when beta determinants are not single-connected."""
        dets_alpha = [0b0011, 0b0011, 0b0101, 0b0101]
        # any cross-block pair has beta excitation level 2
        dets_beta = [0b0011, 0b0011, 0b1100, 0b1100]

        A_indices = construct_A(dets_alpha)

        connected_indices = list(
            find_connected_internal_determinants_oppositespin(dets_alpha, dets_beta, A_indices)
        )

        assert connected_indices == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
