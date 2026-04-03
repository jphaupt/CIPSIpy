"""
Slater determinant operations using bitstring representations.

This module provides efficient bitstring-based operations for quantum chemistry
determinants using JAX for GPU acceleration.

Determinants are represented as integers where each bit represents an orbital:
    - Bit i = 1 means orbital i is occupied
    - Bit i = 0 means orbital i is unoccupied
    - Example: 3 (binary 0b0011) = orbitals 0 and 1 occupied

Separate integers are used for alpha and beta spin electrons.
"""

from dataclasses import dataclass
from itertools import combinations

from typing import Tuple, Optional
import jax.numpy as jnp

# ============================================================================
# Utility functions
# ============================================================================

def spinorb2spatorb(p_so: int, norb: int) -> tuple[int, bool]:
    """
    converts spinorbital index to spatial orbital index, given norb spatial
    orbitals. It also returns True if the p_so refers to an electron in the alpha
    block, False for the beta block
    Note we store alpha then beta, so for example
    spinorb2spatorb(6, 4) -> 2, False
    spinorb2spatorb(2, 4) -> 2, True
    """
    if p_so < norb:
        return p_so, True
    else:
        return p_so - norb, False

def get_det_subset_size(coeffs, cutoff1, cutoff2):
    """
    Given coeffs c_i, get two integer values N1 N2 corresponding to the cutoffs
    cutoff1 = \sum_{i=1}^N |c_i|^2
    Precondition: coeffs are sorted according to |c_i|^2
    """
    cumsum = jnp.cumsum(jnp.abs(coeffs)**2)
    N1 = jnp.searchsorted(cumsum, cutoff1 * cumsum[-1])
    N2 = jnp.searchsorted(cumsum, cutoff2 * cumsum[-1])
    return min(int(N1)+1, len(coeffs)), min(int(N2)+1, len(coeffs)) # plus one to correct from 0-based indexing

# ============================================================================
# Helper functions for bitwise operations
# ============================================================================

def get_creation_pair(G_pq: int, S: int, norb: int) -> Optional[Tuple[int, int]]:
    """
    Check if there exists a pair (r,s) such that G_pq^rs = S

    G_pq and S are determinants in spinorbitals, i.e. length 2*norb

    Returns
        Tuple (r, s) if such a pair exists, None otherwise
    """
    diff = S ^ G_pq
    positions = []
    for i in range(2 * norb):
        if (diff >> i) & 1:
            positions.append(i)
    if len(positions) == 2:
        # check set in Sdet but not in G_pq
        r, s = positions[0], positions[1]
        if ((S >> r) & 1) and ((S >> s) & 1):
            if not ((G_pq >> r) & 1) and not ((G_pq >> s) & 1):
                return (r, s)
    return None


def get_creation_pairs(G_pq: int, S: int, norb: int) -> list[Tuple[int, int]]:
    """
    Return all creation pairs (r, s) such that |G_pq^(rs)> is at most a double
    excitation away from |S>.

    The logic follows the case split described in Garniron's thesis.

    We use local shorthand here:
        A = occ(S) \ occ(G_pq)
        R = occ(G_pq) \ occ(S)
    where occ(det) is the set of occupied spin-orbitals in ``det``.

    This ``A``/``R`` notation is only explanatory for the implementation; it is
    not intended to mirror a named symbol from the thesis.

    Returns canonical pairs with r < s in ascending lexicographic order.

    TODO also calculate the excitation operator T s.t. |S⟩=±T|G_pq^rs⟩
    """
    n_spinorb = 2 * norb

    added = []
    removed = []
    virtual_in_gpq = []
    for i in range(n_spinorb):
        g_occ = (G_pq >> i) & 1
        s_occ = (S >> i) & 1
        if s_occ and not g_occ:
            added.append(i)
        elif g_occ and not s_occ:
            removed.append(i)
        if not g_occ:
            virtual_in_gpq.append(i)

    # Expected precondition for doubly-ionized G_pq relative to S.
    if len(added) - len(removed) != 2:
        return []

    pairs: list[Tuple[int, int]] = []
    added_set = set(added)
    outside_added_virtual = [i for i in virtual_in_gpq if i not in added_set]

    # Case 1: |A|=2, |R|=0 -> all virtual pairs are allowed.
    if len(added) == 2 and len(removed) == 0:
        pairs.extend(combinations(virtual_in_gpq, 2))
    # Case 2: |A|=3, |R|=1 -> at least one created orbital must come from A.
    elif len(added) == 3 and len(removed) == 1:
        pairs.extend(combinations(added, 2))
        for r in added:
            for s in outside_added_virtual:
                pairs.append((r, s) if r < s else (s, r))
    # Case 3: |A|=4, |R|=2 -> both created orbitals must come from A.
    elif len(added) == 4 and len(removed) == 2:
        pairs.extend(combinations(added, 2))
    # Case 4: more differences -> not connected by <= double excitation.
    else:
        return []

    # Normalize and sort deterministically.
    return sorted(set(pairs))

def get_excitation_level(det_i, det_j):
    """
    get excitation level between bitstring-represented determinants det_i and det_j
    i.e. 1/2 ||I⊕J||
    """
    xor_det = det_i ^ det_j
    return count_electrons(xor_det) // 2

def spatorb2spinorb_det(det_alpha, det_beta, norb):
    return (det_beta << norb) | det_alpha


def spinorb2spatorb_det(det_spinorb, norb):
    """Split a spin-orbital determinant into (alpha, beta) spatial blocks.

    Determinants are stored as ``[alpha block | beta block]`` where the low
    ``norb`` bits are alpha occupations and the next ``norb`` bits are beta
    occupations.
    """
    mask = (1 << norb) - 1
    det_alpha = det_spinorb & mask
    det_beta = det_spinorb >> norb
    return det_alpha, det_beta

def is_spinorbital_occupied(det_alpha, det_beta, spinorb, norb):
    """
    given two bitstring determinants for alpha and beta blocks, determine if the
    *spin*-orbital spinorb is occupied
    """
    det = spatorb2spinorb_det(det_alpha, det_beta, norb)
    return is_orbital_occupied(det, spinorb)


def is_orbital_occupied(det_int, orbital_idx):
    """
    Check if an orbital is occupied in a determinant.

    Args:
        det_int: Integer representation of determinant
        orbital_idx: Orbital index to check

    Returns:
        True if orbital is occupied, False otherwise
    """
    return bool(det_int & (1 << orbital_idx))


def set_orbital_bit(det_int, orbital_idx):
    """
    Set the bit for an orbital (mark as occupied).

    Args:
        det_int: Integer representation of determinant
        orbital_idx: Orbital index to set

    Returns:
        New determinant with orbital bit set
    """
    return det_int | (1 << orbital_idx)


def clear_orbital_bit(det_int, orbital_idx):
    """
    Clear the bit for an orbital (mark as unoccupied).

    Args:
        det_int: Integer representation of determinant
        orbital_idx: Orbital index to clear

    Returns:
        New determinant with orbital bit cleared
    """
    return det_int & ~(1 << orbital_idx)


# ============================================================================
# Core determinant operations
# ============================================================================

def construct_A(sorted_alpha):
    """
    helper function for finding connected internal determinants, called A in the
    thesis by garniron

    A[n] tells you the index of the nth unique appearance of a value in the sorted
    array sorted_alpha
    """
    A = [0]
    for i in range(1, len(sorted_alpha)):
        if sorted_alpha[i] != sorted_alpha[i-1]:
            A.append(i)
    A.append(len(sorted_alpha))
    return A

def find_connected_internal_determinants_beta(dets_alpha, dets_beta, A_indices):
    """
    algorithm 9 of the thesis - finds all single- and double-connected determinants
    in the list dets. This gives all beta single- and double-excitations (to get
    alpha, just swap the spins)

    PRECONDITION: the arrays are sorted in alpha-major order

    Yields:
        Tuples of determinant index pairs (i, j) that are connected.
    """
    # A_indices = construct_A(dets_alpha)
    # iterate over unique alpha blocks (A stores block starts and a sentinel)
    for a in range(len(A_indices) - 1):
        # all determinants sharing alpha part are in the range A[a], A[a+1)-1
        for b1 in range(A_indices[a], A_indices[a + 1]):
            for b2 in range(b1 + 1, A_indices[a + 1]):
                if get_excitation_level(dets_beta[b1], dets_beta[b2]) <= 2:
                    yield (b1, b2)

def find_connected_internal_determinants_oppositespin(dets_alpha, dets_beta, A_indices):
    """
    algorithm 10 of garniron's thesis: sequential opposite-spin internal det
    connectivity finder, i.e. find all alpha-beta double excitations

    PRECONDITION: determinants are sorted in alpha-major order

    NOTE this is a slow/sequential implementation

    TODO parallelise via JAX (his algorithm 12 is good for CPUs)

    Yields:
        Tuples of determinant index pairs (i, j) that are connected.
    """
    for a1 in range(len(A_indices)-1):
        for a2 in range(a1+1, len(A_indices)-1):
            alpha_1 = dets_alpha[A_indices[a1]]
            alpha_2 = dets_alpha[A_indices[a2]]
            if get_excitation_level(alpha_1, alpha_2) != 1:
                continue
            for b1 in range(A_indices[a1], A_indices[a1+1]):
                for b2 in range(A_indices[a2], A_indices[a2+1]):
                    if get_excitation_level(dets_beta[b1], dets_beta[b2]) == 1:
                        yield (b1, b2)

def radix_sort_rec(dets, i, keys=None) -> tuple[list[int], list[int]]:
    """
    algorithm 11 of the thesis implemented "manually" -- no fancy JAX business
    i.e. it is a standard recursive implementation of the radix sort algorithm
    works for nonnegative integers

    Args:
        dets: The list of integers (determinants) to sort
        i: The index of the bit we are currently inspecting (e.g., 63 down to 0)
        keys: optional list of keys (use None for standard argsort -- used for recursion)

    Returns:
        sorted_dets: the sorted list (dets)
        sorted_keys: original indices, sorted

    note: to sort arbitrary 64-bit integers, use i = 63

    TODO: speed up with parallelism/jax
    """
    # initialize keys on first call
    if keys is None:
        keys = list(range(len(dets)))

    # base case - all bits have been checked or no elements ([] is sorted)
    if i < 0 or len(dets) == 0:
        return dets, keys

    dets0 = []  # pigeonhole for bit == 0
    dets1 = []  # pigeonhole for bit == 1
    keys0 = []
    keys1 = []

    # check the i-th bit of every determinant in dets
    for d, k in zip(dets, keys):
        if is_orbital_occupied(d, i):
            dets1.append(d)
            keys1.append(k)
        else:
            dets0.append(d)
            keys0.append(k)

    dets0_sorted, keys0_sorted = radix_sort_rec(dets0, i - 1, keys0)
    dets1_sorted, keys1_sorted = radix_sort_rec(dets1, i - 1, keys1)

    return dets0_sorted + dets1_sorted, keys0_sorted + keys1_sorted

def sort_wavefunction(coeffs, dets_alpha, dets_beta, norb, sort_alg=radix_sort_rec):
    """
    TODO since we are using recursive radix right now, it does not work well with
        JAX, so we coerce JAX arrays to Python lists, then convert back to JAX arrays
        This is obviously not a great long-term solution, and we need to use JAX
        parallelism

    Sorts the wavefunction in alpha-major order, given a number of spatial
    orbitals norb, using the sorting algorithm sort_alg.
    The coeffs, dets_alpha, and dets_beta are all reordered so they
    remain physically consistent.
    """
    if len(coeffs) == 0:
        raise ValueError("Wavefunction cannot be empty.")

    coeffs = jnp.asarray(coeffs)
    dets_alpha = jnp.asarray(dets_alpha)
    dets_beta = jnp.asarray(dets_beta)

    # pass dets_alpha as keys so it follows dets_beta's movement
    _, keys = sort_alg(dets_beta.tolist(), norb - 1)

    keys_array = jnp.asarray(keys, dtype=jnp.int32)
    alpha_sorted = dets_alpha[keys_array]

    # then primary key
    final_alpha, final_keys = sort_alg(alpha_sorted.tolist(), norb - 1, keys)
    final_keys_array = jnp.asarray(final_keys, dtype=jnp.int32)
    final_beta = dets_beta[final_keys_array]
    final_coeffs = coeffs[final_keys_array]

    return jnp.array(final_coeffs), jnp.array(final_alpha), jnp.array(final_beta), final_keys

def sort_wavefunction_jax(coeffs, dets_alpha, dets_beta, norb):
    """
    Sorts determinants and coeffs in alpha-major order using JAX built-ins.
    lexsort( (secondary_key, primary_key) )
    """
    # lexsort sorts by the last array in the tuple first.
    # To get alpha-major (alpha first, then beta), we pass (beta, alpha).
    idx = jnp.lexsort((dets_beta, dets_alpha))
    return coeffs[idx], dets_alpha[idx], dets_beta[idx], idx

def sort_wavefunction_by_coeffs_jax(coeffs, dets_alpha, dets_beta):
    """
    Sorts determinants and coeffs according to square of coeffs in descending order

    TODO I am quite confused by this: we need the wave function sorted in terms
        of coeffs^2, but this requires NlogN as far as I'm aware, yet we went
        through so much effort with radix sort to sort in O(N) time. Doesn't this
        effectively undermine all that effort?
    """
    idx = jnp.argsort(coeffs**2)[::-1]
    # jnp.argsort(coeffs.real**2 + coeffs.imag**2)[::-1]  # For complex coeffs
    return coeffs[idx], dets_alpha[idx], dets_beta[idx], idx


def get_occupied_indices(det_int, n_orbitals):
    """
    Returns a mask of occupied orbitals.

    Args:
        det_int: Integer representation of determinant
        n_orbitals: Number of orbitals

    Returns:
        Array of shape (n_orbitals,) with 1 for occupied, 0 for unoccupied

    Example:
        3 (binary 0011) -> [1, 1, 0, 0] for 4 orbitals
    """
    idx = jnp.arange(n_orbitals)
    return (det_int >> idx) & 1


def count_electrons(det_int):
    """
    Count the number of electrons (occupied orbitals) in a determinant.

    Args:
        det_int: Integer representation of determinant

    Returns:
        Number of occupied orbitals (set bits)

    Example:
        3 (binary 0011) -> 2 electrons
        7 (binary 0111) -> 3 electrons
    """
    # Count set bits using Brian Kernighan's algorithm
    count = 0
    while det_int:
        det_int &= det_int - 1
        count += 1
    return count


def create_determinant(occ_indices, n_orbitals):
    """
    Create a determinant from a list of occupied orbital indices.

    Args:
        occ_indices: List of occupied orbital indices (0-based)
        n_orbitals: Total number of orbitals

    Returns:
        Integer representation of determinant

    Example:
        [0, 1] -> 3 (binary 0011)
        [0, 2] -> 5 (binary 0101)
    """
    det = 0
    for idx in occ_indices:
        if idx < n_orbitals:
            det = set_orbital_bit(det, idx)
    return det


# ============================================================================
# Annihilation and creation operators
# ============================================================================


def annihilate(det_int, orbital_idx):
    """
    Apply annihilation operator to remove electron from orbital.

    Args:
        det_int: Integer representation of determinant
        orbital_idx: Orbital index to remove electron from

    Returns:
        New determinant with electron removed, or 0 if orbital was unoccupied

    Note:
        This returns 0 for invalid operations (Pauli exclusion principle)
    """
    # Check if orbital is occupied
    if not is_orbital_occupied(det_int, orbital_idx):
        return 0  # Orbital not occupied - invalid operation

    # Remove electron
    return clear_orbital_bit(det_int, orbital_idx)


def create(det_int, orbital_idx):
    """
    Apply creation operator to add electron to orbital.

    Args:
        det_int: Integer representation of determinant
        orbital_idx: Orbital index to add electron to

    Returns:
        New determinant with electron added, or 0 if orbital was occupied

    Note:
        This returns 0 for invalid operations (Pauli exclusion principle)
    """
    # Check if orbital is unoccupied
    if is_orbital_occupied(det_int, orbital_idx):
        return 0  # Orbital already occupied - invalid operation

    # Add electron
    return set_orbital_bit(det_int, orbital_idx)


# ============================================================================
# Phase calculation and excitations
# ============================================================================


def phase_single(det_int, i, a):
    """
    Calculate fermionic phase for single excitation i -> a.

    The phase is (-1)^N where N is the number of occupied orbitals
    between indices i and a.

    Args:
        det_int: Integer representation of determinant
        i: Hole orbital index (occupied, to be annihilated)
        a: Particle orbital index (unoccupied, to be created)

    Returns:
        Phase factor: +1 or -1

    Note:
        Returns 0 if the excitation is invalid
    """
    # Check if i is occupied and a is unoccupied
    if not is_orbital_occupied(det_int, i):
        return 0
    if is_orbital_occupied(det_int, a):
        return 0

    # Ensure i < a for counting
    if i > a:
        i, a = a, i

    # Count occupied orbitals between i and a (exclusive)
    mask = ((1 << a) - 1) & ~((1 << (i + 1)) - 1)
    n_between = count_electrons(det_int & mask)

    return 1 if n_between % 2 == 0 else -1


def apply_single_excitation(det_int, i, a):
    """
    Apply single excitation i -> a to determinant.

    Args:
        det_int: Integer representation of determinant
        i: Hole orbital index (occupied, to be annihilated)
        a: Particle orbital index (unoccupied, to be created)

    Returns:
        Tuple (new_det, phase) where:
            - new_det: New determinant after excitation (0 if invalid)
            - phase: Fermionic phase factor (+1, -1, or 0 if invalid)
    """
    # Calculate phase first (this also validates the excitation)
    phase = phase_single(det_int, i, a)
    if phase == 0:
        return 0, 0

    # Apply the excitation using annihilate and create operators
    # We can safely clear and set bits since phase already validated
    new_det = clear_orbital_bit(det_int, i)
    new_det = set_orbital_bit(new_det, a)

    return new_det, phase


def generate_single_excited_determinants(det_int, n_orbitals):
    """Generate all single-excited determinants

    Args:
        det_int: Integer representation of determinant
        n_orbitals: Number of spatial orbitals

    Returns:
        List of integer determinants reachable by valid single excitations.
    """
    occupied = []
    virtual = []
    for orbital in range(n_orbitals):
        if is_orbital_occupied(det_int, orbital):
            occupied.append(orbital)
        else:
            virtual.append(orbital)

    excited_dets = []
    for i in occupied:
        for a in virtual:
            new_det = clear_orbital_bit(det_int, i)
            new_det = set_orbital_bit(new_det, a)
            excited_dets.append(new_det)

    return excited_dets


def generate_double_excited_determinants(det_int, n_orbitals):
    """Generate all double-excited determinants

    Args:
        det_int: Integer representation of determinant
        n_orbitals: Number of spatial orbitals

    Returns:
        List of integer determinants reachable by valid double excitations.
    """
    occupied = []
    virtual = []
    for orbital in range(n_orbitals):
        if is_orbital_occupied(det_int, orbital):
            occupied.append(orbital)
        else:
            virtual.append(orbital)

    excited_dets = []
    for occ_i in range(len(occupied)):
        i = occupied[occ_i]
        for occ_j in range(occ_i + 1, len(occupied)):
            j = occupied[occ_j]
            for vir_a in range(len(virtual)):
                a = virtual[vir_a]
                for vir_b in range(vir_a + 1, len(virtual)):
                    b = virtual[vir_b]
                    new_det = clear_orbital_bit(det_int, i)
                    new_det = clear_orbital_bit(new_det, j)
                    new_det = set_orbital_bit(new_det, a)
                    new_det = set_orbital_bit(new_det, b)
                    excited_dets.append(new_det)

    return excited_dets


def phase_double(det_int, i, j, a, b):
    """
    Calculate fermionic phase for double excitation i,j -> a,b.

    The phase accounts for the order of annihilation and creation operators.

    Args:
        det_int: Integer representation of determinant
        i, j: Hole orbital indices (occupied, to be annihilated)
        a, b: Particle orbital indices (unoccupied, to be created)

    Returns:
        Phase factor: +1 or -1, or 0 if invalid

    Note:
        Convention: i < j and a < b
    """
    # Check validity
    if not is_orbital_occupied(det_int, i) or not is_orbital_occupied(det_int, j):
        return 0  # Holes must be occupied
    if is_orbital_occupied(det_int, a) or is_orbital_occupied(det_int, b):
        return 0  # Particles must be unoccupied
    if i == j or a == b:
        return 0  # Can't excite same orbital twice

    # Ensure canonical ordering: i < j, a < b
    if i > j:
        i, j = j, i
    if a > b:
        a, b = b, a

    # Calculate phase using the formula for double excitations
    # Phase = (-1)^(sum of fermion hops)

    # Create a temporary determinant after first annihilation
    temp_det = clear_orbital_bit(det_int, i)

    # Count electrons between i and j
    if j > i:
        mask = ((1 << j) - 1) & ~((1 << (i + 1)) - 1)
        n_ij = count_electrons(det_int & mask)
    else:
        n_ij = 0

    # Count electrons between j and a in temp_det
    min_ja = min(j, a)
    max_ja = max(j, a)
    mask = ((1 << max_ja) - 1) & ~((1 << (min_ja + 1)) - 1)
    n_ja = count_electrons(temp_det & mask)

    # Create after first creation
    temp_det = set_orbital_bit(temp_det, a)

    # Count electrons between a and b in temp_det
    if b > a:
        mask = ((1 << b) - 1) & ~((1 << (a + 1)) - 1)
        n_ab = count_electrons(temp_det & mask)
    else:
        n_ab = 0

    total_count = n_ij + n_ja + n_ab
    return 1 if total_count % 2 == 0 else -1


def apply_double_excitation(det_int, i, j, a, b):
    """
    Apply double excitation i,j -> a,b to determinant.

    Args:
        det_int: Integer representation of determinant
        i, j: Hole orbital indices (occupied, to be annihilated)
        a, b: Particle orbital indices (unoccupied, to be created)

    Returns:
        Tuple (new_det, phase) where:
            - new_det: New determinant after excitation (0 if invalid)
            - phase: Fermionic phase factor (+1, -1, or 0 if invalid)
    """
    # Calculate phase first (this also validates the excitation)
    phase = phase_double(det_int, i, j, a, b)
    if phase == 0:
        return 0, 0

    # Apply the excitation using helper functions
    # We can safely clear and set bits since phase already validated
    new_det = clear_orbital_bit(det_int, i)
    new_det = clear_orbital_bit(new_det, j)
    new_det = set_orbital_bit(new_det, a)
    new_det = set_orbital_bit(new_det, b)

    return new_det, phase

@dataclass(frozen=True)
class Wavefunction:
    """Immutable container for CI wavefunction amplitudes and determinants.

    This mirrors the functional API in this module while providing a grouped
    object for coefficients and determinant arrays.
    """

    coeffs: jnp.ndarray
    dets_alpha: jnp.ndarray
    dets_beta: jnp.ndarray
    norb: int

    def __post_init__(self):
        if self.norb < 1:
            raise ValueError("norb must be positive")
        if len(self.coeffs) != len(self.dets_alpha) or len(self.coeffs) != len(self.dets_beta):
            raise ValueError("coeffs, dets_alpha, and dets_beta must have the same length")

    def coeff_sorted(self):
        coeffs_sorted, alpha_sorted, beta_sorted, idx = sort_wavefunction_by_coeffs_jax(
            self.coeffs,
            self.dets_alpha,
            self.dets_beta,
        )
        return (
            Wavefunction(
                coeffs=coeffs_sorted,
                dets_alpha=alpha_sorted,
                dets_beta=beta_sorted,
                norb=self.norb,
            ),
            idx,
        )

    def alpha_sorted(self, sort_alg=radix_sort_rec):
        """Return alpha-major sorted wavefunction and the sort index list."""
        coeffs_sorted, alpha_sorted, beta_sorted, idx = sort_wavefunction(
            self.coeffs,
            self.dets_alpha,
            self.dets_beta,
            self.norb,
            sort_alg=sort_alg,
        )
        return (
            Wavefunction(
                coeffs=coeffs_sorted,
                dets_alpha=alpha_sorted,
                dets_beta=beta_sorted,
                norb=self.norb,
            ),
            idx,
        )

    def alpha_sorted_jax(self):
        """Return alpha-major sorted wavefunction using JAX lexsort."""
        coeffs_sorted, alpha_sorted, beta_sorted, idx = sort_wavefunction_jax(
            self.coeffs,
            self.dets_alpha,
            self.dets_beta,
            self.norb,
        )
        return (
            Wavefunction(
                coeffs=coeffs_sorted,
                dets_alpha=alpha_sorted,
                dets_beta=beta_sorted,
                norb=self.norb,
            ),
            idx,
        )

    def with_coeffs(self, coeffs):
        """Return a new wavefunction with updated coefficients."""
        if len(coeffs) != len(self.dets_alpha):
            raise ValueError("coeffs must have the same length as determinant arrays")
        return Wavefunction(
            coeffs=jnp.asarray(coeffs),
            dets_alpha=self.dets_alpha,
            dets_beta=self.dets_beta,
            norb=self.norb,
        )


# ============================================================================
# JAX-native bitwise helpers (no Python control flow; vmappable under jax.vmap)
# ============================================================================

def jax_popcount(x, norb):
    """Count set bits in x (lowest *norb* bits only).

    Works with JAX scalar integers: vmappable and JIT-compatible.
    ``norb`` must be a concrete Python int at compile time.
    """
    idx = jnp.arange(norb, dtype=jnp.int64)
    return jnp.sum((x.astype(jnp.int64) >> idx) & jnp.int64(1), dtype=jnp.int32)


def excitation_level_jax(det_i, det_j, norb):
    """Excitation level between two JAX-integer determinants (vmappable)."""
    return jax_popcount(det_i ^ det_j, norb) >> jnp.int32(1)


def first_set_bit_pos_jax(x, norb):
    """Index of the lowest set bit in *x* (0-indexed). JAX-native, vmappable."""
    idx = jnp.arange(norb, dtype=jnp.int64)
    bits = (x.astype(jnp.int64) >> idx) & jnp.int64(1)
    return jnp.argmax(bits).astype(jnp.int32)


def two_set_bit_pos_jax(x, norb):
    """Positions ``(pos1, pos2)`` with ``pos1 <= pos2`` of the two lowest set bits.

    JAX-native and vmappable.  When *x* has fewer than two set bits (e.g. only
    one set bit at position *p*), ``pos2`` falls back to 0 (the index returned by
    ``jnp.argmax`` on an all-zeros array).  Note that 0 is a valid bit position,
    so callers must not rely on this fallback to detect the degenerate case.
    This behaviour is intentional: in a vmapped ``jnp.where`` context the result
    of the "wrong branch" is always masked out, so a finite fallback value is
    sufficient and no special-casing is needed.
    """
    idx = jnp.arange(norb, dtype=jnp.int64)
    bits = (x.astype(jnp.int64) >> idx) & jnp.int64(1)
    pos1 = jnp.argmax(bits).astype(jnp.int32)
    pos2 = jnp.argmax(bits.at[pos1].set(jnp.int64(0))).astype(jnp.int32)
    return pos1, pos2


def phase_single_jax(det_int, i, a, norb):
    """Fermionic phase for single excitation ``i -> a``. JAX-native, vmappable.

    Assumes *i* is occupied and *a* is unoccupied in *det_int*.
    Returns ``+1.0`` or ``-1.0`` (float64).
    """
    det = det_int.astype(jnp.int64)
    lo = jnp.minimum(i, a).astype(jnp.int64)
    hi = jnp.maximum(i, a).astype(jnp.int64)
    one = jnp.int64(1)
    # Bits strictly between lo and hi (exclusive on both ends):
    #   hi_mask covers bits 0..hi-1, lo_mask covers bits 0..lo.
    #   Their difference is bits lo+1..hi-1.
    hi_mask = jnp.left_shift(one, hi) - one
    lo_mask = jnp.left_shift(one, lo + one) - one
    mask = hi_mask & ~lo_mask
    n_between = jax_popcount(det & mask, norb)
    return jnp.where(n_between % 2 == 0, 1.0, -1.0)


def phase_double_jax(det_int, i, j, a, b, norb):
    """Fermionic phase for double excitation ``i,j -> a,b``. JAX-native, vmappable.

    Assumes ``i <= j`` (holes) and ``a <= b`` (particles), as returned by
    :func:`two_set_bit_pos_jax`.  When called on an "invalid" pair (e.g. when
    the wrong branch fires inside a ``jnp.where``), the result is ±1 with no
    NaN/Inf, so the caller's outer mask is safe.
    Returns ``+1.0`` or ``-1.0`` (float64).
    """
    det = det_int.astype(jnp.int64)
    i64 = i.astype(jnp.int64)
    j64 = j.astype(jnp.int64)
    a64 = a.astype(jnp.int64)
    b64 = b.astype(jnp.int64)
    one = jnp.int64(1)

    # Step 1: remove lower hole i.
    temp_det = det & ~jnp.left_shift(one, i64)

    # n_ij: electrons strictly between i and j in original det.
    hi_ij = jnp.left_shift(one, j64) - one
    lo_ij = jnp.left_shift(one, i64 + one) - one
    n_ij = jax_popcount(det & (hi_ij & ~lo_ij), norb)

    # n_ja: electrons strictly between j and a in temp_det (order-agnostic).
    lo_ja = jnp.minimum(j64, a64)
    hi_ja = jnp.maximum(j64, a64)
    hi_ja_mask = jnp.left_shift(one, hi_ja) - one
    lo_ja_mask = jnp.left_shift(one, lo_ja + one) - one
    n_ja = jax_popcount(temp_det & (hi_ja_mask & ~lo_ja_mask), norb)

    # Step 2: add lower particle a.
    temp_det2 = temp_det | jnp.left_shift(one, a64)

    # n_ab: electrons strictly between a and b in temp_det2.
    hi_ab = jnp.left_shift(one, b64) - one
    lo_ab = jnp.left_shift(one, a64 + one) - one
    n_ab = jax_popcount(temp_det2 & (hi_ab & ~lo_ab), norb)

    total_count = n_ij + n_ja + n_ab
    return jnp.where(total_count % 2 == 0, 1.0, -1.0)


def precompute_connections(dets_alpha, dets_beta, norb):
    """Collect all off-diagonal connected determinant pairs into index arrays.

    Runs eagerly (outside JIT), using the existing sort/generator logic.
    Returns ``(row_idx, col_idx)`` as JAX ``int32`` arrays of shape
    ``(n_pairs,)`` containing original determinant indices.  Each pair
    satisfies ``1 <= total_excitation_level <= 2`` and appears exactly once.

    The returned arrays can be passed directly to :func:`precompute_h_vals` and
    :func:`scatter_add_matvec` in ``hamiltonian.py``.
    """
    dets_alpha = jnp.asarray(dets_alpha)
    dets_beta = jnp.asarray(dets_beta)
    ndet = len(dets_alpha)
    dummy_coeffs = jnp.ones(ndet)
    row_list: list = []
    col_list: list = []

    # Alpha-major sort: beta singles/doubles + opposite-spin doubles.
    _, s_da, s_db, s_idx_jax = sort_wavefunction_jax(dummy_coeffs, dets_alpha, dets_beta, norb)
    s_idx = s_idx_jax.tolist()
    A_alpha = construct_A(s_da)

    for i, j in find_connected_internal_determinants_beta(s_da, s_db, A_alpha):
        row_list.append(s_idx[i])
        col_list.append(s_idx[j])

    for i, j in find_connected_internal_determinants_oppositespin(s_da, s_db, A_alpha):
        row_list.append(s_idx[i])
        col_list.append(s_idx[j])

    # Beta-major sort: alpha singles/doubles.
    _, s_db2, s_da2, s_idx2_jax = sort_wavefunction_jax(dummy_coeffs, dets_beta, dets_alpha, norb)
    s_idx2 = s_idx2_jax.tolist()
    A_beta = construct_A(s_db2)

    for i, j in find_connected_internal_determinants_beta(s_db2, s_da2, A_beta):
        row_list.append(s_idx2[i])
        col_list.append(s_idx2[j])

    if not row_list:
        return jnp.empty(0, dtype=jnp.int32), jnp.empty(0, dtype=jnp.int32)

    return jnp.array(row_list, dtype=jnp.int32), jnp.array(col_list, dtype=jnp.int32)
