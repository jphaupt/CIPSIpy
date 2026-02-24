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

import jax.numpy as jnp

# ============================================================================
# Helper functions for bitwise operations
# ============================================================================

def get_excitation_level(det_i, det_j):
    """
    get excitation level between bitstring-represented determinants det_i and det_j
    i.e. 1/2 ||I⊕J||
    """
    xor_det = det_i ^ det_j
    return count_electrons(xor_det) // 2


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

def find_connected_internal_determinants_beta(dets_alpha, dets_beta):
    """
    algorithm 9 of the thesis - finds all single- and double-connected determinants
    in the list dets. This gives all beta single- and double-excitations (to get
    alpha, just swap the spins)

    PRECONDITION: the arrays are sorted in alpha-major order

    TODO consider a generator/on-the-fly version of this algorithm
    """
    connected_indices = []
    A_indices = construct_A(dets_alpha)
    # iterate over unique alpha blocks (A stores block starts and a sentinel)
    for a in range(len(A_indices) - 1):
        # all determinants sharing alpha part are in the range A[a], A[a+1)-1
        for b1 in range(A_indices[a], A_indices[a + 1]):
            for b2 in range(b1 + 1, A_indices[a + 1]):
                if get_excitation_level(dets_beta[b1], dets_beta[b2]) <= 2:
                    connected_indices.append((b1, b2))
    return connected_indices

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
    if i < 0 or not dets:
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

def sort_determinants(dets_alpha, dets_beta, norb, sort_alg=radix_sort_rec):
    """
    sorts determinants dets_alpha, dets_beta in alpha-major order, given a
    number of spatial orbitals norb, using the sorting algorithm
    """
    # pass dets_alpha as keys so it follows dets_beta's movement
    beta_sorted, alpha_carried = sort_alg(dets_beta, norb - 1, dets_alpha)

    # then primary key
    final_alpha, final_beta = sort_alg(alpha_carried, norb - 1, beta_sorted)

    return final_alpha, final_beta


def sort_determinants_jax(dets_alpha, dets_beta, norb):
    """
    Sorts determinants in alpha-major order using JAX built-ins.
    lexsort( (secondary_key, primary_key) )
    """
    # lexsort sorts by the last array in the tuple first.
    # To get alpha-major (alpha first, then beta), we pass (beta, alpha).
    idx = jnp.lexsort((dets_beta, dets_alpha))

    return dets_alpha[idx], dets_beta[idx]


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
