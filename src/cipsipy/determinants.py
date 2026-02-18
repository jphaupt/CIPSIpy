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
            det |= 1 << idx
    return det


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
    if not (det_int & (1 << orbital_idx)):
        return 0  # Orbital not occupied - invalid operation

    # Remove electron
    return det_int & ~(1 << orbital_idx)


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
    if det_int & (1 << orbital_idx):
        return 0  # Orbital already occupied - invalid operation

    # Add electron
    return det_int | (1 << orbital_idx)


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
    if not (det_int & (1 << i)):
        return 0
    if det_int & (1 << a):
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

    # Apply the excitation: remove from i, add to a
    new_det = det_int & ~(1 << i)  # Remove electron from i
    new_det = new_det | (1 << a)  # Add electron to a

    return new_det, phase


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
    if not (det_int & (1 << i)) or not (det_int & (1 << j)):
        return 0  # Holes must be occupied
    if (det_int & (1 << a)) or (det_int & (1 << b)):
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
    temp_det = det_int & ~(1 << i)

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
    temp_det = temp_det | (1 << a)

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

    # Apply the excitation: remove from i and j, add to a and b
    new_det = det_int & ~(1 << i)  # Remove electron from i
    new_det = new_det & ~(1 << j)  # Remove electron from j
    new_det = new_det | (1 << a)  # Add electron to a
    new_det = new_det | (1 << b)  # Add electron to b

    return new_det, phase
