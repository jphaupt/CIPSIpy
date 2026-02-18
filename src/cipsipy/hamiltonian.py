"""
Hamiltonian matrix elements using Slater-Condon rules.

This module computes matrix elements <Det_i|H|Det_j> between Slater determinants
using the Slater-Condon rules, which relate the matrix element to the number of
orbital differences between the determinants.

Determinants are represented as a tuple (det_alpha, det_beta) where each is an
integer bitstring representing occupied spatial orbitals for that spin.

Slater-Condon Rules:
- 0 excitations (same determinant): Full diagonal energy
- 1 excitation i→a: h[i,a] + Σ_k [(ia|kk) - (ik|ka)]
- 2 excitations i,j→a,b: (ij|ab)
- 3+ excitations: 0 (by orthogonality)
"""

import jax.numpy as jnp

from cipsipy.determinants import (
    count_electrons,
    get_occupied_indices,
    phase_double,
    phase_single,
)


def excitation_level(det_i, det_j):
    """
    Determine the excitation level between two determinants.

    Args:
        det_i: First determinant (integer)
        det_j: Second determinant (integer)

    Returns:
        Number of orbital differences (0, 1, 2, 3, ...)

    The excitation level is half the number of different bits between
    the two determinants.
    """
    # XOR gives bits that differ
    diff = det_i ^ det_j
    # Count differing bits and divide by 2 (since each excitation changes 2 bits)
    return count_electrons(diff) // 2


def get_excitation_operators(det_i, det_j):
    """
    Get hole and particle orbital indices for excitation from det_i to det_j.

    Args:
        det_i: Initial determinant (integer)
        det_j: Final determinant (integer)

    Returns:
        Tuple (holes, particles) where:
            - holes: List of orbital indices in det_i but not det_j
            - particles: List of orbital indices in det_j but not det_i
    """
    # Find orbitals unique to each determinant
    holes_bits = det_i & ~det_j  # In i but not in j
    particles_bits = det_j & ~det_i  # In j but not in i

    # Convert to lists of indices
    holes = []
    particles = []

    idx = 0
    while holes_bits:
        if holes_bits & 1:
            holes.append(idx)
        holes_bits >>= 1
        idx += 1

    idx = 0
    while particles_bits:
        if particles_bits & 1:
            particles.append(idx)
        particles_bits >>= 1
        idx += 1

    return holes, particles


def hamiltonian_diagonal(det, n_orb, h_core, eri):
    """
    Calculate diagonal Hamiltonian element <Det|H|Det>.

    Args:
        det: Determinant (integer)
        n_orb: Number of orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Diagonal matrix element (float)

    Formula:
        E = Σ_i h[i,i] + 1/2 Σ_{i,j} [(ii|jj) - (ij|ji)]
    """
    occ_mask = get_occupied_indices(det, n_orb)
    occ_indices = jnp.where(occ_mask)[0]

    if len(occ_indices) == 0:
        return 0.0

    # One-electron contribution
    one_electron = jnp.sum(h_core[occ_indices, occ_indices])

    # Two-electron contribution
    two_electron = 0.0
    for i_idx in range(len(occ_indices)):
        i = occ_indices[i_idx]
        for j_idx in range(len(occ_indices)):
            j = occ_indices[j_idx]
            if i != j:
                # Coulomb - Exchange
                two_electron += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])

    return one_electron + two_electron


def hamiltonian_single(det_i, det_j, n_orb, h_core, eri):
    """
    Calculate single excitation Hamiltonian element <Det_i|H|Det_j>.

    Args:
        det_i: Initial determinant (integer)
        det_j: Final determinant (integer)
        n_orb: Number of orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Single excitation matrix element (float)

    Formula for excitation i→a:
        H = h[i,a] + Σ_k [(ia|kk) - (ik|ka)]
        where k runs over all occupied orbitals in both determinants
    """
    holes, particles = get_excitation_operators(det_i, det_j)

    if len(holes) != 1 or len(particles) != 1:
        return 0.0

    i = holes[0]
    a = particles[0]

    # Calculate phase
    phase = phase_single(det_i, i, a)
    if phase == 0:
        return 0.0

    # One-electron term
    element = h_core[i, a]

    # Two-electron term: sum over orbitals occupied in both determinants
    occ_i = get_occupied_indices(det_i, n_orb)
    occ_j = get_occupied_indices(det_j, n_orb)
    # Orbitals occupied in both (common orbitals)
    common_occ = occ_i & occ_j
    common_indices = jnp.where(common_occ)[0]

    for k in common_indices:
        # Coulomb - Exchange
        element += eri[i, a, k, k] - eri[i, k, k, a]

    return phase * element


def hamiltonian_double(det_i, det_j, n_orb, h_core, eri):
    """
    Calculate double excitation Hamiltonian element <Det_i|H|Det_j>.

    Args:
        det_i: Initial determinant (integer)
        det_j: Final determinant (integer)
        n_orb: Number of orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Double excitation matrix element (float)

    Formula for excitation i,j→a,b:
        H = (ij|ab)
    """
    holes, particles = get_excitation_operators(det_i, det_j)

    if len(holes) != 2 or len(particles) != 2:
        return 0.0

    i, j = holes[0], holes[1]
    a, b = particles[0], particles[1]

    # Calculate phase
    phase = phase_double(det_i, i, j, a, b)
    if phase == 0:
        return 0.0

    # Two-electron integral (ij|ab)
    element = eri[i, j, a, b]

    return phase * element


def hamiltonian_element(det_i, det_j, n_orb, h_core, eri):
    """
    Calculate Hamiltonian matrix element <Det_i|H|Det_j> using Slater-Condon rules.

    Args:
        det_i: First determinant (integer)
        det_j: Second determinant (integer)
        n_orb: Number of orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Hamiltonian matrix element (float)

    Uses Slater-Condon rules based on excitation level:
        - 0 excitations: Diagonal element
        - 1 excitation: Single excitation element
        - 2 excitations: Double excitation element
        - 3+ excitations: 0 (orthogonal)
    """
    exc_level = excitation_level(det_i, det_j)

    if exc_level == 0:
        return hamiltonian_diagonal(det_i, n_orb, h_core, eri)
    elif exc_level == 1:
        return hamiltonian_single(det_i, det_j, n_orb, h_core, eri)
    elif exc_level == 2:
        return hamiltonian_double(det_i, det_j, n_orb, h_core, eri)
    else:
        # Higher excitations are orthogonal
        return 0.0


# ============================================================================
# Spin-separated determinant functions (alpha and beta spins handled separately)
# ============================================================================


def hamiltonian_element_spin(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri):
    """
    Calculate Hamiltonian matrix element between spin-separated determinants.

    This is the proper formulation where alpha and beta spins are represented
    by separate bitstrings.

    Args:
        det_i_alpha: Alpha spin determinant for state i (integer)
        det_i_beta: Beta spin determinant for state i (integer)
        det_j_alpha: Alpha spin determinant for state j (integer)
        det_j_beta: Beta spin determinant for state j (integer)
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb] in physicist notation (ij|kl)

    Returns:
        Hamiltonian matrix element (float)

    The total excitation level is the sum of alpha and beta excitation levels.
    """
    # Get excitation levels for each spin
    exc_alpha = excitation_level(det_i_alpha, det_j_alpha)
    exc_beta = excitation_level(det_i_beta, det_j_beta)
    total_exc = exc_alpha + exc_beta

    # Slater-Condon rules based on total excitation level
    if total_exc == 0:
        # Same determinant - diagonal element
        return _hamiltonian_diagonal_spin(det_i_alpha, det_i_beta, n_orb, h_core, eri)
    elif total_exc == 1:
        # Single excitation (alpha or beta)
        if exc_alpha == 1:
            return _hamiltonian_single_spin(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="alpha"
            )
        else:
            return _hamiltonian_single_spin(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="beta"
            )
    elif total_exc == 2:
        # Double excitation (alpha-alpha, beta-beta, or alpha-beta)
        if exc_alpha == 2:
            return _hamiltonian_double_same_spin(
                det_i_alpha, det_j_alpha, n_orb, h_core, eri, spin="alpha"
            )
        elif exc_beta == 2:
            return _hamiltonian_double_same_spin(
                det_i_beta, det_j_beta, n_orb, h_core, eri, spin="beta"
            )
        else:
            # One alpha excitation and one beta excitation
            return _hamiltonian_double_opposite_spin(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
            )
    else:
        # Higher excitations are orthogonal
        return 0.0


def _hamiltonian_diagonal_spin(det_alpha, det_beta, n_orb, h_core, eri):
    """
    Calculate diagonal element for spin-separated determinants.

    Args:
        det_alpha: Alpha spin determinant (integer)
        det_beta: Beta spin determinant (integer)
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]

    Returns:
        Diagonal energy (float)

    Formula:
        E = Σ_i h[i,i] + 1/2 Σ_{i,j} [(ii|jj) - δ_σiσj (ij|ji)]
        where σ indicates spin
    """
    occ_alpha = get_occupied_indices(det_alpha, n_orb)
    occ_beta = get_occupied_indices(det_beta, n_orb)
    alpha_indices = jnp.where(occ_alpha)[0]
    beta_indices = jnp.where(occ_beta)[0]

    # One-electron contribution from both spins
    energy = 0.0
    energy += jnp.sum(h_core[alpha_indices, alpha_indices])
    energy += jnp.sum(h_core[beta_indices, beta_indices])

    # Two-electron contribution
    # Alpha-alpha interactions
    for i_idx in range(len(alpha_indices)):
        i = alpha_indices[i_idx]
        for j_idx in range(len(alpha_indices)):
            j = alpha_indices[j_idx]
            if i != j:
                energy += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])

    # Beta-beta interactions
    for i_idx in range(len(beta_indices)):
        i = beta_indices[i_idx]
        for j_idx in range(len(beta_indices)):
            j = beta_indices[j_idx]
            if i != j:
                energy += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])

    # Alpha-beta interactions (only Coulomb, no exchange)
    for i in alpha_indices:
        for j in beta_indices:
            energy += eri[i, i, j, j]

    return energy


def _hamiltonian_single_spin(
    det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="alpha"
):
    """
    Calculate single excitation element for spin-separated determinants.

    Args:
        det_i_alpha, det_i_beta: Initial determinants
        det_j_alpha, det_j_beta: Final determinants
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals
        eri: Two-electron integrals
        spin: 'alpha' or 'beta' indicating which spin is excited

    Returns:
        Single excitation matrix element (float)
    """
    if spin == "alpha":
        det_i = det_i_alpha
        det_j = det_j_alpha
        spectator_det = det_i_beta
    else:
        det_i = det_i_beta
        det_j = det_j_beta
        spectator_det = det_i_alpha

    holes, particles = get_excitation_operators(det_i, det_j)
    if len(holes) != 1 or len(particles) != 1:
        return 0.0

    i = holes[0]
    a = particles[0]

    # Calculate phase
    phase = phase_single(det_i, i, a)
    if phase == 0:
        return 0.0

    # One-electron term
    element = h_core[i, a]

    # Two-electron terms with same spin electrons
    occ_same = get_occupied_indices(det_j if spin == "alpha" else det_j, n_orb)
    occ_same_indices = jnp.where(occ_same)[0]
    for k in occ_same_indices:
        if k != a:  # k is in the final state of the excited spin
            element += eri[i, a, k, k] - eri[i, k, k, a]

    # Two-electron terms with opposite spin electrons (only Coulomb)
    occ_opp = get_occupied_indices(spectator_det, n_orb)
    opp_indices = jnp.where(occ_opp)[0]
    for k in opp_indices:
        element += eri[i, a, k, k]

    return phase * element


def _hamiltonian_double_same_spin(det_i, det_j, n_orb, h_core, eri, spin="alpha"):
    """
    Calculate double excitation element within the same spin.

    Args:
        det_i: Initial determinant for this spin (integer)
        det_j: Final determinant for this spin (integer)
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals
        eri: Two-electron integrals
        spin: 'alpha' or 'beta' (for phase tracking)

    Returns:
        Double excitation matrix element (float)
    """
    holes, particles = get_excitation_operators(det_i, det_j)
    if len(holes) != 2 or len(particles) != 2:
        return 0.0

    i, j = holes[0], holes[1]
    a, b = particles[0], particles[1]

    # Calculate phase
    phase = phase_double(det_i, i, j, a, b)
    if phase == 0:
        return 0.0

    # Two-electron integral with exchange
    element = eri[i, j, a, b] - eri[i, j, b, a]

    return phase * element


def _hamiltonian_double_opposite_spin(
    det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
):
    """
    Calculate double excitation with one alpha and one beta excitation.

    Args:
        det_i_alpha, det_i_beta: Initial determinants
        det_j_alpha, det_j_beta: Final determinants
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals
        eri: Two-electron integrals

    Returns:
        Double excitation matrix element (float)
    """
    holes_alpha, particles_alpha = get_excitation_operators(det_i_alpha, det_j_alpha)
    holes_beta, particles_beta = get_excitation_operators(det_i_beta, det_j_beta)

    if len(holes_alpha) != 1 or len(particles_alpha) != 1:
        return 0.0
    if len(holes_beta) != 1 or len(particles_beta) != 1:
        return 0.0

    i_alpha = holes_alpha[0]
    a_alpha = particles_alpha[0]
    i_beta = holes_beta[0]
    a_beta = particles_beta[0]

    # Calculate phases for each spin
    phase_alpha = phase_single(det_i_alpha, i_alpha, a_alpha)
    phase_beta = phase_single(det_i_beta, i_beta, a_beta)
    if phase_alpha == 0 or phase_beta == 0:
        return 0.0

    # Two-electron integral (only Coulomb, no exchange between different spins)
    element = eri[i_alpha, i_beta, a_alpha, a_beta]

    return phase_alpha * phase_beta * element
