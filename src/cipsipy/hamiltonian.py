"""
Hamiltonian matrix elements using Slater-Condon rules.

Computes <Det_i|H|Det_j> between Slater determinants using Slater-Condon rules.

Determinants: (det_alpha, det_beta) tuples where each is an integer bitstring
representing occupied spatial orbitals for that spin.

Two-electron integrals: Chemist's notation (pq|rs) = eri[p,q,r,s]

Slater-Condon Rules:
- 0 excitations: Diagonal energy
- 1 excitation i→a: h[i,a] + Σ_k [(ia|kk) - (ik|ka)]
- 2 excitations i,j→a,b: (ij|ab)
- 3+ excitations: 0 (orthogonal)
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
    Determine excitation level between two determinants.

    Args:
        det_i: First determinant (integer)
        det_j: Second determinant (integer)

    Returns:
        Number of orbital differences (0, 1, 2, ...)
    """
    diff = det_i ^ det_j
    return count_electrons(diff) // 2


def get_excitation_operators(det_i, det_j):
    """
    Get hole and particle orbital indices for excitation from det_i to det_j.

    Args:
        det_i: Initial determinant (integer)
        det_j: Final determinant (integer)

    Returns:
        (holes, particles) where:
            holes: List of orbitals in det_i but not det_j
            particles: List of orbitals in det_j but not det_i
    """
    holes_bits = det_i & ~det_j
    particles_bits = det_j & ~det_i

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


def hamiltonian_element(det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri):
    """
    Calculate Hamiltonian matrix element between spin-separated determinants.

    Args:
        det_i_alpha, det_i_beta: Initial determinants (α, β spins)
        det_j_alpha, det_j_beta: Final determinants (α, β spins)
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb] (chemist's notation)

    Returns:
        Hamiltonian matrix element (float)
    """
    exc_alpha = excitation_level(det_i_alpha, det_j_alpha)
    exc_beta = excitation_level(det_i_beta, det_j_beta)
    total_exc = exc_alpha + exc_beta

    if total_exc == 0:
        return _diagonal_element(det_i_alpha, det_i_beta, n_orb, h_core, eri)
    elif total_exc == 1:
        if exc_alpha == 1:
            return _single_excitation_element(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="alpha"
            )
        else:
            return _single_excitation_element(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="beta"
            )
    elif total_exc == 2:
        if exc_alpha == 2:
            return _double_same_spin_element(
                det_i_alpha, det_j_alpha, n_orb, h_core, eri, spin="alpha"
            )
        elif exc_beta == 2:
            return _double_same_spin_element(
                det_i_beta, det_j_beta, n_orb, h_core, eri, spin="beta"
            )
        else:
            return _double_opposite_spin_element(
                det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
            )
    else:
        return 0.0


def _diagonal_element(det_alpha, det_beta, n_orb, h_core, eri):
    """
    Diagonal element for spin-separated determinants.

    Formula:
        E = Σ_i h[i,i] + 1/2 Σ_{i≠j,same_spin} [(ii|jj) - (ij|ji)]
            + Σ_{i_α,j_β} (ii|jj)

    Chemist's notation: (pq|rs) = eri[p,q,r,s]
    """
    occ_alpha = get_occupied_indices(det_alpha, n_orb)
    occ_beta = get_occupied_indices(det_beta, n_orb)
    alpha_indices = jnp.where(occ_alpha)[0]
    beta_indices = jnp.where(occ_beta)[0]

    energy = 0.0
    energy += jnp.sum(h_core[alpha_indices, alpha_indices])
    energy += jnp.sum(h_core[beta_indices, beta_indices])

    # Alpha-alpha interactions (Coulomb - Exchange)
    for i_idx in range(len(alpha_indices)):
        i = alpha_indices[i_idx]
        for j_idx in range(len(alpha_indices)):
            j = alpha_indices[j_idx]
            if i != j:
                energy += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])

    # Beta-beta interactions (Coulomb - Exchange)
    for i_idx in range(len(beta_indices)):
        i = beta_indices[i_idx]
        for j_idx in range(len(beta_indices)):
            j = beta_indices[j_idx]
            if i != j:
                energy += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])

    # Alpha-beta interactions (Coulomb only)
    for i in alpha_indices:
        for j in beta_indices:
            energy += eri[i, i, j, j]

    return energy


def _single_excitation_element(
    det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri, spin="alpha"
):
    """
    Single excitation element for spin-separated determinants.

    Formula for i→a excitation:
        H = h[i,a] + Σ_{k,same_spin} [(ia|kk) - (ik|ka)]
            + Σ_{k,opposite_spin} (ia|kk)
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

    phase = phase_single(det_i, i, a)
    if phase == 0:
        return 0.0

    element = h_core[i, a]

    # Same spin interactions (Coulomb - Exchange)
    occ_same = get_occupied_indices(det_j, n_orb)
    occ_same_indices = jnp.where(occ_same)[0]
    for k in occ_same_indices:
        if k != a:
            element += eri[i, a, k, k] - eri[i, k, k, a]

    # Opposite spin interactions (Coulomb only)
    occ_opp = get_occupied_indices(spectator_det, n_orb)
    opp_indices = jnp.where(occ_opp)[0]
    for k in opp_indices:
        element += eri[i, a, k, k]

    return phase * element


def _double_same_spin_element(det_i, det_j, n_orb, h_core, eri, spin="alpha"):
    """
    Double excitation within the same spin.

    Formula for i,j→a,b (same spin):
        H = (ij|ab) - (ij|ba) in chemist's notation
    """
    holes, particles = get_excitation_operators(det_i, det_j)
    if len(holes) != 2 or len(particles) != 2:
        return 0.0

    i, j = holes[0], holes[1]
    a, b = particles[0], particles[1]

    phase = phase_double(det_i, i, j, a, b)
    if phase == 0:
        return 0.0

    element = eri[i, j, a, b] - eri[i, j, b, a]

    return phase * element


def _double_opposite_spin_element(
    det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
):
    """
    Double excitation with one alpha and one beta excitation.

    Formula for i_α→a_α and i_β→a_β (opposite spin):
        H = (i_α a_α | i_β a_β) in chemist's notation
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

    phase_alpha = phase_single(det_i_alpha, i_alpha, a_alpha)
    phase_beta = phase_single(det_i_beta, i_beta, a_beta)
    if phase_alpha == 0 or phase_beta == 0:
        return 0.0

    element = eri[i_alpha, a_alpha, i_beta, a_beta]

    return phase_alpha * phase_beta * element
