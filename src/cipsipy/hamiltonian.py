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

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from cipsipy.determinants import (
    count_electrons,
    find_connected_internal_determinants_beta,
    find_connected_internal_determinants_oppositespin,
    construct_A,
    sort_wavefunction,
    get_occupied_indices,
    phase_double,
    phase_single,
    excitation_level_jax,
    first_set_bit_pos_jax,
    two_set_bit_pos_jax,
    phase_single_jax,
    phase_double_jax,
    precompute_connections,
)

# dataclass to generate boilerplate setters
# frozen to indicate it is immutable after construction
@dataclass(frozen=True)
class Hamiltonian:
    """Immutable Hamiltonian container for one electronic-structure problem.

    This class stores static integrals and orbital metadata. Numerical kernels
    remain as top-level pure functions for straightforward JAX usage.
    """

    norb: int
    h_core: jnp.ndarray
    eri: jnp.ndarray
    e_nuc: float

    def __post_init__(self):
        if self.norb < 1:
            raise ValueError("norb must be positive")
        if self.h_core.shape != (self.norb, self.norb):
            raise ValueError("h_core must have shape (norb, norb)")
        if self.eri.shape != (self.norb, self.norb, self.norb, self.norb):
            raise ValueError("eri must have shape (norb, norb, norb, norb)")

    def diagonal(self, coeffs, dets_alpha, dets_beta):
        """Return diagonal Hamiltonian elements for the determinant list."""
        return get_hamiltonian_diagonal(
            coeffs,
            dets_alpha,
            dets_beta,
            self.norb,
            self.h_core,
            self.eri,
        )

    def element(self, det_i_alpha, det_i_beta, det_j_alpha, det_j_beta):
        """Return matrix element <D_i|H|D_j>."""
        return hamiltonian_element(
            det_i_alpha,
            det_i_beta,
            det_j_alpha,
            det_j_beta,
            self.norb,
            self.h_core,
            self.eri,
        )

    def matvec(self, coeffs, dets_alpha, dets_beta, diag_h=None):
        """Return Hamiltonian-vector product without building full H."""
        diagonal = diag_h
        if diagonal is None:
            diagonal = self.diagonal(coeffs, dets_alpha, dets_beta)
        return hamiltonian_vector_product(
            coeffs,
            dets_alpha,
            dets_beta,
            diagonal,
            self.norb,
            self.h_core,
            self.eri,
        )

@partial(jax.jit, static_argnums=(3,))
def get_hamiltonian_diagonal(coeffs, dets_alpha, dets_beta, norb, h_core, eri):
    dets_alpha = jnp.asarray(dets_alpha)
    dets_beta = jnp.asarray(dets_beta)
    _diag_batched = jax.vmap(lambda da, db: _diagonal_element(da, db, norb, h_core, eri))
    return _diag_batched(dets_alpha, dets_beta)

def hamiltonian_vector_product(coeffs, dets_alpha, dets_beta, diag_h, norb, h_core, eri):
    """Build a Hamiltonian matrix-vector product

    This function calculates the product H @ dets without explicitly constructing
    the Hamiltonian matrix H

    The wave function is |ψ⟩ = ∑c_i |D_i⟩ᵅ|D_i⟩ᵝ. This is described by three
    parallel arrays:
    - vector of c_i (coeffs)
    - vector of alpha determinants |D_i⟩ᵅ (dets_alpha)
    - vector of beta determinants |D_i⟩ᵝ (dets_beta)

    TODO organise this data into a Wavefunction class -- but be careful with JAX (you need PyTree I think)

    TODO parallelise via JAX -- right now uses some ugly for loops!

    Recall (Hv)_i = ∑_j H_ij c_j

    Args:
        coeffs: coefficients c_i for the wavefunction |Ψ⟩=∑_i c_i|D_α⟩|D_β⟩
        dets_alpha: Sequence of alpha-spin determinants |D_α⟩ (bitstring integers)
        dets_beta: Sequence of beta-spin determinants |D_β⟩ (bitstring integers)
        diag_h: diagonal of the Hamiltonian
        norb: Number of spatial orbitals
        h_core: One-electron integrals [norb, norb]
        eri: Two-electron integrals [norb, norb, norb, norb]
    """
    ndet = len(coeffs)
    v_out = jnp.zeros(ndet) # return value, Hv

    # first do diagonal contribution: H_ii * c_i
    v_out = diag_h * coeffs

    # first sort dets (alpha-major)
    s_c, s_da, s_db, s_idx = sort_wavefunction(coeffs, dets_alpha, dets_beta, norb)
    A_array_alpha = construct_A(s_da)

    # get beta excitations and add to v_out
    for i, j in find_connected_internal_determinants_beta(s_da, s_db, A_array_alpha):
        h_ij = hamiltonian_element(s_da[i], s_db[i], s_da[j], s_db[j], norb, h_core, eri)
        v_out = v_out.at[s_idx[i]].add(h_ij * s_c[j])
        v_out = v_out.at[s_idx[j]].add(h_ij * s_c[i])

    # get opposite-spin excitations and add to v_out
    for i, j in find_connected_internal_determinants_oppositespin(s_da, s_db, A_array_alpha):
        h_ij = hamiltonian_element(s_da[i], s_db[i], s_da[j], s_db[j], norb, h_core, eri)
        v_out = v_out.at[s_idx[i]].add(h_ij * s_c[j])
        v_out = v_out.at[s_idx[j]].add(h_ij * s_c[i])

    # sort beta-major
    s_c, s_db, s_da, s_idx = sort_wavefunction(coeffs, dets_beta, dets_alpha, norb)
    A_array_beta = construct_A(s_db)

    # get alpha excitations and add to v_out
    for i, j in find_connected_internal_determinants_beta(s_db, s_da, A_array_beta):
        # h_ij = hamiltonian_element(s_db[i], s_da[i], s_db[j], s_da[j], norb, h_core, eri)
        h_ij = hamiltonian_element(s_da[i], s_db[i], s_da[j], s_db[j], norb, h_core, eri)
        v_out = v_out.at[s_idx[i]].add(h_ij * s_c[j])
        v_out = v_out.at[s_idx[j]].add(h_ij * s_c[i])
    return v_out

# ===========================================================================
# Bitstring helper functions
# ===========================================================================

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


# ============================================================================
# Matrix element evaluation routines
# ============================================================================

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

    TODO JAX optimisation
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


@partial(jax.jit, static_argnums=(2,))
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
    occ_alpha_f = occ_alpha.astype(eri.dtype)
    occ_beta_f = occ_beta.astype(eri.dtype)

    # One-electron terms
    h_diag = jnp.diag(h_core)
    energy = jnp.dot(occ_alpha_f, h_diag) + jnp.dot(occ_beta_f, h_diag)

    # Alpha-alpha interactions (Coulomb - Exchange)
    # einsum includes i=j terms, but those contribute zero: eri[i,i,i,i] - eri[i,i,i,i] = 0,
    # so this is equivalent to the original sum_{i!=j} loop
    J_aa = jnp.einsum('i,j,iijj->', occ_alpha_f, occ_alpha_f, eri)
    K_aa = jnp.einsum('i,j,ijji->', occ_alpha_f, occ_alpha_f, eri)
    energy += 0.5 * (J_aa - K_aa)

    # Beta-beta interactions (Coulomb - Exchange)
    J_bb = jnp.einsum('i,j,iijj->', occ_beta_f, occ_beta_f, eri)
    K_bb = jnp.einsum('i,j,ijji->', occ_beta_f, occ_beta_f, eri)
    energy += 0.5 * (J_bb - K_bb)

    # Alpha-beta interactions (Coulomb only)
    J_ab = jnp.einsum('i,j,iijj->', occ_alpha_f, occ_beta_f, eri)
    energy += J_ab

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
    # sum_{k != a} occ_same[k] * (eri[i,a,k,k] - eri[i,k,k,a])
    # a is occupied in det_j (the post-excitation determinant) and must be excluded
    # from the sum; we zero out the k=a mask entry rather than branching on k != a
    occ_same = get_occupied_indices(det_j, n_orb)
    occ_same_f = occ_same.astype(h_core.dtype)
    occ_same_f = occ_same_f.at[a].set(0.0)  # exclude k = a
    element += jnp.dot(occ_same_f, jnp.diag(eri[i, a]) - jnp.diag(eri[i, :, :, a]))

    # Opposite spin interactions (Coulomb only)
    occ_opp = get_occupied_indices(spectator_det, n_orb)
    occ_opp_f = occ_opp.astype(h_core.dtype)
    element += jnp.dot(occ_opp_f, jnp.diag(eri[i, a]))

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


# ===========================================================================
# JAX-native (vmappable) matrix element kernels
# ===========================================================================

def _single_excitation_element_spin_jax(det_i, det_j, spectator_det, norb, h_core, eri):
    """Single excitation ``i -> a`` for one spin channel. JAX-native, vmappable.

    Args:
        det_i: pre-excitation determinant of the excited spin.
        det_j: post-excitation determinant of the excited spin.
        spectator_det: determinant of the unchanged spin.
        norb: number of spatial orbitals (compile-time constant).
        h_core: one-electron integrals (norb, norb).
        eri: two-electron integrals (norb, norb, norb, norb).
    """
    i = first_set_bit_pos_jax(det_i & ~det_j, norb)
    a = first_set_bit_pos_jax(det_j & ~det_i, norb)
    phase = phase_single_jax(det_i.astype(jnp.int64), i, a, norb)

    element = h_core[i, a]

    # Coulomb term shared by same-spin and opposite-spin: eri[i, a, k, k]
    # eri[i][a] selects eri[i, :, :, :][a, :, :] = eri[i, a, :, :], shape (norb, norb);
    # jnp.diag extracts the diagonal eri[i, a, k, k], shape (norb,).
    coulomb_row = jnp.diag(eri[i][a])  # eri[i, a, k, k], shape (norb,)

    # Same-spin Coulomb-Exchange: sum_k occ_j[k] * (eri[i,a,k,k] - eri[i,k,k,a])
    # k=a excluded: occ_same uses det_j (a is occupied there) but we zero it out.
    # eri[i][..., a] selects eri[i, :, :, :][..., a] = eri[i, :, :, a], shape (norb, norb);
    # jnp.diag extracts the diagonal eri[i, k, k, a], shape (norb,).
    exchange_row = jnp.diag(eri[i][..., a])  # eri[i, k, k, a], shape (norb,)
    occ_same = get_occupied_indices(det_j, norb).astype(h_core.dtype)
    occ_same = occ_same.at[a].set(0.0)  # exclude k=a
    element += jnp.dot(occ_same, coulomb_row - exchange_row)

    # Opposite-spin Coulomb: sum_k occ_opp[k] * eri[i,a,k,k]
    occ_opp = get_occupied_indices(spectator_det, norb).astype(h_core.dtype)
    element += jnp.dot(occ_opp, coulomb_row)

    return phase * element


def _double_same_spin_element_jax(det_i, det_j, norb, h_core, eri):
    """Double same-spin excitation ``i,j -> a,b``. JAX-native, vmappable."""
    i, j = two_set_bit_pos_jax(det_i & ~det_j, norb)  # holes, i <= j
    a, b = two_set_bit_pos_jax(det_j & ~det_i, norb)  # particles, a <= b
    phase = phase_double_jax(det_i.astype(jnp.int64), i, j, a, b, norb)
    element = eri[i][j][a, b] - eri[i][j][b, a]
    return phase * element


def _double_opposite_spin_element_jax(da_i, db_i, da_j, db_j, norb, h_core, eri):
    """Double opposite-spin excitation ``(i_α->a_α) + (i_β->a_β)``. JAX-native, vmappable."""
    i_alpha = first_set_bit_pos_jax(da_i & ~da_j, norb)
    a_alpha = first_set_bit_pos_jax(da_j & ~da_i, norb)
    i_beta = first_set_bit_pos_jax(db_i & ~db_j, norb)
    a_beta = first_set_bit_pos_jax(db_j & ~db_i, norb)
    phase_a = phase_single_jax(da_i.astype(jnp.int64), i_alpha, a_alpha, norb)
    phase_b = phase_single_jax(db_i.astype(jnp.int64), i_beta, a_beta, norb)
    element = eri[i_alpha][a_alpha][i_beta, a_beta]
    return phase_a * phase_b * element


def hamiltonian_element_batch(da_i, db_i, da_j, db_j, norb, h_core, eri):
    """Matrix element ``<D_i|H|D_j>`` for off-diagonal pairs. JAX-native, vmappable.

    Assumes ``total_excitation_level >= 1``.  All five excitation sub-types
    (single-α, single-β, double-αα, double-ββ, double-αβ) are computed and the
    correct value is selected via ``jnp.where`` with no Python branching, so
    this function can be safely passed to ``jax.vmap``.

    When an "irrelevant" branch fires (wrong excitation type), it produces a
    finite ±value that is discarded by the outer ``jnp.where``, so no
    NaN/Inf propagates.
    """
    exc_alpha = excitation_level_jax(da_i, da_j, norb)
    exc_beta = excitation_level_jax(db_i, db_j, norb)
    total_exc = exc_alpha + exc_beta

    val_sing_a = _single_excitation_element_spin_jax(da_i, da_j, db_i, norb, h_core, eri)
    val_sing_b = _single_excitation_element_spin_jax(db_i, db_j, da_i, norb, h_core, eri)
    val_dbl_aa = _double_same_spin_element_jax(da_i, da_j, norb, h_core, eri)
    val_dbl_bb = _double_same_spin_element_jax(db_i, db_j, norb, h_core, eri)
    val_dbl_ab = _double_opposite_spin_element_jax(da_i, db_i, da_j, db_j, norb, h_core, eri)

    return jnp.where(
        total_exc == 1,
        jnp.where(exc_alpha == 1, val_sing_a, val_sing_b),
        jnp.where(
            exc_alpha == 2, val_dbl_aa,
            jnp.where(exc_alpha == 0, val_dbl_bb, val_dbl_ab),
        ),
    )


@partial(jax.jit, static_argnums=(4,))
def precompute_h_vals(dets_alpha, dets_beta, row_idx, col_idx, norb, h_core, eri):
    """Compute ``H_ij`` for all precomputed connected pairs in one batched call.

    Returns a ``(n_pairs,)`` float64 array.  Call this **once** before the
    Davidson loop; the result is fixed as long as the determinant list and
    integrals do not change.
    """
    dets_alpha = jnp.asarray(dets_alpha)
    dets_beta = jnp.asarray(dets_beta)
    da_i = dets_alpha[row_idx]
    db_i = dets_beta[row_idx]
    da_j = dets_alpha[col_idx]
    db_j = dets_beta[col_idx]
    return jax.vmap(
        hamiltonian_element_batch,
        in_axes=(0, 0, 0, 0, None, None, None),
    )(da_i, db_i, da_j, db_j, norb, h_core, eri)


@jax.jit
def scatter_add_matvec(coeffs, diag_h, h_vals, row_idx, col_idx):
    """Apply ``H`` to a vector using precomputed pair values.

    ``(H v)_i = diag_h[i] * v[i]``
    ``        + sum_{k: row[k]=i} h_vals[k] * v[col[k]]``
    ``        + sum_{k: col[k]=i} h_vals[k] * v[row[k]]``

    This is a single batched scatter-add; no Python loops are involved.
    JIT-compiled for repeated use inside the Davidson inner loop.
    """
    ndet = coeffs.shape[0]
    v_off = jnp.zeros(ndet, dtype=coeffs.dtype)
    v_off = v_off.at[row_idx].add(h_vals * coeffs[col_idx])
    v_off = v_off.at[col_idx].add(h_vals * coeffs[row_idx])
    return diag_h * coeffs + v_off
