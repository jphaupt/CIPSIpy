"""
Unit test suite for the (Davidson) diagonaliser
"""

import sys
import os

import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.diagonaliser import Diagonaliser
from cipsipy.hamiltonian import get_hamiltonian_diagonal, hamiltonian_element, hamiltonian_vector_product

def _lowest_n_eigs(H, n):
    """helper function"""
    return jnp.linalg.eigvalsh(H)[:n]

def test_davidson_ground_state():
    # 1. Setup a simple 3x3 diagonal dominant matrix
    # H = [[1.0, 0.1, 0.0], [0.1, 2.0, 0.1], [0.0, 0.1, 3.0]]
    h_diag = jnp.array([1.0, 2.0, 3.0])


    # 2. Initialize diagonaliser
    solver = Diagonaliser(H_diag=h_diag, nstate=1, residual_tol=1e-8)

    # Explicit matrix product for testing
    H = jnp.array([[1.0, 0.1, 0.0],
                   [0.1, 2.0, 0.1],
                   [0.0, 0.1, 3.0]])

    mock_hvp = lambda c: jnp.dot(H,c)

    # 3. Solve
    energy, wavefunction = solver.davidson(mock_hvp)

    # 4. Compare with exact linalg
    expected_energy = jnp.linalg.eigvalsh(H)[0]

    assert jnp.allclose(energy, expected_energy, atol=1e-10)

def test_davidson_two_lowest_states():
    h_diag = jnp.array([1.0, 2.0, 3.0, 4.0])
    H = jnp.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.2, 2.0, 0.2, 0.0],
            [0.0, 0.2, 3.0, 0.2],
            [0.0, 0.0, 0.2, 4.0],
        ]
    )

    solver = Diagonaliser(H_diag=h_diag, nstate=2, residual_tol=1e-8)
    mock_hvp = lambda c: jnp.dot(H, c)

    energies, _ = solver.davidson(mock_hvp)
    energies = jnp.sort(jnp.ravel(energies))[:2]
    expected = _lowest_n_eigs(H, 2)

    assert energies.shape == (2,)
    assert jnp.allclose(energies, expected, atol=1e-8)


def test_davidson_accepts_vector_only_hvp():
    """Davidson should work when H_vec_prod only accepts 1D vectors."""
    h_diag = jnp.array([1.0, 2.0, 3.0])
    H = jnp.array([
        [1.0, 0.1, 0.0],
        [0.1, 2.0, 0.1],
        [0.0, 0.1, 3.0],
    ])

    solver = Diagonaliser(H_diag=h_diag, nstate=1, residual_tol=1e-8)

    def vector_only_hvp(v):
        assert v.ndim == 1
        return H @ v

    energies, _ = solver.davidson(vector_only_hvp)
    expected = jnp.linalg.eigvalsh(H)[0]

    assert jnp.allclose(energies[0], expected, atol=1e-10)


def test_davidson_with_sparse_qc_matvec():
    """Use chemistry sparse matvec closure directly in Davidson."""
    norb = 2
    da = [0b01, 0b10]
    db = [0b01, 0b10]
    ndet = len(da)

    h_core = jnp.array([[1.0, 0.2], [0.2, 1.5]])
    eri = jnp.zeros((norb, norb, norb, norb))
    eri = eri.at[0, 0, 0, 0].set(0.6)
    eri = eri.at[1, 1, 1, 1].set(0.4)
    eri = eri.at[0, 0, 1, 1].set(0.2)
    eri = eri.at[1, 1, 0, 0].set(0.2)

    # Build dense reference Hamiltonian
    H = jnp.zeros((ndet, ndet))
    for i in range(ndet):
        for j in range(ndet):
            H = H.at[i, j].set(hamiltonian_element(da[i], db[i], da[j], db[j], norb, h_core, eri))

    h_diag = get_hamiltonian_diagonal(jnp.ones(ndet), da, db, norb, h_core, eri)

    def sparse_hvp(v):
        return hamiltonian_vector_product(v, da, db, h_diag, norb, h_core, eri)

    solver = Diagonaliser(H_diag=h_diag, nstate=1, residual_tol=1e-8)
    energies, _ = solver.davidson(sparse_hvp)

    expected = jnp.linalg.eigvalsh(H)[0]
    assert jnp.allclose(energies[0], expected, atol=1e-8)

# TODO tests for larger systems, for real systems (see assets files) against FCI
# TODO tests for excited states
