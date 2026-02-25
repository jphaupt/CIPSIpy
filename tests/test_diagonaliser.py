"""
Unit test suite for the (Davidson) diagonaliser
"""

import sys
import os

import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.diagonaliser import Diagonaliser

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
# TODO tests for larger systems, for real systems (see assets files) against FCI
# TODO tests for excited states
