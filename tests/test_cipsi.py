"""
Tests for CIPSI operations (particularly selection)
"""

import os
import sys

# For development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import jax.numpy as jnp
import pytest
import cipsipy.determinants as detops

from cipsipy.cipsi import (
    CIPSISolver,
        apply_epv_and_single_tagging
)

@pytest.fixture
def empty_Bmat():
    """
    returns a true-initialised (2norb x 2norb) matrix for 4 spatorbs
    """
    norb = 4
    return jnp.ones((2*norb, 2*norb), dtype=bool)

def test_Bmat_epv_single(empty_Bmat):
    """
    test case where p can be used for double excitations but not q
    Calculation was done by hand

    False in Bmat represents tagging (i.e. NOT included)
        note this is the reverse of the thesis
    """
    norb = 4
    Bmat_expected = empty_Bmat.copy()
    Gdet = 0b01110011 # (0011, 0111): alpha {1a,2a}, beta {1b,2b,3b}
    ps = 4 # 1b
    qs = 1 # 2a
    Gpq_expected = 0b01100001 # (0001, 0110): alpha {1a}, beta {2b,3b}
    G_pq = detops.annihilate(detops.annihilate(Gdet, ps), qs)
    assert Gpq_expected == G_pq

    # diagonals are Pauli-violating
    for rs in range(2*norb):
        Bmat_expected = Bmat_expected.at[rs,rs].set(False)

    # 1a, 2b, 3b tagged (occupied in G_pq)
    # BOTH 1b (p) and 2a (q) are tagged in the base pass
    #   - p=1b is the lowest occupied beta in G -> untagging allowed, but ONLY
    #     for specific (4,s) cells where s is an alpha orbital (s < norb),
    #     i.e. cell-level untagging, NOT whole-row untagging
    #   - q=2a is NOT the lowest occupied alpha in G (that is 1a) -> stay tagged
    # finally all of {1a, 2a, 1b, 2b, 3b} = {0, 1, 4, 5, 6} are tagged
    for rs in [0, 1, 4, 5, 6]:
        for i in range(2*norb):
            Bmat_expected = Bmat_expected.at[rs,i].set(False)
            Bmat_expected = Bmat_expected.at[i,rs].set(False)

    # p=1b is the lowest occupied beta in G -> untag (4,s) for free alpha s != 2a
    # s must be alpha (s < norb); s=7 (beta) is never a candidate, so (4,7) stays False
    # (4,7) would mean "restore 1b, create 4b" which is a spin-flip single with
    # zero Hamiltonian coupling and the wrong spin sector
    for s in [2, 3]: # 3a, 4a are the free unoccupied alpha orbitals
        Bmat_expected = Bmat_expected.at[4, s].set(True)
        Bmat_expected = Bmat_expected.at[s, 4].set(True)

    Bmat = apply_epv_and_single_tagging(ps, qs, Gdet, G_pq, norb)
    print("Got:")
    print(Bmat.astype(int))
    print("Expected:")
    print(Bmat_expected.astype(int))
    for i in range(2*norb):
        for j in range(2*norb):
            assert Bmat[i,j] == Bmat_expected[i,j], f"Matrices differ at indices ({i},{j})"
