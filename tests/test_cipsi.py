"""
Tests for CIPSI operations (particularly selection)
"""

import os
import sys

import jax.numpy as jnp
import pytest

# For development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.cipsi import (
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
    Gdet = 0b01110011 # (0011, 0111)
    ps = 4 # 1b
    qs = 1 # 1a
    Gpq = 0b01100001 # (0001, 0110)

    # diagonals are Pauli-violating
    for rs in range(2*norb):
        Bmat_expected = Bmat_expected.at[rs,rs].set(False)

    # rs = 1a, 2b, 3b tagged (occupied)
    # rs = 1b, 2a tagged (p,q)
    # q=2a is alpha and p=1b is lowest beta in G-> untag p=1b to allow singles
    # p=1b is beta, but q=2a is NOT lowest alpha in G -> keep q tagged
    # finally: 1a, 2b, 3b, 2a get tagged
    # these indices are rs = 0, 5, 6, 1
    for rs in [0, 1, 5, 6]:
        for i in range(2*norb):
            Bmat_expected = Bmat_expected.at[rs,i].set(False)
            Bmat_expected = Bmat_expected.at[i,rs].set(False)

    Bmat = apply_epv_and_single_tagging(empty_Bmat, ps, qs, Gdet, norb)
    for i in range(2*norb):
        for j in range(2*norb):
            assert Bmat[i,j] == Bmat_expected[i,j], f"Matrices differ at indices ({i},{j})"
