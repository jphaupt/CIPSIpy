#!/usr/bin/env python3
"""
Demonstration of matrix element evaluation functions.

This script demonstrates the usage of determinant operations and
Hamiltonian matrix element calculations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import jax.numpy as jnp
from cipsipy.determinants import (
    get_occupied_indices,
    count_electrons,
    create_determinant,
    apply_single_excitation,
    apply_double_excitation,
)
from cipsipy.hamiltonian import hamiltonian_element


def main():
    print("=" * 70)
    print("Matrix Element Evaluation Functions Demo")
    print("=" * 70)

    # Example 1: Bitstring operations
    print("\n1. Determinant Bitstring Operations")
    print("-" * 70)
    det = 3  # Binary 0b0011 - orbitals 0 and 1 occupied
    n_orb = 4
    occ = get_occupied_indices(det, n_orb)
    n_elec = count_electrons(det)
    print(f"Determinant: {det} (binary: 0b{det:04b})")
    print(f"Occupation mask: {occ}")
    print(f"Number of electrons: {n_elec}")

    # Example 2: Creating determinants
    print("\n2. Creating Determinants from Occupation Lists")
    print("-" * 70)
    occ_indices = [0, 2, 3]  # Occupy orbitals 0, 2, and 3
    det = create_determinant(occ_indices, n_orbitals=4)
    print(f"Occupation indices: {occ_indices}")
    print(f"Created determinant: {det} (binary: 0b{det:04b})")

    # Example 3: Single excitations
    print("\n3. Single Excitation Operations")
    print("-" * 70)
    det_i = 3  # 0b0011 (orbitals 0, 1)
    print(f"Initial determinant: {det_i} (binary: 0b{det_i:04b})")
    det_j, phase = apply_single_excitation(det_i, i=0, a=2)
    print(f"After excitation 0→2: {det_j} (binary: 0b{det_j:04b})")
    print(f"Phase factor: {phase}")

    # Example 4: Double excitations
    print("\n4. Double Excitation Operations")
    print("-" * 70)
    det_i = 3  # 0b0011 (orbitals 0, 1)
    print(f"Initial determinant: {det_i} (binary: 0b{det_i:04b})")
    det_j, phase = apply_double_excitation(det_i, i=0, j=1, a=2, b=3)
    print(f"After excitation 0,1→2,3: {det_j} (binary: 0b{det_j:04b})")
    print(f"Phase factor: {phase}")

    # Example 5: Hamiltonian matrix elements
    print("\n5. Hamiltonian Matrix Elements (Simple H2-like System)")
    print("-" * 70)

    # Simple 2-orbital, 2-electron system (1 alpha, 1 beta)
    n_orb = 2
    h_core = jnp.array([[-1.0, 0.1], [0.1, -0.5]])

    eri = jnp.zeros((2, 2, 2, 2))
    eri = eri.at[0, 0, 1, 1].set(0.2)
    eri = eri.at[1, 1, 0, 0].set(0.2)

    # Spin-separated determinants: alpha electron in orbital 0, beta in orbital 1
    det_alpha = 1  # 0b01: orbital 0 occupied
    det_beta = 2   # 0b10: orbital 1 occupied

    # Diagonal element
    H_diag = hamiltonian_element(det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri)
    print(f"Determinant: α={det_alpha} (0b{det_alpha:02b}), β={det_beta} (0b{det_beta:02b})")
    print(f"Diagonal matrix element: {H_diag:.6f}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
