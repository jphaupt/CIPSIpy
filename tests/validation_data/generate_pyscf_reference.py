"""
Generate reference Hamiltonian matrix using PySCF for validation.

This script builds the H2 Hamiltonian matrix manually using PySCF's integrals
and saves it for comparison with our CIPSIpy implementation.
"""

from pyscf import gto, scf
import numpy as np


def generate_h2_reference_matrix():
    """Generate reference H2 Hamiltonian matrix using PySCF."""
    # Build H2 molecule - same parameters as FCIDUMP
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr", symmetry=False)

    # Run HF
    mf = scf.RHF(mol)
    mf.kernel()

    # Get MO integrals
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_ao = mol.intor("int2e")
    eri = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", eri_ao, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff
    )

    # Build 4x4 Hamiltonian matrix
    # Determinants: |00>, |01>, |10>, |11>
    # where |ij> means alpha in orbital i, beta in orbital j

    H = np.zeros((4, 4))

    # Diagonal elements
    # |00>: both in orbital 0
    H[0, 0] = 2 * h1e[0, 0] + eri[0, 0, 0, 0]

    # |01>: alpha in 0, beta in 1
    H[1, 1] = h1e[0, 0] + h1e[1, 1] + eri[0, 0, 1, 1]

    # |10>: alpha in 1, beta in 0
    H[2, 2] = h1e[1, 1] + h1e[0, 0] + eri[1, 1, 0, 0]

    # |11>: both in orbital 1
    H[3, 3] = 2 * h1e[1, 1] + eri[1, 1, 1, 1]

    # Off-diagonal elements
    # <00|H|11>: double excitation, both electrons 0->1
    H[0, 3] = H[3, 0] = eri[0, 0, 1, 1]

    # <01|H|10>: exchange of alpha and beta
    H[1, 2] = H[2, 1] = eri[0, 1, 1, 0]

    return H, h1e, eri, mol.energy_nuc()


if __name__ == "__main__":
    print("=" * 70)
    print("Generating PySCF Reference Hamiltonian Matrix for H2/STO-3G")
    print("=" * 70)

    H_ref, h1e, eri, e_nuc = generate_h2_reference_matrix()

    print("\nReference Hamiltonian Matrix:")
    print(H_ref)

    print("\nOne-electron integrals:")
    print(h1e)

    print("\nTwo-electron integrals (selected):")
    print(f"eri[0,0,0,0] = {eri[0,0,0,0]:.10f}")
    print(f"eri[0,0,1,1] = {eri[0,0,1,1]:.10f}")
    print(f"eri[0,1,0,1] = {eri[0,1,0,1]:.10f}")

    print(f"\nNuclear repulsion: {e_nuc:.10f}")

    # Diagonalize to get reference energy
    eigenvalues, eigenvectors = np.linalg.eigh(H_ref)
    E_ground = eigenvalues[0] + e_nuc

    print(f"\nGround state energy from diagonalization: {E_ground:.10f} Ha")
    print(f"(Electronic: {eigenvalues[0]:.10f} + Nuclear: {e_nuc:.10f})")

    # Save for validation tests
    np.save("h2_pyscf_hamiltonian.npy", H_ref)
    np.save("h2_pyscf_h1e.npy", h1e)
    np.save("h2_pyscf_eri.npy", eri)
    np.save("h2_pyscf_enuc.npy", np.array([e_nuc]))

    print("\n" + "=" * 70)
    print("Reference data saved to:")
    print("  - h2_pyscf_hamiltonian.npy")
    print("  - h2_pyscf_h1e.npy")
    print("  - h2_pyscf_eri.npy")
    print("  - h2_pyscf_enuc.npy")
    print("=" * 70)
