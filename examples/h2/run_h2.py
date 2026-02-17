"""
Example: H2 molecule in minimal basis (STO-3G)

This is the simplest possible test case:
- 2 electrons
- 2 spatial orbitals
- Only 6 possible determinants total

Expected FCI energy: approximately -1.137 Hartree
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from scipy.fcidump import read_fcidump


def main():
    """Run H2 example"""
    print("=" * 60)
    print("H2 Molecule Example - Minimal Basis (STO-3G)")
    print("=" * 60)
    print()
    
    # Read FCIDUMP
    fcidump_path = os.path.join(os.path.dirname(__file__), 'FCIDUMP')
    n_elec, n_orb, h_core, eri, e_nuc = read_fcidump(fcidump_path)
    
    print(f"Number of electrons: {n_elec}")
    print(f"Number of orbitals: {n_orb}")
    print(f"Nuclear repulsion: {e_nuc:.6f} Hartree")
    print()
    
    print("One-electron integrals (h_core):")
    print(h_core)
    print()
    
    print("Two-electron integrals shape:", eri.shape)
    print(f"Number of non-zero ERIs: {(abs(eri) > 1e-10).sum()}")
    print()
    
    # Calculate HF energy (diagonal element of first determinant)
    # For closed-shell: E = 2*h[0,0] + (2*(00|00) - (00|00))
    # But need to sum over occupied orbitals
    # For minimal H2, both electrons in orbital 0 (assuming both alpha and beta)
    hf_energy = 2 * h_core[0, 0] + eri[0, 0, 0, 0] + e_nuc
    print(f"Estimated HF energy: {hf_energy:.6f} Hartree")
    print()
    
    # TODO: Run CIPSI when implemented
    print("CIPSI implementation coming soon!")


if __name__ == "__main__":
    main()
