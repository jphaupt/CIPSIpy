"""
Generate FCIDUMP file for H2 molecule using PySCF

This script creates the FCIDUMP file used for testing and examples.
It documents the exact parameters and versions used for reproducibility.

System requirements:
- Python >= 3.9
- PySCF >= 2.3.0

Usage:
    python generate_fcidump.py
"""

import sys
import os

try:
    import pyscf
    from pyscf import gto, scf, tools
except ImportError:
    print("Error: PySCF is not installed.")
    print("Install with: pip install pyscf")
    sys.exit(1)


def print_versions():
    """Print version information for reproducibility"""
    import numpy
    import scipy
    
    print("=" * 60)
    print("Version Information")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PySCF: {pyscf.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print("=" * 60)
    print()


def generate_h2_fcidump(
    bond_length=1.4,  # Bohr
    basis='sto-3g',
    output_file='FCIDUMP'
):
    """
    Generate FCIDUMP file for H2 molecule
    
    Parameters
    ----------
    bond_length : float
        H-H bond length in Bohr (default: 1.4 Bohr ≈ 0.74 Angstrom)
    basis : str
        Basis set name (default: 'sto-3g')
    output_file : str
        Output FCIDUMP filename (default: 'FCIDUMP')
    
    Returns
    -------
    None
        Writes FCIDUMP file to disk
    """
    
    print("=" * 60)
    print("Generating H2 FCIDUMP")
    print("=" * 60)
    print(f"Bond length: {bond_length} Bohr ({bond_length * 0.529177:.4f} Angstrom)")
    print(f"Basis set: {basis}")
    print(f"Output file: {output_file}")
    print()
    
    # Build molecule
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {bond_length}',
        basis=basis,
        unit='Bohr',
        symmetry=False,  # Disable symmetry for simplicity
        verbose=3
    )
    
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Number of orbitals: {mol.nao}")
    print(f"Nuclear repulsion: {mol.energy_nuc():.10f} Hartree")
    print()
    
    # Run Hartree-Fock calculation
    print("Running Hartree-Fock calculation...")
    mf = scf.RHF(mol)
    mf.kernel()
    
    print(f"HF energy: {mf.e_tot:.10f} Hartree")
    print()
    
    # Write FCIDUMP
    print(f"Writing FCIDUMP to {output_file}...")
    tools.fcidump.from_scf(mf, output_file, tol=1e-15)
    
    print("Done!")
    print()
    
    # Verify the file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✓ FCIDUMP file created successfully ({file_size} bytes)")
    else:
        print("✗ Error: FCIDUMP file was not created")
        sys.exit(1)


def main():
    """Main function"""
    print_versions()
    
    # Generate H2 FCIDUMP with standard parameters
    # These parameters match the test case used in the project
    generate_h2_fcidump(
        bond_length=1.4,      # Bohr (near equilibrium geometry)
        basis='sto-3g',       # Minimal basis set
        output_file='FCIDUMP'
    )
    
    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Verify the FCIDUMP file:")
    print("   python run_h2.py")
    print()
    print("2. Expected results:")
    print("   - Number of electrons: 2")
    print("   - Number of orbitals: 2")
    print("   - Nuclear repulsion: ~0.716 Hartree")
    print("   - HF energy: ~-1.114 Hartree")
    print("   - FCI energy (target): ~-1.137 Hartree")
    print("=" * 60)


if __name__ == "__main__":
    main()
