"""
Generate FCIDUMP file for H2 molecule using PySCF for CIPSIpy

This script creates the FCIDUMP file used for testing and examples.
It also optionally runs an FCI calculation to provide the exact ground-state
energy target for benchmarking.

System requirements:
- Python >= 3.9
- PySCF >= 2.3.0
"""

import sys
import os

try:
    import pyscf
    from pyscf import gto, scf, tools, fci
except ImportError:
    print("Error: PySCF is not installed.")
    print("Install with: pip install pyscf")
    sys.exit(1)


def print_versions():
    """Print version information for reproducibility"""
    import numpy
    import scipy

    print("=" * 60)
    print("CIPSIpy - Environment Information")
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
    output_file='FCIDUMP',
    run_fci=True      # The optional FCI step
):
    """
    Generate FCIDUMP file for H2 molecule and optionally run FCI
    """

    print("=" * 60)
    print("Generating H2 Data for CIPSIpy")
    print("=" * 60)
    print(f"Bond length: {bond_length} Bohr")
    print(f"Basis set: {basis}")
    print(f"Output file: {output_file}")
    print()

    # Build molecule
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {bond_length}',
        basis=basis,
        unit='Bohr',
        symmetry=False,
        verbose=0  # Reduced verbosity to keep output clean
    )

    # Run Hartree-Fock calculation
    print("Step 1: Running Hartree-Fock (RHF)...")
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    print(f"✓ HF energy: {hf_energy:.10f} Hartree")

    # Write FCIDUMP
    print(f"Step 2: Writing FCIDUMP to {output_file}...")
    tools.fcidump.from_scf(mf, output_file, tol=1e-15)

    # Optional FCI Step
    fci_energy = None
    if run_fci:
        print("Step 3: Running Full CI (FCI) Benchmark...")
        cisolver = fci.FCI(mf)
        fci_energy = cisolver.kernel()[0]
        correlation_energy = fci_energy - hf_energy
        print(f"✓ FCI energy:         {fci_energy:.10f} Hartree")
        print(f"✓ Correlation energy: {correlation_energy:.10f} Hartree")
    else:
        print("Step 3: Skipping FCI Benchmark.")

    print("\nGeneration Complete!")
    return hf_energy, fci_energy


def main():
    """Main function"""
    print_versions()

    # Generate H2 FCIDUMP and run FCI as truth
    hf_e, fci_e = generate_h2_fcidump(
        bond_length=1.4,
        basis='sto-3g',
        output_file='FCIDUMP',
        run_fci=True
    )

    print()
    print("=" * 60)
    print("CIPSIpy Benchmarking Target")
    print("=" * 60)
    print(f"Reference HF:  {hf_e:.10f}")
    if fci_e:
        print(f"Target FCI:    {fci_e:.10f}")
        print(f"Total Corr:    {(fci_e - hf_e):.10f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
