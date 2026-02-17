# H2 Molecule Example

This directory contains test data and examples for the H2 molecule in a minimal basis set (STO-3G).

## Files

- `FCIDUMP` - Molecular integrals for H2 molecule
- `generate_fcidump.py` - Script to regenerate the FCIDUMP file
- `run_h2.py` - Example script that reads and displays the FCIDUMP data

## FCIDUMP Data Generation

The FCIDUMP file was generated using PySCF with the following parameters:

### System Parameters
- **Molecule**: H2 (hydrogen molecule)
- **Bond length**: 1.4 Bohr (≈ 0.74 Angstrom)
- **Basis set**: STO-3G (minimal basis)
- **Symmetry**: Disabled (C1 symmetry)

### Software Versions

The FCIDUMP was originally generated with:
- Python 3.9+
- PySCF >= 2.3.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0

### Expected Properties
- Number of electrons: 2
- Number of spatial orbitals: 2
- Number of determinants (full CI): 6
- Nuclear repulsion energy: ~0.716 Hartree
- Hartree-Fock energy: ~-1.114 Hartree
- FCI energy (target): ~-1.137 Hartree

## Regenerating the FCIDUMP

To regenerate the FCIDUMP file with current software versions:

```bash
# Install PySCF if not already installed
pip install pyscf

# Generate FCIDUMP
python generate_fcidump.py
```

The script will:
1. Print version information for reproducibility
2. Build the H2 molecule with specified parameters
3. Run a Hartree-Fock calculation
4. Write the FCIDUMP file
5. Verify the file was created successfully

## Running the Example

To test the FCIDUMP reader:

```bash
python run_h2.py
```

Expected output:
```
============================================================
H2 Molecule Example - Minimal Basis (STO-3G)
============================================================

Number of electrons: 2
Number of orbitals: 2
Nuclear repulsion: 0.715996 Hartree

One-electron integrals (h_core):
[[-1.2520906  -0.47529125]
 [-0.47529125 -0.47663063]]

Two-electron integrals shape: (2, 2, 2, 2)
Number of non-zero ERIs: 16

Estimated HF energy: -1.113537 Hartree
```

## Validation

The H2 molecule in STO-3G basis is an ideal test case because:

1. **Small size**: Only 2 electrons and 2 orbitals, making it easy to verify by hand
2. **Well-known**: Literature values are readily available for validation
3. **Full CI tractable**: Only 6 determinants total, so full CI is trivial
4. **Reference values**:
   - HF energy: -1.1136 Hartree
   - FCI energy: -1.1373 Hartree (at R=1.4 Bohr)

## References

- Szabo & Ostlund, "Modern Quantum Chemistry" (1996) - contains H2 examples
- PySCF documentation: https://pyscf.org/
- FCIDUMP format specification: https://theochem.github.io/horton/2.1.0b3/user_hamiltonian_io.html

## Troubleshooting

### PySCF Installation Issues

If you encounter issues installing PySCF:

```bash
# Try with pip
pip install pyscf

# Or with conda
conda install -c pyscf pyscf
```

### Numerical Differences

Small numerical differences (< 1e-10) in the FCIDUMP are expected due to:
- Different BLAS/LAPACK implementations
- Different compiler optimizations
- Different PySCF versions

These differences should not affect the test results as long as they're below numerical precision thresholds.
