# How to Validate Your CIPSI Implementation

This document explains how to verify that your CIPSI implementation is correct.

## Quick Answer: Three-Level Validation Strategy

### Level 1: Component Testing (Unit Tests)
Each component should work correctly in isolation:
- ✅ FCIDUMP reader correctly parses integrals
- ✅ Determinants are generated and manipulated correctly  
- ✅ Hamiltonian matrix elements match Slater-Condon rules
- ✅ PT2 estimates have correct sign and magnitude

### Level 2: Small System Validation (H2)
Use H2 in minimal basis (STO-3G):
- **Known FCI Energy**: -1.137 Hartree
- **Test**: CIPSI should converge to this value
- **Why it works**: Only 6 determinants total, can verify by hand

### Level 3: Literature Comparison (Larger Systems)
Compare with published results or other codes (PySCF, Quantum Package)

## Detailed Validation for H2

### Expected Behavior

Starting from Hartree-Fock reference:

```
Iteration 0 (HF reference only):
  E_var = -1.113 Hartree
  E_PT2 = -0.024 Hartree (estimated)
  # determinants = 1

Iteration 1 (add most important determinants):
  E_var = -1.136 Hartree  
  E_PT2 = -0.001 Hartree
  # determinants = ~4-5

Iteration 2 (full CI reached):
  E_var = -1.137 Hartree
  E_PT2 = 0.000 Hartree
  # determinants = 6 (all possible)
```

### Key Checks

1. **Energy is variational**: E[i+1] ≤ E[i]
2. **PT2 decreases**: PT2[i+1] ≤ PT2[i]
3. **Converges to FCI**: Final E ≈ -1.137 Hartree
4. **Determinant count**: Eventually reaches 6

## Mathematical Validation

### 1. FCIDUMP Reader

**Test**: Read the H2 FCIDUMP and verify:
- Nuclear repulsion ≈ 0.716 Hartree
- h[0,0] ≈ -1.252 Hartree (most negative diagonal)
- Symmetries: h[i,j] = h[j,i]
- Symmetries: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)

**Status**: ✅ Implemented and tested

### 2. Slater Determinants

For H2 with 2 orbitals and 2 electrons, there are exactly 6 spin-adapted determinants:

```
1. |↑↓  ·  | - both in orbital 0 (HF reference)
2. |↑·  ↓· | - singlet coupled
3. |↑·  ·↓ | - singlet coupled  
4. |·↑  ↓· | - singlet coupled
5. |·↑  ·↓ | - singlet coupled
6. | ·   ↑↓| - both in orbital 1
```

**Test**: Generate all determinants and count them
- Should get exactly 6 for this system
- Each should be unique

### 3. Hamiltonian Matrix Elements

For the HF reference determinant |HF⟩ = |↑↓ · |:

**Diagonal (same determinant)**:
```
⟨HF|H|HF⟩ = 2*h[0,0] + (00|00) + E_nuc
         ≈ 2*(-1.252) + 0.675 + 0.716
         ≈ -1.113 Hartree
```

**Single excitation** (e.g., orbital 0→1 for one electron):
```
⟨HF|H|0→1⟩ = h[0,1] + Σ (exchange and coulomb terms)
           ≠ 0 (connects to wavefunction)
```

**Test**: Build 6×6 Hamiltonian for H2
- Diagonal elements should give reasonable energies
- Off-diagonal should have correct structure
- Lowest eigenvalue should be FCI energy

### 4. Full Diagonalization (FCI) Test

For H2, you can build the full 6×6 Hamiltonian and diagonalize:

**Expected result**:
- Lowest eigenvalue ≈ -1.137 Hartree
- This is your reference answer

**Test procedure**:
1. Generate all 6 determinants
2. Build 6×6 Hamiltonian matrix
3. Diagonalize with `jax.numpy.linalg.eigh`
4. Lowest eigenvalue is FCI energy
5. Compare with literature: -1.1372 Hartree (at R=1.4 bohr)

### 5. CIPSI Convergence

**PT2 accuracy test**: 
For any external determinant |D⟩ not in variational space:

```
E_PT2(D) = |⟨Ψ₀|H|D⟩|² / (E₀ - ⟨D|H|D⟩)
```

This should be:
- Negative (for ground state)
- Larger (more negative) for more important determinants
- Should sum to give total PT2 correction

**Test**: Before iteration 1→2:
- Calculate PT2 from remaining determinants
- Sum should match decrease in E_var when added

## Validation Against Other Codes

### Option 1: PySCF FCI

```python
from pyscf import gto, scf, fci

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')
mf = scf.RHF(mol).run()
cisolver = fci.FCI(mf)
e_fci = cisolver.kernel()[0]

# Should get: e_fci ≈ -1.137 Hartree
```

### Option 2: Quantum Package

Run CIPSI in Quantum Package on same FCIDUMP:
```bash
qp set_file FCIDUMP  
qp run fci
```

Compare final energies - should match to μHartree precision.

### Option 3: Analytical Solution

For H2 in minimal basis, there exist analytical solutions in the literature. 
Key reference: Szabo & Ostlund, "Modern Quantum Chemistry" (1996)

## Common Errors and How to Detect Them

### Error 1: Wrong Slater-Condon Rules
**Symptom**: FCI energy is wrong
**Test**: Calculate ⟨HF|H|HF⟩ by hand and compare

### Error 2: Missing Determinants  
**Symptom**: Never converges to FCI
**Test**: Count total determinants - should reach combinatorial limit

### Error 3: Wrong Phase (Sign)
**Symptom**: Wrong off-diagonal matrix elements
**Test**: Check that Hamiltonian is Hermitian

### Error 4: Wrong PT2 Formula
**Symptom**: PT2 has wrong sign or magnitude
**Test**: PT2 should be negative and small compared to E_var

### Error 5: Integer Overflow in Bit Manipulation
**Symptom**: Incorrect determinants generated
**Test**: Use 64-bit integers, check for negative numbers

## Summary: Validation Checklist

- [ ] FCIDUMP reader tests pass
- [ ] Can generate all determinants for H2 (count = 6)
- [ ] HF energy matches manual calculation
- [ ] Hamiltonian is Hermitian
- [ ] FCI energy on H2 matches literature (-1.137 Hartree)
- [ ] CIPSI converges to FCI energy
- [ ] PT2 correction decreases monotonically
- [ ] Energies are variational (always decreasing)
- [ ] Final variational space contains all important determinants
- [ ] (Optional) Results match PySCF or Quantum Package

## Next Steps After Validation

Once H2 works correctly:

1. **Test on HeH+**: Similar size, validates generality
2. **Test on larger basis**: H2 in 6-31G (10 orbitals)
3. **Test on larger molecules**: LiH, H2O in minimal basis
4. **Add GPU acceleration**: Profile and optimize with JAX
5. **Compare performance**: Timing vs other codes

## Quick Start: Minimal Validation

If you want to quickly check correctness:

```python
# 1. Read FCIDUMP
n_elec, n_orb, h_core, eri, e_nuc = read_fcidump('h2/FCIDUMP')

# 2. Build and diagonalize full H for H2
# (implement this in test_cipsi.py)
H = build_hamiltonian_matrix(...)  # 6x6 for H2
E, C = jnp.linalg.eigh(H)

# 3. Check
assert abs(E[0] - (-1.137)) < 0.001  # FCI energy
print(f"Success! FCI energy = {E[0]:.6f} Hartree")
```

If this passes, your Hamiltonian is correct and you can proceed to implement iterative CIPSI.
