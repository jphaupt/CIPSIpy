# CIPSI Project: Problem Statement and Solution

## Problem Statement

**Goal**: Build a simple selected configuration interaction algorithm (CIPSI) with two objectives:
1. Become more familiar with CIPSI and how it parallelizes
2. Learn JAX and especially using it on GPUs

**Starting Point**: FCIDUMP file (from PySCF) containing molecular integrals

**Questions to Answer**:
1. What will be the project outline?
2. How will we know we have the right answer?

---

## Solution: Project Outline

### Phase 1: Foundation (COMPLETED ✅)
- [x] Project structure and documentation
- [x] FCIDUMP reader for molecular integrals
- [x] Test infrastructure
- [x] H2 example with known data

### Phase 2: Determinant Operations
- [ ] Bit string representation of Slater determinants
- [ ] Generate single and double excitations
- [ ] Calculate fermionic phases (signs)
- [ ] Unit tests for determinant operations

### Phase 3: Hamiltonian Matrix Elements
- [ ] Implement Slater-Condon rules
- [ ] Calculate ⟨Det_i|H|Det_j⟩ for any two determinants
- [ ] Build Hamiltonian matrix for set of determinants
- [ ] Test against hand calculations

### Phase 4: Basic CI (Full CI for small systems)
- [ ] Diagonalize Hamiltonian to get CI energy
- [ ] Validate on H2 minimal basis (6 determinants)
- [ ] Compare with literature FCI value (-1.137 Hartree)

### Phase 5: CIPSI Selection
- [ ] Calculate PT2 energy contributions
- [ ] Select important determinants based on threshold
- [ ] Test selection criteria

### Phase 6: Full CIPSI Algorithm
- [ ] Iterative procedure: diagonalize → select → grow
- [ ] Convergence checks
- [ ] Test on H2 and HeH+ systems

### Phase 7: JAX Optimization
- [ ] JIT compile hot loops
- [ ] Vectorize operations with vmap
- [ ] Test on GPU
- [ ] Performance profiling

### Phase 8: Documentation
- [ ] Document all modules
- [ ] Add usage examples
- [ ] Tutorial notebook

---

## How to Know You Have the Right Answer

### Level 1: Component Testing ✅

Each component works correctly in isolation:

| Component | Test | Status |
|-----------|------|--------|
| FCIDUMP reader | Parses H2 file, symmetries correct | ✅ PASS |
| Determinants | Generate all 6 for H2 | TODO |
| Hamiltonian | Matrix elements match Slater-Condon | TODO |
| CIPSI | Converges to FCI | TODO |

### Level 2: H2 Validation (Primary Test)

**Why H2?**
- Only 2 electrons, 2 orbitals → 6 total determinants
- Small enough to verify by hand
- Literature FCI value is well-known
- If H2 works, algorithm is correct

**Expected Results for H2 (STO-3G basis at R=1.4 bohr)**:

```
Iteration 0 (Hartree-Fock):
  E_var = -1.113 Hartree
  E_PT2 ≈ -0.024 Hartree
  N_det = 1

Iteration 1:
  E_var ≈ -1.136 Hartree
  E_PT2 ≈ -0.001 Hartree
  N_det = 4-5

Iteration 2 (Full CI):
  E_var = -1.137 Hartree  ← TARGET
  E_PT2 = 0.000 Hartree
  N_det = 6
```

**Success Criteria**:
- ✅ Final energy = -1.137 ± 0.001 Hartree
- ✅ Energy decreases monotonically (variational principle)
- ✅ PT2 correction decreases to zero
- ✅ Reaches all 6 determinants

### Level 3: Additional Validation

**Test Case 2: HeH+ (Helium hydride cation)**
- 2 electrons, 4 orbitals in minimal basis
- 20 total determinants
- Tests algorithm on slightly larger system
- Literature FCI: -2.927 Hartree (basis-dependent)

**Test Case 3: Comparison with Other Codes**
- PySCF FCI module
- Quantum Package CIPSI
- Should agree to μHartree precision

### Mathematical Validation

**1. Variational Principle**:
```
E[n+1] ≤ E[n]  for all iterations
```
If energy ever increases → bug in code

**2. PT2 Estimates**:
```
E_PT2 < 0  (for ground state)
|E_PT2[n+1]| < |E_PT2[n]|  (decreasing)
```
If PT2 is positive or increasing → bug in selection

**3. Hamiltonian Properties**:
```
H† = H  (Hermitian)
⟨ψ|H|ψ⟩ ≥ E_FCI  (variational)
```

**4. Determinant Count**:
For N electrons in M orbitals:
```
Max determinants = C(M,N_alpha) × C(M,N_beta)
For H2: C(2,1) × C(2,1) = 2 × 2 = 4 (spatial)
With spin: 6 (singlet-adapted)
```

### Validation Checklist

- [ ] Unit tests pass for all components
- [ ] H2 HF energy = -1.113 Hartree
- [ ] H2 FCI energy = -1.137 Hartree
- [ ] Energy strictly decreasing
- [ ] PT2 correction decreasing
- [ ] Hamiltonian is Hermitian
- [ ] Correct determinant count
- [ ] Results reproducible
- [ ] (Optional) Matches PySCF results

---

## How to Use This Project

### For Learning CIPSI:

1. **Read** `PROJECT_OUTLINE.md` for algorithm details
2. **Study** the H2 example (simplest possible case)
3. **Trace** through the iterations by hand
4. **Compare** with implemented code
5. **Verify** each component independently

### For Learning JAX:

1. **Start** with NumPy-style code
2. **Profile** to find bottlenecks
3. **Add JIT** to hot functions
4. **Vectorize** with vmap where applicable
5. **Test GPU** vs CPU performance
6. **Understand** how JAX transforms work

### Parallelization Opportunities (JAX Focus):

1. **PT2 Screening**: 
   - Most expensive step
   - Independent calculations for each external determinant
   - Perfect for vmap/GPU parallelization

2. **Matrix Construction**:
   - Many independent matrix elements
   - Can compute in parallel

3. **Determinant Generation**:
   - Generate excitations in batches
   - Vectorized bit operations

---

## Success Metrics

### Correctness (Primary Goal):
- ✅ H2 example matches literature FCI
- ✅ All unit tests pass
- ✅ Reproducible results

### Learning (Primary Goal):
- ✅ Understand CIPSI algorithm flow
- ✅ Know where parallelization helps
- ✅ Can use JAX for quantum chemistry
- ✅ Comfortable with JIT, vmap, GPU

### Performance (Secondary Goal):
- Can run H2 in < 1 second
- Can run small molecules (4-6 orbitals)
- GPU speedup over CPU (if available)
- Not trying to compete with production codes!

---

## Current Status

✅ **Phase 1 Complete**: Foundation is ready
- FCIDUMP reader working and tested
- H2 example data available
- Test infrastructure in place
- Documentation comprehensive

🚧 **Next Steps**: Phase 2 - Implement determinant operations
- Design bit string representation
- Implement excitation generation
- Add phase calculations
- Write unit tests

📚 **Documentation Available**:
- `PROJECT_OUTLINE.md` - Detailed implementation plan
- `VALIDATION.md` - How to verify correctness  
- `QUICKSTART.md` - Getting started guide
- `README.md` - Overview

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/scipy/fcidump.py` | Read molecular integrals | ✅ Complete |
| `src/scipy/determinants.py` | Slater determinant ops | ⏳ TODO |
| `src/scipy/hamiltonian.py` | Matrix elements | ⏳ TODO |
| `src/scipy/cipsi.py` | Main algorithm | ⏳ TODO |
| `examples/h2/FCIDUMP` | H2 test data | ✅ Complete |
| `examples/h2/run_h2.py` | H2 example | ✅ Complete |
| `tests/test_fcidump.py` | FCIDUMP tests | ✅ Passing |

---

## Summary

**You will know you have the right answer when:**

1. Your H2 calculation gives E = -1.137 Hartree (matches literature)
2. Energy decreases monotonically (variational principle)
3. PT2 correction goes to zero (convergence)
4. All unit tests pass
5. (Optional) Results match PySCF or Quantum Package

**The project outline is:**

1. ✅ Build foundation (FCIDUMP reader, tests, examples)
2. Implement determinant operations
3. Implement Hamiltonian matrix elements  
4. Test with full CI on H2
5. Add CIPSI selection algorithm
6. Optimize with JAX (JIT, vmap, GPU)
7. Document and benchmark

**Start here**: Run `python examples/h2/run_h2.py` and read `PROJECT_OUTLINE.md`
