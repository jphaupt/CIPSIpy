# CIPSI Implementation Project Outline

## Overview
This project implements a Configuration Interaction using a Perturbative Selection made Iteratively (CIPSI) algorithm in JAX, with a focus on GPU parallelization and learning JAX fundamentals.

## Goals
1. Learn and understand the CIPSI algorithm and its parallelization strategies
2. Gain practical experience with JAX, particularly GPU acceleration
3. Build a working (though not necessarily highly optimized) CIPSI implementation

## Project Structure

```
SCIpy/
├── src/
│   └── cipsypy/
│       ├── __init__.py
│       ├── fcidump.py          # FCIDUMP file parser
│       ├── determinants.py     # Slater determinant operations
│       ├── hamiltonian.py      # Hamiltonian matrix elements
│       ├── cipsi.py            # Main CIPSI algorithm
│       ├── selection.py        # Determinant selection step
│       └── utils.py            # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_fcidump.py
│   ├── test_determinants.py
│   ├── test_hamiltonian.py
│   └── test_cipsi.py
├── examples/
│   ├── h2/                     # H2 molecule example
│   │   ├── FCIDUMP
│   │   └── run_h2.py
│   └── heh_plus/               # HeH+ example
│       ├── FCIDUMP
│       └── run_heh.py
├── pyproject.toml
├── README.md
└── PROJECT_OUTLINE.md
```

## CIPSI Algorithm Overview

### What is CIPSI?
CIPSI is a selected configuration interaction method that:
1. Starts with a reference determinant (typically Hartree-Fock)
2. Iteratively grows the wavefunction by selecting important determinants
3. Uses perturbation theory to estimate the importance of external determinants
4. Continues until convergence or target accuracy is reached

### Algorithm Steps

#### 1. Initialization
- Read molecular integrals from FCIDUMP file
- Set up the reference determinant (HF state)
- Initialize the variational space with the reference

#### 2. Diagonalization
- Build Hamiltonian matrix for current variational space
- Diagonalize to get current CI coefficients and energy
- This is the "variational" energy

#### 3. Selection (PT2 calculation)
- Loop over all determinants in variational space
- Generate single and double excitations
- For each external determinant:
  - Calculate coupling to variational space
  - Estimate PT2 energy contribution: ΔE_PT2 ≈ |⟨Ψ|H|D⟩|² / (E₀ - ⟨D|H|D⟩)
  - Calculate selection criterion (e.g., |ΔE_PT2|)

#### 4. Growth
- Select determinants exceeding selection threshold
- Add selected determinants to variational space
- Return to step 2

#### 5. Convergence
- Check if PT2 correction is below threshold
- Check if energy has converged
- Check if no new determinants were added

## Implementation Details

### 1. FCIDUMP Reader (`fcidump.py`)
**Purpose**: Parse FCIDUMP files containing molecular integrals

**Key Functions**:
- `read_fcidump(filename)`: Parse FCIDUMP file
- Returns: `n_elec`, `n_orb`, `h_core`, `eri`, `nuclear_repulsion`

**Format**: FCIDUMP contains:
- One-electron integrals: h[i,j]
- Two-electron integrals: (ij|kl) in physicist notation
- Nuclear repulsion energy

### 2. Slater Determinants (`determinants.py`)
**Purpose**: Represent and manipulate Slater determinants efficiently

**Key Data Structure**:
- Use bit strings to represent occupied orbitals
- Alpha and beta electrons stored separately
- Example: |0110⟩ means orbitals 1 and 2 occupied

**Key Functions**:
- `create_determinant(occ_alpha, occ_beta)`: Create determinant from occupation lists
- `apply_excitation(det, holes, particles)`: Generate excited determinant
- `generate_excitations(det, n_orb, max_rank=2)`: Generate all singles/doubles
- `phase_single(det, i, a)`: Calculate fermionic phase for excitation
- `phase_double(det, i, j, a, b)`: Calculate fermionic phase for double excitation

### 3. Hamiltonian Matrix Elements (`hamiltonian.py`)
**Purpose**: Calculate ⟨Det_i|H|Det_j⟩

**Key Functions**:
- `hamiltonian_element(det_i, det_j, h_core, eri)`: 
  - Same determinant: Sum of orbital energies
  - Single excitation: One-electron + two-electron terms
  - Double excitation: Two-electron term only
  - Higher: 0 (Slater-Condon rules)

**Slater-Condon Rules**:
- 0 excitations: Full diagonal energy
- 1 excitation i→a: h[i,a] + Σ_k [(ia|kk) - (ik|ka)]
- 2 excitations i,j→a,b: (ij|ab)

### 4. CIPSI Main Algorithm (`cipsi.py`)
**Purpose**: Orchestrate the CIPSI iterations

**Key Functions**:
- `cipsi(fcidump_file, selection_threshold, max_iterations)`:
  - Main driver function
  - Returns energy trajectory and final wavefunction

**Workflow**:
```python
# Initialize
dets = [reference_determinant]
energies = []

for iteration in range(max_iterations):
    # Diagonalize
    H = build_hamiltonian(dets, h_core, eri)
    E, C = jnp.linalg.eigh(H)
    energies.append(E[0])
    
    # Selection
    candidates, pt2_contribs = select_determinants(
        dets, C[:, 0], h_core, eri, threshold
    )
    
    # Check convergence
    total_pt2 = jnp.sum(pt2_contribs)
    if total_pt2 < convergence_threshold:
        break
    
    # Grow
    dets.extend(candidates)

return energies, dets, C[:, 0]
```

### 5. Selection Step (`selection.py`)
**Purpose**: Select important determinants via PT2

**Key Functions**:
- `calculate_pt2_contributions(det, wfn_dets, wfn_coeffs, h_core, eri)`:
  - For given external determinant
  - Calculate coupling to current wavefunction
  - Estimate PT2 contribution

- `select_determinants(wfn_dets, wfn_coeffs, h_core, eri, threshold)`:
  - Generate all singles/doubles from current wavefunction
  - Calculate PT2 for each
  - Return those above threshold

**PT2 Formula**:
```
ΔE_PT2(D) = |⟨Ψ₀|H|D⟩|² / (E₀ - ⟨D|H|D⟩)

where:
  ⟨Ψ₀|H|D⟩ = Σ_i c_i ⟨D_i|H|D⟩
```

## Validation Strategy

### How to Know We Have the Right Answer

#### 1. Exact Benchmarks (Small Systems)
For systems small enough for Full CI:
- **H2 (2 electrons, minimal basis)**: 2-4 spatial orbitals
  - Run FCI and CIPSI
  - Final CIPSI energy should match FCI exactly when all determinants included
  
- **HeH+ (2 electrons)**: Similar size
  - Validate against published FCI results
  - Check PT2 estimates against actual energy lowering

#### 2. Energy Checks
- Energy should be strictly decreasing (variational principle)
- E_variational + E_PT2 should be smooth
- Final energy should be above exact FCI energy (variational upper bound)

#### 3. PT2 Convergence
- PT2 correction should decrease monotonically
- Eventually PT2 → 0 as we approach FCI

#### 4. Determinant Selection
- Number of determinants should grow each iteration
- Eventually saturate when all important determinants included

#### 5. Comparison with Reference Codes
- Compare against: Quantum Package, PySCF's CIPSI, NECI
- For small systems, energies should agree to μHartree precision

### Test Cases

#### Test 1: H2 Minimal Basis (STO-3G)
- 2 electrons, 2 spatial orbitals
- Only 6 determinants total (spin-adapted)
- Should converge to FCI in 1-2 iterations

**Expected Results**:
```
Iteration 0: E = -1.117 Hartree (HF energy)
Iteration 1: E = -1.137 Hartree (FCI energy)
PT2 correction: < 10^-6 Hartree
```

#### Test 2: HeH+ Minimal Basis
- 2 electrons, 4 spatial orbitals
- Should reach FCI (20 determinants)

#### Test 3: H2 Extended Basis (6-31G)
- 2 electrons, 10 spatial orbitals
- Cannot do full FCI (184,756 determinants)
- Check convergence behavior
- Compare with literature values

### Validation Checklist

- [ ] FCIDUMP parser reads correctly (compare with manual parsing)
- [ ] Determinant generation produces correct count (combinatorics)
- [ ] Hamiltonian diagonal matches sum of orbital energies
- [ ] Hamiltonian off-diagonal matches hand calculation for small case
- [ ] Energy is variational (always decreasing)
- [ ] PT2 correction has correct sign (negative for ground state)
- [ ] H2 minimal converges to literature FCI value
- [ ] Increasing basis size gives lower energy (up to FCI limit)

## Implementation Phases

### Phase 1: Core Infrastructure ✓ (Current)
- [x] Project outline document
- [ ] Basic Python package structure
- [ ] Dependencies (pyproject.toml with JAX)
- [ ] FCIDUMP reader
- [ ] Test FCIDUMP reader with H2 example

### Phase 2: Determinant Operations
- [ ] Determinant data structure (bit strings)
- [ ] Excitation generation
- [ ] Phase calculation (fermionic antisymmetry)
- [ ] Unit tests for determinant operations

### Phase 3: Hamiltonian Matrix Elements
- [ ] Implement Slater-Condon rules
- [ ] Test against hand-calculated examples
- [ ] Verify with small H2 Hamiltonian

### Phase 4: Basic CI
- [ ] Build Hamiltonian matrix
- [ ] Diagonalization
- [ ] Test FCI on H2 minimal basis
- [ ] Verify energy matches literature

### Phase 5: CIPSI Selection
- [ ] PT2 contribution calculation
- [ ] Determinant selection logic
- [ ] Test on H2 to verify PT2 estimates

### Phase 6: Full CIPSI Algorithm
- [ ] Integrate all components
- [ ] Iterative algorithm
- [ ] Convergence checks
- [ ] Test on H2 and HeH+

### Phase 7: JAX Optimization
- [ ] JIT compilation of hot loops
- [ ] Vectorization where possible
- [ ] GPU testing
- [ ] Performance profiling

### Phase 8: Documentation and Examples
- [ ] Document all modules
- [ ] Add usage examples
- [ ] Create tutorial notebook
- [ ] Performance notes

## JAX Considerations

### Where JAX Helps
1. **Automatic Differentiation**: Not directly used in CI, but useful for forces/properties
2. **JIT Compilation**: Speed up matrix builds and diagonalization
3. **Vectorization**: Batch process determinant generation
4. **GPU Acceleration**: 
   - Matrix diagonalization
   - Large-scale PT2 screening
   - Hamiltonian matrix construction

### JAX Best Practices
- Use `jax.numpy` instead of `numpy`
- Design pure functions for JIT compilation
- Avoid Python loops in hot paths
- Use `vmap` for vectorization
- Be careful with dynamic shapes

### Initial Implementation
- Start with simple NumPy-style code
- Profile to find bottlenecks
- Gradually add JIT and vmap
- Test GPU vs CPU performance

## Expected Challenges

1. **Memory Management**: Storing all determinants becomes expensive
   - Solution: Iterative generation, don't store all at once
   
2. **Hamiltonian Matrix Build**: O(N²) scaling in determinants
   - Solution: Only compute needed elements, use sparsity
   
3. **Determinant Generation**: Combinatorial explosion
   - Solution: Efficient bit manipulation, avoid duplicates
   
4. **PT2 Screening**: Most expensive step
   - Solution: Early rejection, parallel processing on GPU

5. **Numerical Precision**: Energy differences can be small
   - Solution: Use float64, careful accumulation

## Success Criteria

The implementation will be considered successful when:

1. **Correctness**: H2 and HeH+ examples match FCI within 10^-6 Hartree
2. **Functionality**: Can read FCIDUMP, run CIPSI, output converged energy
3. **Learning**: Code demonstrates understanding of CIPSI and JAX concepts
4. **Documentation**: Clear explanation of algorithm and validation

## References

1. CIPSI Original Papers:
   - Huron, Malrieu, Rancurel, J. Chem. Phys. 58, 5745 (1973)
   - Evangelisti, J. Chem. Phys. 134, 224105 (2011)

2. Quantum Package: https://github.com/QuantumPackage/qp2
   - Reference CIPSI implementation

3. JAX Documentation: https://jax.readthedocs.io/

4. Related Implementations:
   - PySCF: https://github.com/pyscf/pyscf
   - NECI: https://github.com/ghb24/NECI_STABLE

## Next Steps

1. Set up basic Python package structure
2. Create pyproject.toml with JAX dependencies  
3. Implement and test FCIDUMP reader
4. Generate H2 example FCIDUMP files for testing
5. Begin determinant operations implementation
