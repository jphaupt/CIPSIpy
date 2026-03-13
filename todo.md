# CIPSI Python JAX implementation

## Key Goals

The following are my loose goals for the project. If only finishing the first, it's already a success, but it would be nice to get the rest.

- [x] Working unfiltered CIPSI implementation (not purely "vibe-coded" since the goal is to understand the algorithm).
- [ ] JAX optimisation.
- [ ] Documentation and explanations on algorithm (could probably also be used for the "real" SCI documentation for the group). Might also be fun to write about smaller details, e.g. radix sort, that I found interesting.
- [ ] Working filtered CIPSI implementation.
- [ ] A clean implementation, reasonably performant implementation that can be used for rapidly testing new ideas and implementations, though I never expect to be as performant as e.g. quantum package.


## Project Outline

### Phase 1: Foundation
- [x] Project structure and documentation
- [x] FCIDUMP reader for molecular integrals
- [x] Test infrastructure
- [x] H2 example with known data

### Phase 2: Determinant Operations
- [x] Bit string representation of Slater determinants
- [x] Generate single and double excitations
- [x] Calculate fermionic phases (signs)
- [x] Unit tests for determinant operations

### Phase 3: Hamiltonian Matrix Elements
- [x] Implement Slater-Condon rules
- [x] Calculate ⟨Det_i|H|Det_j⟩ for any two determinants
- [x] Build Hamiltonian matrix for set of determinants
- [x] Test against hand calculations

### Phase 4: Basic CI (Full CI for small systems)
- [x] Diagonalize Hamiltonian to get CI energy
- [x] Validate on H2 minimal basis (6 determinants)
- [x] Compare with literature FCI value (-1.137 Hartree)

### Phase 5: CIPSI Selection
- [x] Calculate PT2 energy contributions
- [x] CIPSI based on "external to internal" algorithm, using double-ionised generator determinants and tagging
- [x] Select important determinants based on threshold
- [x] Test selection criteria

### Phase 6: Full CIPSI Algorithm (Unfiltered)
- [x] Iterative procedure: diagonalize → select → grow
- [x] Convergence checks
- [x] Test on H2 and HeH+ systems

### Phase 7: JAX Optimization
- [ ] JIT compile hot loops
- [ ] Vectorize operations with vmap
- [ ] Test on GPU
- [ ] Performance profiling

### Phase 8: Documentation
- [ ] Document all modules
- [x] Add usage examples
- [ ] Tutorial notebook

### Phase 9: Filtered CIPSI
- [ ] Review sCI thesis to better understand what is done here
