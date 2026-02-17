# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jphaupt/SCIpy.git
cd SCIpy
```

2. Install dependencies:
```bash
pip install -e .
```

For development (includes testing tools):
```bash
pip install -e ".[dev]"
```

## Running Your First Example

```bash
python examples/h2/run_h2.py
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

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_fcidump.py -v

# Run with coverage
pytest tests/ --cov=CIPSIpy
```

## Project Structure

```
SCIpy/
├── src/CIPSIpy/        # Main source code
│   ├── fcidump.py      # FCIDUMP reader ✅
│   ├── determinants.py # Determinant operations (TODO)
│   ├── hamiltonian.py  # Matrix elements (TODO)
│   └── cipsi.py        # Main algorithm (TODO)
├── tests/              # Unit tests
├── examples/           # Example calculations
│   └── h2/            # H2 molecule example ✅
└── docs/              # Documentation
    ├── PROJECT_OUTLINE.md  # Detailed plan
    └── VALIDATION.md       # How to verify correctness
```

## Current Status

✅ **Working**:
- FCIDUMP reader for molecular integrals
- H2 example with test data
- Basic test infrastructure

🚧 **In Progress**:
- Slater determinant operations
- Hamiltonian matrix construction
- CIPSI algorithm
- GPU acceleration with JAX

## Next Steps for Development

See [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for the complete implementation plan.

Phase 2 (Current): Implement Slater determinant operations
- Bit string representation
- Excitation generation
- Phase calculation

## Understanding the Files

### src/CIPSIpy/fcidump.py
Reads FCIDUMP files containing molecular integrals from quantum chemistry codes like PySCF.

**Key functions**:
- `read_fcidump(filename)` - Parse FCIDUMP file
- `write_fcidump(filename, ...)` - Write FCIDUMP file

### examples/h2/FCIDUMP
Contains molecular integrals for H2 molecule in minimal basis (STO-3G):
- 2 electrons
- 2 spatial orbitals
- Simplest possible test case

### examples/h2/run_h2.py
Demonstrates how to:
- Read FCIDUMP file
- Display molecular information
- Calculate Hartree-Fock energy estimate

## Learning Resources

### CIPSI Algorithm
- Huron, Malrieu, Rancurel, J. Chem. Phys. 58, 5745 (1973) - Original paper
- Quantum Package documentation: https://quantum-package.readthedocs.io/

### JAX
- JAX documentation: https://jax.readthedocs.io/
- JAX tutorial: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
- GPU programming with JAX: https://jax.readthedocs.io/en/latest/gpu-memory-allocation.html

### Quantum Chemistry
- Szabo & Ostlund, "Modern Quantum Chemistry" - Chapter on CI methods
- Helgaker, Jorgensen, Olsen, "Molecular Electronic Structure Theory" - Comprehensive reference

## Getting Help

1. Check [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for algorithm details
2. Check [VALIDATION.md](VALIDATION.md) for testing strategy
3. Look at existing tests in `tests/` for examples
4. Read the source code - it's designed to be educational!

## Contributing

This is a learning project. Feel free to:
- Add more examples
- Improve documentation
- Optimize performance
- Add features

## Tips for Development

1. **Start small**: Get H2 working first before moving to larger systems
2. **Test frequently**: Run tests after each change
3. **Validate early**: Check against known FCI results
4. **Profile before optimizing**: Use JAX profiler to find bottlenecks
5. **Use JIT gradually**: Start with pure Python, add JIT to hot paths

## Common Issues

### Import errors
Make sure you installed the package:
```bash
pip install -e .
```

### JAX not using GPU
Check GPU availability:
```python
import jax
print(jax.devices())  # Should show GPU if available
```

To force CPU (useful for debugging):
```bash
export JAX_PLATFORM_NAME=cpu
```

### Test failures
Run with verbose output to see details:
```bash
pytest tests/ -v -s
```

## Example Workflow

1. **Read the outline**:
   ```bash
   cat PROJECT_OUTLINE.md
   ```

2. **Understand validation**:
   ```bash
   cat VALIDATION.md
   ```

3. **Run existing tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Try the H2 example**:
   ```bash
   python examples/h2/run_h2.py
   ```

5. **Implement next component** (e.g., determinants.py)

6. **Write tests** for new component

7. **Validate** against known results

8. **Repeat** for next component

Happy coding! 🚀
