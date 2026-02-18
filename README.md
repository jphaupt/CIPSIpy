# CIPSIpy

/ˈsɪpsɪpi/

A simple CIPSI (Configuration Interaction using a Perturbative Selection made Iteratively) implementation in JAX.

Disclaimer: Development of this package made extensive use of LLMs to speed up the process. 

## Overview

This is a minimal implementation of the CIPSI algorithm for learning purposes:
- Learn and understand CIPSI algorithm and its parallelization
- Gain practical experience with JAX, especially GPU acceleration
- Build a working quantum chemistry tool

## Features

- FCIDUMP file reader for molecular integrals from PySCF
- Slater determinant operations with efficient bit manipulation
- Hamiltonian matrix element calculation (Slater-Condon rules)
- Iterative CIPSI algorithm with PT2 selection
- JAX-based implementation for GPU acceleration

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

See the examples directory for working examples:

```bash
# Run H2 minimal basis example
python examples/h2/run_h2.py
```

## Project Status

This is a work in progress. See [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for detailed implementation plan and validation strategy.

Current status:
- [x] Project structure and outline
- [x] FCIDUMP reader
- [x] Slater determinant operations
- [x] Hamiltonian matrix elements
- [ ] Unfiltered batch CIPSI algorithm
- [ ] Validation tests
- [ ] Filtered batch CIPSI algorithm
- [ ] Parallelisation and optimisation

## Documentation

### Start Here
- **[ANSWER.md](ANSWER.md)** - Direct answer to "What is the project outline and how do we know we have the right answer?"
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes

### Detailed Guides
- **[PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)** - Complete implementation plan with algorithm details
- **[VALIDATION.md](VALIDATION.md)** - Comprehensive testing and validation strategy

## License

MIT License - see [LICENSE](LICENSE) for details
