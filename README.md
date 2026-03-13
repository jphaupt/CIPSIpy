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

## Todo List

See `todo.md` for project goals, and todo lists separated by project "phases".

## License

MIT License - see [LICENSE](LICENSE) for details
