# SCIpy

Mini SCI toy program written in JAX

## Features

- **FCIDUMP Parser**: Parse FCIDUMP files containing molecular integrals
- **Determinant Bitstrings**: Efficient bitstring representation of Slater determinants

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### FCIDUMP Parser

```python
from scipy_jax import FCIDump

# Parse FCIDUMP file
fcidump = FCIDump('path/to/file.fcidump')

# Access properties
print(f"Number of orbitals: {fcidump.norb}")
print(f"Number of electrons: {fcidump.nelec}")
print(f"Nuclear repulsion: {fcidump.nuclear_repulsion}")

# Get integrals as matrices/tensors
h = fcidump.get_one_electron_matrix()  # Shape: (norb, norb)
v = fcidump.get_two_electron_tensor()  # Shape: (norb, norb, norb, norb)
```

### Determinant Bitstrings

```python
from scipy_jax import Determinant

# Create determinant from occupation lists
det = Determinant.from_occupation(
    alpha_occ=[0, 1],  # Alpha electrons in orbitals 0 and 1
    beta_occ=[0, 1],   # Beta electrons in orbitals 0 and 1
    norb=4             # Total number of orbitals
)

# Check occupation
print(det.is_alpha_occupied(0))  # True

# Count electrons
print(f"Total electrons: {det.count_electrons()}")

# Create excited determinant
excited = det.excitation(from_orbital=1, to_orbital=2, spin='alpha')

# Calculate excitation level
alpha_exc, beta_exc = det.excitation_level(excited)
print(f"Excitation: {alpha_exc} alpha, {beta_exc} beta")
```

## Examples

See the `examples/` directory for complete usage examples:

```bash
python examples/usage.py
```

## Testing

```bash
PYTHONPATH=. python tests/test_fcidump.py
PYTHONPATH=. python tests/test_determinant.py
```

## License

MIT License - see LICENSE file for details.
