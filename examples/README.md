# Examples

This directory contains example calculations and test data for the CIPSIpy project.

## Available Examples

### H2 Molecule (`h2/`)

The H2 molecule in minimal basis (STO-3G) serves as the primary test case for CIPSI implementation.

**Features:**
- Simple 2-electron, 2-orbital system
- Full CI tractable (6 determinants)
- Well-known reference values
- Ideal for validation

**Files:**
- `FCIDUMP` - Molecular integrals
- `generate_fcidump.py` - Script to regenerate data with versioning
- `run_h2.py` - Example runner
- `requirements.txt` - Dependencies for data generation
- `README.md` - Detailed documentation

**Quick Start:**
```bash
cd h2/
python run_h2.py
```

**To regenerate data:**
```bash
cd h2/
pip install -r requirements.txt
python generate_fcidump.py
```

See `h2/README.md` for complete documentation including:
- Data generation parameters
- Software version requirements
- Expected results
- Validation criteria

## Data Reproducibility

All example data files (e.g., FCIDUMP files) include:
1. Generation scripts with documented parameters
2. Software version requirements
3. Expected results for validation
4. Step-by-step instructions to regenerate

This ensures reproducibility and allows verification that the data matches expectations.

## Adding New Examples

When adding new examples, please include:

1. **Generation script** - Code to create the data from scratch
2. **README** - Documentation with:
   - System parameters (geometry, basis, etc.)
   - Software versions used
   - Expected results
   - Regeneration instructions
3. **requirements.txt** - Dependencies for data generation
4. **Validation** - Expected values to verify correctness

## Future Examples

Planned examples include:
- HeH+ (4 orbitals) - Next complexity level
- LiH (minimal basis) - Different molecule type
- H2 (6-31G) - Larger basis set
- H2O (minimal basis) - Polyatomic molecule
