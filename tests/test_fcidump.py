"""
Tests for FCIDUMP parser.
"""

import os
import tempfile
import numpy as np
from scipy_jax.fcidump import FCIDump


def test_parse_fcidump_header():
    """Test parsing of FCIDUMP header."""
    fcidump_content = """&FCI NORB= 4, NELEC= 4, MS2= 0, ISYM= 1,
 ORBSYM=1,1,1,1,
 /
  0.0  0  0  0  0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fcidump', delete=False) as f:
        f.write(fcidump_content)
        fname = f.name
    
    try:
        fcidump = FCIDump(fname)
        assert fcidump.norb == 4
        assert fcidump.nelec == 4
        assert fcidump.ms2 == 0
        assert fcidump.isym == 1
        assert fcidump.orbsym == [1, 1, 1, 1]
    finally:
        os.unlink(fname)


def test_parse_fcidump_integrals():
    """Test parsing of FCIDUMP integrals."""
    fcidump_content = """&FCI NORB= 2, NELEC= 2, MS2= 0, ISYM= 1,
 ORBSYM=1,1,
 /
  1.0  1  1  1  1
  0.5  2  1  1  1
 -1.0  1  1  0  0
 -0.5  2  2  0  0
  3.14159  0  0  0  0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fcidump', delete=False) as f:
        f.write(fcidump_content)
        fname = f.name
    
    try:
        fcidump = FCIDump(fname)
        
        # Check two-electron integrals
        assert (1, 1, 1, 1) in fcidump.two_electron_integrals
        assert fcidump.two_electron_integrals[(1, 1, 1, 1)] == 1.0
        
        # Check one-electron integrals
        assert (1, 1) in fcidump.one_electron_integrals
        assert fcidump.one_electron_integrals[(1, 1)] == -1.0
        
        # Check nuclear repulsion
        assert abs(fcidump.nuclear_repulsion - 3.14159) < 1e-6
    finally:
        os.unlink(fname)


def test_get_one_electron_matrix():
    """Test conversion of one-electron integrals to matrix."""
    fcidump_content = """&FCI NORB= 2, NELEC= 2, MS2= 0, ISYM= 1,
 /
 -1.0  1  1  0  0
 -0.5  2  2  0  0
 -0.3  1  2  0  0
  0.0  0  0  0  0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fcidump', delete=False) as f:
        f.write(fcidump_content)
        fname = f.name
    
    try:
        fcidump = FCIDump(fname)
        h = fcidump.get_one_electron_matrix()
        
        assert h.shape == (2, 2)
        assert abs(h[0, 0] - (-1.0)) < 1e-10
        assert abs(h[1, 1] - (-0.5)) < 1e-10
        assert abs(h[0, 1] - (-0.3)) < 1e-10
        assert abs(h[1, 0] - (-0.3)) < 1e-10  # Hermitian
    finally:
        os.unlink(fname)


def test_get_two_electron_tensor():
    """Test conversion of two-electron integrals to tensor."""
    fcidump_content = """&FCI NORB= 2, NELEC= 2, MS2= 0, ISYM= 1,
 /
  1.0  1  1  1  1
  0.0  0  0  0  0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fcidump', delete=False) as f:
        f.write(fcidump_content)
        fname = f.name
    
    try:
        fcidump = FCIDump(fname)
        v = fcidump.get_two_electron_tensor()
        
        assert v.shape == (2, 2, 2, 2)
        # Check 8-fold symmetry
        assert abs(v[0, 0, 0, 0] - 1.0) < 1e-10
    finally:
        os.unlink(fname)


def test_fcidump_repr():
    """Test string representation of FCIDump."""
    fcidump_content = """&FCI NORB= 4, NELEC= 4, MS2= 0, ISYM= 1,
 /
  3.14159  0  0  0  0
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fcidump', delete=False) as f:
        f.write(fcidump_content)
        fname = f.name
    
    try:
        fcidump = FCIDump(fname)
        repr_str = repr(fcidump)
        assert 'FCIDump' in repr_str
        assert 'norb=4' in repr_str
        assert 'nelec=4' in repr_str
    finally:
        os.unlink(fname)


if __name__ == '__main__':
    test_parse_fcidump_header()
    test_parse_fcidump_integrals()
    test_get_one_electron_matrix()
    test_get_two_electron_tensor()
    test_fcidump_repr()
    print("All FCIDUMP tests passed!")
