"""
Tests for FCIDUMP file parsing
"""

import os
import sys
import tempfile

import jax.numpy as jnp

# For development: add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cipsipy.fcidump import read_fcidump, write_fcidump


class TestFCIDUMP:
    """Test FCIDUMP reader and writer"""

    def test_read_h2_fcidump(self):
        """Test reading H2 FCIDUMP file"""
        fcidump_path = os.path.join(os.path.dirname(__file__), "../examples/h2/FCIDUMP")

        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(fcidump_path)

        # Check dimensions
        assert n_elec == 2
        assert n_orb == 2
        assert spin == 0
        assert h_core.shape == (2, 2)
        assert eri.shape == (2, 2, 2, 2)

        # Check that h_core is symmetric
        assert jnp.allclose(h_core, h_core.T)

        # Check nuclear repulsion is positive
        assert e_nuc > 0

        # Check some known values (approximate)
        assert jnp.abs(e_nuc - 1 / 1.40) < 0.0001

    def test_write_read_roundtrip(self):
        """Test that write then read gives same data"""
        # Create simple test data
        n_elec = 2
        n_orb = 2
        spin = 0
        h_core = jnp.array([[1.0, 0.5], [0.5, 0.8]])
        eri = jnp.ones((2, 2, 2, 2)) * 0.1
        e_nuc = 1.5

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fcidump") as f:
            tmp_path = f.name

        try:
            write_fcidump(tmp_path, n_elec, n_orb, h_core, eri, e_nuc)

            # Read back
            n_elec_read, n_orb_read, spin_read, h_core_read, eri_read, e_nuc_read = read_fcidump(
                tmp_path
            )

            # Check values match
            assert n_elec_read == n_elec
            assert n_orb_read == n_orb
            assert spin_read == spin
            assert jnp.allclose(h_core_read, h_core, atol=1e-10)
            assert jnp.allclose(eri_read, eri, atol=1e-10)
            assert abs(e_nuc_read - e_nuc) < 1e-10
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_eri_symmetries(self):
        """Test that ERIs have correct permutation symmetries"""
        fcidump_path = os.path.join(os.path.dirname(__file__), "../examples/h2/FCIDUMP")

        _, _, _, _, eri, _ = read_fcidump(fcidump_path)

        # Check 8-fold symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = ...
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        val = eri[i, j, k, l]
                        assert jnp.allclose(eri[j, i, k, l], val)
                        assert jnp.allclose(eri[i, j, l, k], val)
                        assert jnp.allclose(eri[k, l, i, j], val)


if __name__ == "__main__":
    # Run tests
    test = TestFCIDUMP()
    print("Testing read_h2_fcidump...")
    test.test_read_h2_fcidump()
    print("✓ Passed")

    print("Testing write_read_roundtrip...")
    test.test_write_read_roundtrip()
    print("✓ Passed")

    print("Testing eri_symmetries...")
    test.test_eri_symmetries()
    print("✓ Passed")

    print("\nAll tests passed!")
