"""
Validation tests comparing CIPSIpy Hamiltonian matrix elements with PySCF reference.

These tests read pre-generated reference data from PySCF and perform element-wise
comparison with our CIPSIpy implementation to ensure correctness.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from cipsipy.fcidump import read_fcidump
from cipsipy.hamiltonian import hamiltonian_element_spin

# Path to validation data
VALIDATION_DATA_DIR = Path(__file__).parent / "validation_data"


class TestPySCFValidation:
    """Validation tests against PySCF reference data."""

    @pytest.fixture
    def pyscf_reference_data(self):
        """Load PySCF reference data."""
        H_ref = np.load(VALIDATION_DATA_DIR / "h2_pyscf_hamiltonian.npy")
        h1e_ref = np.load(VALIDATION_DATA_DIR / "h2_pyscf_h1e.npy")
        eri_ref = np.load(VALIDATION_DATA_DIR / "h2_pyscf_eri.npy")
        e_nuc_ref = np.load(VALIDATION_DATA_DIR / "h2_pyscf_enuc.npy")[0]

        return {"H": H_ref, "h1e": h1e_ref, "eri": eri_ref, "e_nuc": e_nuc_ref}

    @pytest.fixture
    def cipsipy_data(self):
        """Load CIPSIpy data from FCIDUMP."""
        fcidump_path = Path(__file__).parent.parent / "examples" / "h2" / "FCIDUMP"
        n_elec, n_orb, spin, h_core, eri, e_nuc = read_fcidump(str(fcidump_path))

        return {"n_elec": n_elec, "n_orb": n_orb, "h_core": h_core, "eri": eri, "e_nuc": e_nuc}

    def test_integrals_match_pyscf(self, pyscf_reference_data, cipsipy_data):
        """Test that integrals from FCIDUMP match PySCF reference."""
        # One-electron integrals
        assert jnp.allclose(
            cipsipy_data["h_core"], pyscf_reference_data["h1e"], atol=1e-6
        ), "One-electron integrals do not match PySCF reference"

        # Two-electron integrals
        assert jnp.allclose(
            cipsipy_data["eri"], pyscf_reference_data["eri"], atol=1e-6
        ), "Two-electron integrals do not match PySCF reference"

        # Nuclear repulsion
        assert jnp.isclose(
            cipsipy_data["e_nuc"], pyscf_reference_data["e_nuc"], atol=1e-6
        ), "Nuclear repulsion does not match PySCF reference"

    def test_hamiltonian_matrix_matches_pyscf(self, pyscf_reference_data, cipsipy_data):
        """Test that Hamiltonian matrix elements match PySCF reference element-wise."""
        H_ref = pyscf_reference_data["H"]
        n_orb = cipsipy_data["n_orb"]
        h_core = cipsipy_data["h_core"]
        eri = cipsipy_data["eri"]

        # Build CIPSIpy Hamiltonian matrix
        # Determinants: |00>, |01>, |10>, |11>
        determinants = [
            (1, 1),  # |00>: alpha=0b01 (orbital 0), beta=0b01 (orbital 0)
            (1, 2),  # |01>: alpha=0b01 (orbital 0), beta=0b10 (orbital 1)
            (2, 1),  # |10>: alpha=0b10 (orbital 1), beta=0b01 (orbital 0)
            (2, 2),  # |11>: alpha=0b10 (orbital 1), beta=0b10 (orbital 1)
        ]

        H_cipsipy = jnp.zeros((4, 4))

        for i in range(4):
            det_i_alpha, det_i_beta = determinants[i]
            for j in range(4):
                det_j_alpha, det_j_beta = determinants[j]

                H_cipsipy = H_cipsipy.at[i, j].set(
                    hamiltonian_element_spin(
                        det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
                    )
                )

        # Element-wise comparison
        print("\nPySCF Reference Matrix:")
        print(H_ref)
        print("\nCIPSIpy Matrix:")
        print(np.array(H_cipsipy))
        print("\nDifference:")
        print(np.array(H_cipsipy) - H_ref)

        # Check each element
        tolerance = 1e-6
        max_diff = jnp.max(jnp.abs(H_cipsipy - H_ref))

        assert jnp.allclose(
            H_cipsipy, H_ref, atol=tolerance
        ), f"Hamiltonian matrix elements do not match PySCF reference. Max difference: {max_diff:.2e}"

    def test_diagonal_elements_match_pyscf(self, pyscf_reference_data, cipsipy_data):
        """Test that diagonal Hamiltonian elements match PySCF."""
        H_ref = pyscf_reference_data["H"]
        n_orb = cipsipy_data["n_orb"]
        h_core = cipsipy_data["h_core"]
        eri = cipsipy_data["eri"]

        determinants = [(1, 1), (1, 2), (2, 1), (2, 2)]
        det_names = ["|00>", "|01>", "|10>", "|11>"]

        print("\nDiagonal Element Comparison:")
        for i, (det, name) in enumerate(zip(determinants, det_names)):
            det_alpha, det_beta = det
            H_cipsipy = hamiltonian_element_spin(
                det_alpha, det_beta, det_alpha, det_beta, n_orb, h_core, eri
            )
            H_ref_val = H_ref[i, i]
            diff = float(H_cipsipy - H_ref_val)

            print(f"  {name}: CIPSIpy={H_cipsipy:.10f}, PySCF={H_ref_val:.10f}, Diff={diff:.2e}")

            assert jnp.isclose(
                H_cipsipy, H_ref_val, atol=1e-6
            ), f"Diagonal element {name} does not match PySCF reference"

    def test_off_diagonal_elements_match_pyscf(self, pyscf_reference_data, cipsipy_data):
        """Test that off-diagonal Hamiltonian elements match PySCF."""
        H_ref = pyscf_reference_data["H"]
        n_orb = cipsipy_data["n_orb"]
        h_core = cipsipy_data["h_core"]
        eri = cipsipy_data["eri"]

        determinants = [(1, 1), (1, 2), (2, 1), (2, 2)]
        det_names = ["|00>", "|01>", "|10>", "|11>"]

        print("\nOff-Diagonal Element Comparison:")
        for i in range(4):
            for j in range(i + 1, 4):
                det_i_alpha, det_i_beta = determinants[i]
                det_j_alpha, det_j_beta = determinants[j]

                H_cipsipy = hamiltonian_element_spin(
                    det_i_alpha, det_i_beta, det_j_alpha, det_j_beta, n_orb, h_core, eri
                )
                H_ref_val = H_ref[i, j]
                diff = float(H_cipsipy - H_ref_val)

                print(
                    f"  <{det_names[i]}|H|{det_names[j]}>: CIPSIpy={H_cipsipy:.10f}, PySCF={H_ref_val:.10f}, Diff={diff:.2e}"
                )

                assert jnp.isclose(
                    H_cipsipy, H_ref_val, atol=1e-6
                ), f"Off-diagonal element <{det_names[i]}|H|{det_names[j]}> does not match PySCF reference"
