import os
import sys
import numpy as np
from scipy.special import comb
from pyscf import gto, scf, mcscf, fci, tools

def run_master_benchmarks():
    # (Name, Basis, Geometry-Bohr, Charge, Spin, nFrozen)
    test_suite = [
        ("Li",   "sto-3g", "Li 0 0 0", 0, 1, 0),
        ("H2",   "sto-3g", "H 0 0 0; H 0 0 1.4", 0, 0, 0),
        ("HeH+", "sto-3g", "He 0 0 0; H 0 0 1.46", 1, 0, 0),
        ("H3+",  "3-21g",  "H 0 0 0; H 0 1.51 0; H 0 0.755 1.308", 1, 0, 0),
        ("LiH",  "6-31gstar", "Li 0 0 0; H 0 0 3.01", 0, 0, 1)  # Li 1s frozen
    ]

    for name, basis_label, geom, charge, spin, frozen in test_suite:
        # Create directory: {system}/{basis}/
        # Replacing * for folder name safety
        folder_name = basis_label.replace("*", "star")
        path = os.path.join(name, folder_name)
        os.makedirs(path, exist_ok=True)

        print(f"\nProcessing: {name} | Basis: {basis_label}")

        # Use 6-31g* for LiH, others as specified
        pyscf_basis = basis_label.replace("star", "*")

        # Generate FCIDUMP
        mol = gto.M(atom=geom, basis=pyscf_basis, charge=charge,
                    spin=spin, unit='Bohr', verbose=0)
        mf = scf.RHF(mol).run()

        dump_path = os.path.join(path, 'FCIDUMP')

        if frozen > 0:
            ncas = mol.nao - frozen
            nelecas = (mol.nelec[0] - frozen, mol.nelec[1] - frozen)
            cas = mcscf.CASCI(mf, ncas, nelecas)
            tools.fcidump.from_mcscf(cas, dump_path)
        else:
            tools.fcidump.from_scf(mf, dump_path)

        # Read FCIDUMP and Construct FCI Matrix
        data = tools.fcidump.read(dump_path)
        h1, h2, ecore = data['H1'], data['H2'], data['ECORE']
        norb, nelec = h1.shape[0], data['NELEC']

        # Determine determinant count for pspace
        n_a = nelec // 2 if isinstance(nelec, int) else nelec[0]
        n_b = nelec - n_a if isinstance(nelec, int) else nelec[1]
        n_det = int(comb(norb, n_a) * comb(norb, n_b))

        # pspace(..., np) returns (h_diag, h_matrix)
        # We set np=n_det to get the full matrix
        # see https://github.com/pyscf/pyscf/blob/master/examples/fci/15-FCI_hamiltonian.py
        h_fci = fci.direct_spin1.pspace(h1, h2, norb, (n_a, n_b), np=n_det)[1]
        h_total = h_fci + np.eye(n_det) * ecore

        # Save Outputs
        np.savetxt(os.path.join(path, 'fci_matrix.txt'), h_total)

        e_matrix = np.linalg.eigvalsh(h_total)[0]
        e_kernel = fci.direct_spin1.kernel(h1, h2, norb, (n_a, n_b), ecore=ecore)[0]

        summary_path = os.path.join(path, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"SYSTEM SUMMARY: {name}\n")
            f.write(f"{'='*30}\n")
            f.write(f"Basis:       {basis_label}\n")
            f.write(f"Orbitals:    {norb}\n")
            f.write(f"Electrons:   ({n_a}a, {n_b}b)\n")
            f.write(f"Determinants: {n_det}\n")
            f.write(f"Core Energy:  {ecore:.12f} a.u.\n")
            f.write(f"Ground State (Matrix): {e_matrix:.12f} a.u.\n")
            f.write(f"Ground State (Kernel): {e_kernel:.12f} a.u.\n")

        energy_path = os.path.join(path, 'e_gs.txt')
        with open(energy_path, "w") as f:
            f.write(f"{e_kernel:.12f}\n")

        print(f"Success. Matrix {n_det}x{n_det} saved. Energy: {e_matrix:.8f}")

if __name__ == "__main__":
    run_master_benchmarks()
