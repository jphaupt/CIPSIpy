import sys
import numpy as np
from scipy.special import comb
from pyscf import fci, tools

data = tools.fcidump.read('FCIDUMP')
h1, h2, ecore = data['H1'], data['H2'], data['ECORE']
norb, nelec = h1.shape[0], data['NELEC']

# Number of Determinants
n_a = nelec // 2 if isinstance(nelec, int) else nelec[0]
n_b = nelec - n_a if isinstance(nelec, int) else nelec[1]
n_det = int(comb(norb, n_a) * comb(norb, n_b))

print(f"Orbitals: {norb} | Electrons: ({n_a}a, {n_b}b) | Determinants: {n_det}")

# Safety Check
if n_det > 20:
    sys.exit(f"Error: Determinant count {n_det} exceeds limit.")

# Construct and display the Full FCI Matrix
# pspace(..., np) returns (h_diag, h_matrix)
h_fci = fci.direct_spin1.pspace(h1, h2, norb, (n_a, n_b), np=n_det)[1]
h_total = h_fci + np.eye(n_det) * ecore

print("\nFull FCI Hamiltonian Matrix:")
print(np.round(h_total, 8))

# 5. Quick Verification
e_min = np.linalg.eigvalsh(h_total)[0]
print(f"\nGround State Energy: {e_min:.12f}")
