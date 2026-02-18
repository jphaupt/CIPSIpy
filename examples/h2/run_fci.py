# exists for debugging purposes
import numpy as np
from pyscf import fci, tools

fcidump_file = 'FCIDUMP'

result = tools.fcidump.read(fcidump_file)

h1 = result['H1']
h2 = result['H2']
ecore = result['ECORE']
nelec = result['NELEC']
norb = h1.shape[0]

print(f"Number of orbitals: {norb}")
print(f"Number of electrons: {nelec}")

# We use a dummy object or directly call the fci module functions
cisolver = fci.direct_spin1.FCI()

energy, fcivec = cisolver.kernel(h1, h2, norb, nelec, ecore=ecore)

print(f"FCI Total Energy: {energy:.12f}")

# ============================================================================
# OUTPUT
# ============================================================================
# Parsing FCIDUMP
# Number of orbitals: 2
# Number of electrons: 2
# FCI Total Energy: -1.137275943617
