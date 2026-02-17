"""
FCIDUMP file reader for molecular integrals.

FCIDUMP format contains:
- One-electron integrals: h[i,j]
- Two-electron integrals: (ij|kl) in physicist notation
- Nuclear repulsion energy
- Number of electrons and orbitals
"""

import jax.numpy as jnp
import numpy as np


def read_fcidump(filename):
    """
    Read molecular integrals from FCIDUMP file.
    
    Args:
        filename: Path to FCIDUMP file
        
    Returns:
        tuple: (n_elec, n_orb, h_core, eri, e_nuc)
            - n_elec: Number of electrons
            - n_orb: Number of spatial orbitals
            - h_core: One-electron integrals [n_orb, n_orb]
            - eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]
            - e_nuc: Nuclear repulsion energy
            
    FCIDUMP format:
        Line 1: &FCI NORB=N,NELEC=M,MS2=S,
        Lines 2+: integral_value i j k l
        
        where (i,j,k,l) indices indicate:
        - i,j,k,l > 0: Two-electron integral (ij|kl)
        - i,j > 0, k=l=0: One-electron integral h[i,j]
        - i=j=k=l=0: Nuclear repulsion energy
    """
    with open(filename, 'r') as f:
        # Read header line
        header = f.readline()
        
        # Parse header (format: &FCI NORB=N,NELEC=M,MS2=S,)
        n_orb = None
        n_elec = None
        
        # Simple parsing - look for NORB and NELEC
        parts = header.upper().split(',')
        for part in parts:
            if 'NORB' in part:
                n_orb = int(part.split('=')[1])
            elif 'NELEC' in part:
                n_elec = int(part.split('=')[1])
        
        if n_orb is None or n_elec is None:
            raise ValueError("Could not parse NORB and NELEC from FCIDUMP header")
        
        # Skip any additional header lines
        line = f.readline()
        while line.strip() and not line.strip()[0].replace('-', '').replace('.', '').replace('E', '').replace('e', '').replace('+', '').isdigit():
            line = f.readline()
        
        # Initialize arrays
        h_core = np.zeros((n_orb, n_orb))
        eri = np.zeros((n_orb, n_orb, n_orb, n_orb))
        e_nuc = 0.0
        
        # Read integrals
        while line:
            parts = line.split()
            if len(parts) < 5:
                line = f.readline()
                continue
                
            value = float(parts[0])
            i, j, k, l = map(int, parts[1:5])
            
            # FCIDUMP uses 1-based indexing
            if i == 0 and j == 0 and k == 0 and l == 0:
                # Nuclear repulsion
                e_nuc = value
            elif k == 0 and l == 0:
                # One-electron integral
                # Convert to 0-based indexing
                h_core[i-1, j-1] = value
                h_core[j-1, i-1] = value  # Symmetric
            else:
                # Two-electron integral (ij|kl) in physicist notation
                # Convert to 0-based indexing
                eri[i-1, j-1, k-1, l-1] = value
                # Apply all permutation symmetries
                # (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
                eri[j-1, i-1, k-1, l-1] = value
                eri[i-1, j-1, l-1, k-1] = value
                eri[j-1, i-1, l-1, k-1] = value
                eri[k-1, l-1, i-1, j-1] = value
                eri[l-1, k-1, i-1, j-1] = value
                eri[k-1, l-1, j-1, i-1] = value
                eri[l-1, k-1, j-1, i-1] = value
            
            line = f.readline()
    
    return n_elec, n_orb, jnp.array(h_core), jnp.array(eri), e_nuc


def write_fcidump(filename, n_elec, n_orb, h_core, eri, e_nuc, ms2=0):
    """
    Write molecular integrals to FCIDUMP file.
    
    Args:
        filename: Path to output FCIDUMP file
        n_elec: Number of electrons
        n_orb: Number of spatial orbitals
        h_core: One-electron integrals [n_orb, n_orb]
        eri: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]
        e_nuc: Nuclear repulsion energy
        ms2: 2*Sz (default 0 for singlet)
    """
    with open(filename, 'w') as f:
        # Write header
        f.write(f" &FCI NORB={n_orb},NELEC={n_elec},MS2={ms2},\n")
        f.write("  ORBSYM=" + ",".join(["1"]*n_orb) + ",\n")
        f.write("  ISYM=1,\n")
        f.write(" &END\n")
        
        # Write two-electron integrals (physicist notation: (ij|kl))
        for i in range(n_orb):
            for j in range(i+1):
                for k in range(n_orb):
                    for l in range(k+1):
                        # Only write unique integrals due to symmetry
                        if (i*(i+1)//2 + j) >= (k*(k+1)//2 + l):
                            val = eri[i, j, k, l]
                            if abs(val) > 1e-12:
                                f.write(f" {val:23.16E} {i+1:3d} {j+1:3d} {k+1:3d} {l+1:3d}\n")
        
        # Write one-electron integrals
        for i in range(n_orb):
            for j in range(i+1):
                val = h_core[i, j]
                if abs(val) > 1e-12:
                    f.write(f" {val:23.16E} {i+1:3d} {j+1:3d} {0:3d} {0:3d}\n")
        
        # Write nuclear repulsion
        f.write(f" {e_nuc:23.16E} {0:3d} {0:3d} {0:3d} {0:3d}\n")
