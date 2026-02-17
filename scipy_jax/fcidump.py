"""
FCIDUMP file parser for quantum chemistry integrals.

The FCIDUMP format is a standard format for storing molecular integrals
used in quantum chemistry calculations.
"""

import re
from typing import Dict, Tuple, Optional
import numpy as np


class FCIDump:
    """
    Parser for FCIDUMP files containing molecular integrals.
    
    Attributes:
        norb: Number of orbitals
        nelec: Number of electrons
        ms2: 2 * total spin (2*S)
        isym: Symmetry (1 = no symmetry)
        orbsym: Orbital symmetries
        one_electron_integrals: One-electron integrals h[i,j]
        two_electron_integrals: Two-electron integrals (ij|kl)
        nuclear_repulsion: Nuclear repulsion energy
    """
    
    def __init__(self, filename: Optional[str] = None):
        """
        Initialize FCIDUMP parser.
        
        Args:
            filename: Path to FCIDUMP file to parse
        """
        self.norb: int = 0
        self.nelec: int = 0
        self.ms2: int = 0
        self.isym: int = 1
        self.orbsym: list = []
        self.one_electron_integrals: Dict[Tuple[int, int], float] = {}
        self.two_electron_integrals: Dict[Tuple[int, int, int, int], float] = {}
        self.nuclear_repulsion: float = 0.0
        
        if filename:
            self.parse(filename)
    
    def parse(self, filename: str) -> None:
        """
        Parse an FCIDUMP file.
        
        Args:
            filename: Path to FCIDUMP file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header_end = 0
        header_text = ""
        in_header = False
        for i, line in enumerate(lines):
            if '&FCI' in line.upper():
                in_header = True
                header_text += line
                continue
            if in_header:
                header_text += line
                if '/' in line:
                    header_end = i + 1
                    break
        
        # Extract parameters from header
        self._parse_header(header_text)
        
        # Parse integrals
        self._parse_integrals(lines[header_end:])
    
    def _parse_header(self, header_text: str) -> None:
        """
        Parse header section of FCIDUMP file.
        
        Args:
            header_text: Header text containing NORB, NELEC, etc.
        """
        # Remove comments
        header_text = re.sub(r'!.*', '', header_text)
        
        # Extract NORB
        match = re.search(r'NORB\s*=\s*(\d+)', header_text, re.IGNORECASE)
        if match:
            self.norb = int(match.group(1))
        
        # Extract NELEC
        match = re.search(r'NELEC\s*=\s*(\d+)', header_text, re.IGNORECASE)
        if match:
            self.nelec = int(match.group(1))
        
        # Extract MS2
        match = re.search(r'MS2\s*=\s*(-?\d+)', header_text, re.IGNORECASE)
        if match:
            self.ms2 = int(match.group(1))
        
        # Extract ISYM
        match = re.search(r'ISYM\s*=\s*(\d+)', header_text, re.IGNORECASE)
        if match:
            self.isym = int(match.group(1))
        
        # Extract ORBSYM
        match = re.search(r'ORBSYM\s*=\s*([\d,\s]+)', header_text, re.IGNORECASE)
        if match:
            orbsym_str = match.group(1)
            self.orbsym = [int(x) for x in re.findall(r'\d+', orbsym_str)]
    
    def _parse_integrals(self, lines: list) -> None:
        """
        Parse integral values from FCIDUMP file.
        
        Args:
            lines: Lines containing integral values
        """
        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                value = float(parts[0])
                i = int(parts[1])
                j = int(parts[2])
                k = int(parts[3])
                l = int(parts[4])
                
                if i == 0 and j == 0 and k == 0 and l == 0:
                    # Nuclear repulsion energy
                    self.nuclear_repulsion = value
                elif k == 0 and l == 0:
                    # One-electron integral
                    self.one_electron_integrals[(i, j)] = value
                else:
                    # Two-electron integral
                    self.two_electron_integrals[(i, j, k, l)] = value
            except (ValueError, IndexError):
                continue
    
    def get_one_electron_matrix(self) -> np.ndarray:
        """
        Get one-electron integrals as a matrix.
        
        Returns:
            numpy array of shape (norb, norb) containing one-electron integrals
        """
        h = np.zeros((self.norb, self.norb))
        for (i, j), value in self.one_electron_integrals.items():
            # FCIDUMP uses 1-based indexing
            h[i-1, j-1] = value
            h[j-1, i-1] = value  # Hermitian
        return h
    
    def get_two_electron_tensor(self) -> np.ndarray:
        """
        Get two-electron integrals as a tensor.
        
        Returns:
            numpy array of shape (norb, norb, norb, norb) containing two-electron integrals
        """
        v = np.zeros((self.norb, self.norb, self.norb, self.norb))
        for (i, j, k, l), value in self.two_electron_integrals.items():
            # FCIDUMP uses 1-based indexing
            # Store with 8-fold symmetry
            idx_set = [
                (i-1, j-1, k-1, l-1),
                (j-1, i-1, l-1, k-1),
                (k-1, l-1, i-1, j-1),
                (l-1, k-1, j-1, i-1),
                (i-1, j-1, l-1, k-1),
                (j-1, i-1, k-1, l-1),
                (k-1, l-1, j-1, i-1),
                (l-1, k-1, i-1, j-1),
            ]
            for idx in idx_set:
                v[idx] = value
        return v
    
    def __repr__(self) -> str:
        """String representation of FCIDump object."""
        return (f"FCIDump(norb={self.norb}, nelec={self.nelec}, "
                f"ms2={self.ms2}, isym={self.isym}, "
                f"E_nuc={self.nuclear_repulsion:.6f})")
