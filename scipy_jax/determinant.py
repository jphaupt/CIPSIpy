"""
Determinant representation using bitstrings.

Slater determinants are represented as bitstrings where each bit
indicates whether an orbital is occupied (1) or unoccupied (0).
"""

from typing import List, Tuple, Optional
import numpy as np


class Determinant:
    """
    Represents a Slater determinant using bitstrings.
    
    For unrestricted spin, we use two separate bitstrings:
    - alpha_string: occupation of alpha (spin-up) orbitals
    - beta_string: occupation of beta (spin-down) orbitals
    
    For restricted spin, both spins have the same occupation.
    
    Attributes:
        alpha_string: Bitstring for alpha electrons
        beta_string: Bitstring for beta electrons
        norb: Number of spatial orbitals
    """
    
    def __init__(
        self,
        alpha_string: int,
        beta_string: int,
        norb: int
    ):
        """
        Initialize a determinant from bitstrings.
        
        Args:
            alpha_string: Bitstring representing alpha electron occupation
            beta_string: Bitstring representing beta electron occupation
            norb: Number of spatial orbitals
        """
        self.alpha_string = alpha_string
        self.beta_string = beta_string
        self.norb = norb
    
    @classmethod
    def from_occupation(
        cls,
        alpha_occ: List[int],
        beta_occ: List[int],
        norb: int
    ) -> 'Determinant':
        """
        Create determinant from occupation lists.
        
        Args:
            alpha_occ: List of occupied alpha orbitals (0-indexed)
            beta_occ: List of occupied beta orbitals (0-indexed)
            norb: Number of spatial orbitals
        
        Returns:
            Determinant object
        
        Example:
            >>> det = Determinant.from_occupation([0, 1], [0, 1], 4)
            >>> bin(det.alpha_string)
            '0b11'
        """
        alpha_string = 0
        for orb in alpha_occ:
            alpha_string |= (1 << orb)
        
        beta_string = 0
        for orb in beta_occ:
            beta_string |= (1 << orb)
        
        return cls(alpha_string, beta_string, norb)
    
    def to_occupation(self) -> Tuple[List[int], List[int]]:
        """
        Convert bitstrings to occupation lists.
        
        Returns:
            Tuple of (alpha_occ, beta_occ) lists
        
        Example:
            >>> det = Determinant(0b11, 0b11, 4)
            >>> det.to_occupation()
            ([0, 1], [0, 1])
        """
        alpha_occ = []
        beta_occ = []
        
        for i in range(self.norb):
            if self.alpha_string & (1 << i):
                alpha_occ.append(i)
            if self.beta_string & (1 << i):
                beta_occ.append(i)
        
        return alpha_occ, beta_occ
    
    def count_alpha(self) -> int:
        """
        Count number of alpha electrons.
        
        Returns:
            Number of alpha electrons
        """
        return bin(self.alpha_string).count('1')
    
    def count_beta(self) -> int:
        """
        Count number of beta electrons.
        
        Returns:
            Number of beta electrons
        """
        return bin(self.beta_string).count('1')
    
    def count_electrons(self) -> int:
        """
        Count total number of electrons.
        
        Returns:
            Total number of electrons
        """
        return self.count_alpha() + self.count_beta()
    
    def is_alpha_occupied(self, orbital: int) -> bool:
        """
        Check if an alpha orbital is occupied.
        
        Args:
            orbital: Orbital index (0-indexed)
        
        Returns:
            True if orbital is occupied
        """
        return bool(self.alpha_string & (1 << orbital))
    
    def is_beta_occupied(self, orbital: int) -> bool:
        """
        Check if a beta orbital is occupied.
        
        Args:
            orbital: Orbital index (0-indexed)
        
        Returns:
            True if orbital is occupied
        """
        return bool(self.beta_string & (1 << orbital))
    
    def excitation(
        self,
        from_orbital: int,
        to_orbital: int,
        spin: str
    ) -> 'Determinant':
        """
        Create a new determinant by exciting an electron.
        
        Args:
            from_orbital: Orbital to excite from (0-indexed)
            to_orbital: Orbital to excite to (0-indexed)
            spin: 'alpha' or 'beta'
        
        Returns:
            New Determinant object with excitation applied
        """
        if spin == 'alpha':
            new_alpha = self.alpha_string
            # Remove electron from from_orbital
            new_alpha &= ~(1 << from_orbital)
            # Add electron to to_orbital
            new_alpha |= (1 << to_orbital)
            return Determinant(new_alpha, self.beta_string, self.norb)
        elif spin == 'beta':
            new_beta = self.beta_string
            # Remove electron from from_orbital
            new_beta &= ~(1 << from_orbital)
            # Add electron to to_orbital
            new_beta |= (1 << to_orbital)
            return Determinant(self.alpha_string, new_beta, self.norb)
        else:
            raise ValueError(f"Invalid spin: {spin}. Must be 'alpha' or 'beta'")
    
    def excitation_level(self, other: 'Determinant') -> Tuple[int, int]:
        """
        Calculate excitation level between two determinants.
        
        Args:
            other: Another Determinant object
        
        Returns:
            Tuple of (alpha_excitations, beta_excitations)
        """
        alpha_diff = self.alpha_string ^ other.alpha_string
        beta_diff = self.beta_string ^ other.beta_string
        
        alpha_excitations = bin(alpha_diff).count('1') // 2
        beta_excitations = bin(beta_diff).count('1') // 2
        
        return alpha_excitations, beta_excitations
    
    def __eq__(self, other: object) -> bool:
        """Check equality of two determinants."""
        if not isinstance(other, Determinant):
            return False
        return (self.alpha_string == other.alpha_string and
                self.beta_string == other.beta_string and
                self.norb == other.norb)
    
    def __hash__(self) -> int:
        """Hash function for determinants."""
        return hash((self.alpha_string, self.beta_string, self.norb))
    
    def __repr__(self) -> str:
        """String representation of determinant."""
        alpha_occ, beta_occ = self.to_occupation()
        return (f"Determinant(alpha={alpha_occ}, beta={beta_occ}, "
                f"norb={self.norb})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        alpha_bits = format(self.alpha_string, f'0{self.norb}b')[::-1]
        beta_bits = format(self.beta_string, f'0{self.norb}b')[::-1]
        return f"|{alpha_bits}⟩_α ⊗ |{beta_bits}⟩_β"
