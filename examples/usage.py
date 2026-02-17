"""
Example usage of FCIDUMP parser and Determinant bitstrings.
"""

from scipy_jax import FCIDump, Determinant
import os


def example_fcidump_parser():
    """Demonstrate FCIDUMP parsing."""
    print("=" * 60)
    print("FCIDUMP Parser Example")
    print("=" * 60)
    
    # Get the example FCIDUMP file path
    example_dir = os.path.dirname(__file__)
    fcidump_file = os.path.join(example_dir, 'h2_sto3g.fcidump')
    
    # Parse the FCIDUMP file
    fcidump = FCIDump(fcidump_file)
    
    print(f"\nParsed: {fcidump}")
    print(f"Number of orbitals: {fcidump.norb}")
    print(f"Number of electrons: {fcidump.nelec}")
    print(f"Nuclear repulsion energy: {fcidump.nuclear_repulsion:.6f}")
    
    # Get one-electron integrals as a matrix
    h = fcidump.get_one_electron_matrix()
    print(f"\nOne-electron integral matrix shape: {h.shape}")
    print(f"h[0,0] = {h[0,0]:.6f}")
    
    # Get two-electron integrals as a tensor
    v = fcidump.get_two_electron_tensor()
    print(f"Two-electron integral tensor shape: {v.shape}")
    print(f"v[0,0,0,0] = {v[0,0,0,0]:.6f}")


def example_determinant_bitstrings():
    """Demonstrate Determinant bitstring representation."""
    print("\n" + "=" * 60)
    print("Determinant Bitstring Example")
    print("=" * 60)
    
    # Create a determinant from occupation lists
    # For H2 with 4 orbitals, electrons in orbitals 0 and 1
    alpha_occ = [0, 1]
    beta_occ = [0, 1]
    norb = 4
    
    det = Determinant.from_occupation(alpha_occ, beta_occ, norb)
    print(f"\nDeterminant: {det}")
    print(f"String representation: {str(det)}")
    print(f"Alpha bitstring: {bin(det.alpha_string)}")
    print(f"Beta bitstring: {bin(det.beta_string)}")
    
    # Check occupation
    print(f"\nOrbital 0 alpha occupied: {det.is_alpha_occupied(0)}")
    print(f"Orbital 2 alpha occupied: {det.is_alpha_occupied(2)}")
    
    # Count electrons
    print(f"\nTotal electrons: {det.count_electrons()}")
    print(f"Alpha electrons: {det.count_alpha()}")
    print(f"Beta electrons: {det.count_beta()}")
    
    # Create an excited determinant
    print("\n--- Single Excitation ---")
    excited_det = det.excitation(from_orbital=1, to_orbital=2, spin='alpha')
    print(f"Original: {det}")
    print(f"Excited:  {excited_det}")
    
    # Calculate excitation level
    alpha_exc, beta_exc = det.excitation_level(excited_det)
    print(f"Excitation level: {alpha_exc} alpha, {beta_exc} beta")
    
    # Convert back to occupation
    alpha_occ_recovered, beta_occ_recovered = excited_det.to_occupation()
    print(f"Recovered occupation: alpha={alpha_occ_recovered}, beta={beta_occ_recovered}")


def example_combined():
    """Demonstrate combined usage of FCIDUMP and Determinant."""
    print("\n" + "=" * 60)
    print("Combined Example: FCIDUMP + Determinants")
    print("=" * 60)
    
    # Parse FCIDUMP
    example_dir = os.path.dirname(__file__)
    fcidump_file = os.path.join(example_dir, 'h2_sto3g.fcidump')
    fcidump = FCIDump(fcidump_file)
    
    # Create ground state determinant
    nelec = fcidump.nelec
    nalpha = (nelec + fcidump.ms2) // 2
    nbeta = nelec - nalpha
    
    # Fill lowest orbitals
    alpha_occ = list(range(nalpha))
    beta_occ = list(range(nbeta))
    
    ground_state = Determinant.from_occupation(alpha_occ, beta_occ, fcidump.norb)
    print(f"\nGround state determinant: {ground_state}")
    print(f"Ground state: {str(ground_state)}")
    
    # Generate some excited determinants
    print("\n--- Excited Determinants ---")
    excited_states = []
    for i in range(min(nalpha, 2)):
        for a in range(nalpha, min(fcidump.norb, nalpha + 2)):
            exc_det = ground_state.excitation(i, a, 'alpha')
            excited_states.append(exc_det)
            print(f"Excitation {i} -> {a}: {str(exc_det)}")
    
    print(f"\nGenerated {len(excited_states)} excited determinants")


if __name__ == '__main__':
    example_fcidump_parser()
    example_determinant_bitstrings()
    example_combined()
