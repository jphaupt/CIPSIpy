"""
Tests for Determinant bitstring representation.
"""

from scipy_jax.determinant import Determinant


def test_from_occupation():
    """Test creating determinant from occupation lists."""
    det = Determinant.from_occupation([0, 1], [0, 1], 4)
    assert det.alpha_string == 0b11
    assert det.beta_string == 0b11
    assert det.norb == 4


def test_to_occupation():
    """Test converting determinant to occupation lists."""
    det = Determinant(0b101, 0b110, 4)
    alpha_occ, beta_occ = det.to_occupation()
    assert alpha_occ == [0, 2]
    assert beta_occ == [1, 2]


def test_count_electrons():
    """Test counting electrons."""
    det = Determinant(0b11, 0b101, 4)
    assert det.count_alpha() == 2
    assert det.count_beta() == 2
    assert det.count_electrons() == 4


def test_is_occupied():
    """Test checking orbital occupation."""
    det = Determinant(0b101, 0b110, 4)
    
    # Alpha occupation
    assert det.is_alpha_occupied(0) == True
    assert det.is_alpha_occupied(1) == False
    assert det.is_alpha_occupied(2) == True
    
    # Beta occupation
    assert det.is_beta_occupied(0) == False
    assert det.is_beta_occupied(1) == True
    assert det.is_beta_occupied(2) == True


def test_excitation():
    """Test single excitation."""
    det = Determinant(0b11, 0b11, 4)  # Orbitals 0,1 occupied
    
    # Alpha excitation from 0 to 2
    new_det = det.excitation(0, 2, 'alpha')
    alpha_occ, _ = new_det.to_occupation()
    assert alpha_occ == [1, 2]
    
    # Beta excitation from 1 to 3
    new_det = det.excitation(1, 3, 'beta')
    _, beta_occ = new_det.to_occupation()
    assert beta_occ == [0, 3]


def test_excitation_level():
    """Test calculating excitation level between determinants."""
    det1 = Determinant(0b11, 0b11, 4)     # alpha: [0,1], beta: [0,1]
    det2 = Determinant(0b101, 0b11, 4)    # alpha: [0,2], beta: [0,1]
    det3 = Determinant(0b1001, 0b110, 4)  # alpha: [0,3], beta: [1,2]
    
    # Single excitation in alpha
    alpha_exc, beta_exc = det1.excitation_level(det2)
    assert alpha_exc == 1
    assert beta_exc == 0
    
    # Double excitation (one in each spin)
    alpha_exc, beta_exc = det1.excitation_level(det3)
    assert alpha_exc == 1
    assert beta_exc == 1


def test_equality():
    """Test determinant equality."""
    det1 = Determinant(0b11, 0b11, 4)
    det2 = Determinant(0b11, 0b11, 4)
    det3 = Determinant(0b101, 0b11, 4)
    
    assert det1 == det2
    assert det1 != det3


def test_hash():
    """Test determinant hashing."""
    det1 = Determinant(0b11, 0b11, 4)
    det2 = Determinant(0b11, 0b11, 4)
    det3 = Determinant(0b101, 0b11, 4)
    
    assert hash(det1) == hash(det2)
    assert hash(det1) != hash(det3)
    
    # Can be used in sets/dicts
    det_set = {det1, det2, det3}
    assert len(det_set) == 2


def test_repr():
    """Test string representation."""
    det = Determinant(0b11, 0b101, 4)
    repr_str = repr(det)
    assert 'Determinant' in repr_str
    assert 'alpha=[0, 1]' in repr_str
    assert 'beta=[0, 2]' in repr_str


def test_str():
    """Test human-readable string."""
    det = Determinant(0b11, 0b101, 4)
    str_repr = str(det)
    assert '1100' in str_repr or '0011' in str_repr
    assert '1010' in str_repr or '0101' in str_repr


def test_roundtrip():
    """Test that from_occupation and to_occupation are inverses."""
    alpha_occ = [0, 2, 3]
    beta_occ = [1, 2]
    
    det = Determinant.from_occupation(alpha_occ, beta_occ, 5)
    recovered_alpha, recovered_beta = det.to_occupation()
    
    assert recovered_alpha == alpha_occ
    assert recovered_beta == beta_occ


if __name__ == '__main__':
    test_from_occupation()
    test_to_occupation()
    test_count_electrons()
    test_is_occupied()
    test_excitation()
    test_excitation_level()
    test_equality()
    test_hash()
    test_repr()
    test_str()
    test_roundtrip()
    print("All Determinant tests passed!")
