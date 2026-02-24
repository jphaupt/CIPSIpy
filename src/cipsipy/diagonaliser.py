import jax
import jax.numpy as jnp
class Diagonaliser:
    """Matrix diagonaliser
    This class will contain methods for diagonlisation routines, such as for FCI
    and certain steps in sCI. These can either be sparse or dense diagonalisations
    jax.numpy.linalg.eigh or the Davidson method (for now assume Hermitian)
    Later may want jax.numpy.linalg.eig and the Arnoldi method
    """
    def __init__(self,
                 H_diag,
                 nstate: int = 1,
                 residual_tol: float = 1.e-10,
                 max_micro_iterations: int = 10,
                 max_macro_iterations: int = 50) -> None:
        """
        H_diag: diagonal of the matrix/Hamiltonian (1D array)
        nstate: number of states to solve for
        residual_tol: convergence tolerance
        max_micro_iterations: maximum number of micro (inner) iterations
        max_macro_iterations: maximum number of macro (outer) iterations
        """
        self.nstate = nstate
        self.residual_tol = residual_tol
        self.max_micro_iterations = max_micro_iterations
        self.max_macro_iterations = max_macro_iterations
        if self.nstate < 1:
            raise ValueError("Cannot have fewer than one state to solve.")
        elif self.nstate != 1:
            raise ValueError("Excited states not yet implemented.")
        self.H_diag = H_diag

    def davidson(self, H_vec_prod):
        """
        H_vec_prod: function that returns the matrix-vector product Hv
        """
        dim = self.H_diag.shape[0]
        v1 = jnp.zeros((dim, dim))
        # TODO see algorithm 8 of the garniron thesis
