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
        self.H_diag = H_diag

    def davidson(self, H_vec_prod, initial_guess=None):
        """
        H_vec_prod: A callable that takes a vector v and returns Hv
        initial_guess: Optional starting vector (length dim)
        """
        dim = self.H_diag.shape[0]
        energies = jnp.zeros((self.nstate,))
        eigenstates = jnp.zeros((dim, self.nstate,))
        print("STUB")
        # Minimal starting vector for design/testing scaffold.
        v0 = jnp.zeros((dim,))
        v0 = v0.at[0].set(1.0)

        # we pass callables as arguments to prevent needing to pass so many vectors
        # while keeping functional purity
        # In the full Davidson routine this would be used to build subspace data.
        hv0 = H_vec_prod(v0)
        if hv0.shape != (dim,):
            raise ValueError(f"H_vec_prod must return shape ({dim},)")

        # Store one mat-vec result in the first state column to expose data flow.
        eigenstates = eigenstates.at[:, 0].set(hv0)
        # TODO see algorithm 8 of the garniron thesis

        return energies, eigenstates
