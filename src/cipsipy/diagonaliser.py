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

        TODO the JAX array gets rebuilt every time we add a new vector guess, so
            this wastes a lot of time -- use padding with masking
        """
        dim = self.H_diag.shape[0]

        # initial guess for eigenstates
        Vmat = jnp.eye(dim, self.nstate)

        for _ in range(self.max_macro_iterations):
            Vmat, _ = jnp.linalg.qr(Vmat)
            Wmat = H_vec_prod(Vmat) # (dim, m)

            # subspace matrix
            Tmat = Vmat.T @ Wmat    # (m, m)
            evals, Smat = jnp.linalg.eigh(Tmat)
            idx = jnp.argsort(evals[:self.nstate])
            evals = evals[idx]
            Smat = Smat[:, idx]

            # get Ritz vectors
            Umat = Vmat @ Smat
            HUmat = Wmat @ Smat
            denom = self.H_diag[:, jnp.newaxis] - evals[jnp.newaxis, :]
            residuals = (HUmat - Umat * evals[jnp.newaxis, :]) / denom

            rnorm = jnp.linalg.norm(residuals, axis=0)

            if jnp.all(rnorm < self.residual_tol):
                return evals, Umat

            Vmat = jnp.hstack((Vmat, residuals))

        print("Exiting Davidson diagonalisation due to max iterations reached")
        return evals, Umat
