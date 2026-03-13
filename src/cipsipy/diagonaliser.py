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
        self.max_micro_iterations = max_micro_iterations # TODO you should actually use this...
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

        # initial guess for eigenstates: [1,0,0,...], [0,1,0,...], ...
        Vmat = jnp.eye(dim, self.nstate)

        for _ in range(self.max_macro_iterations):
            Vmat, _ = jnp.linalg.qr(Vmat)
            Wmat = self._apply_h_vec_prod(H_vec_prod, Vmat) # (dim, m)

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
            residuals = HUmat - Umat * evals[jnp.newaxis, :]

            rnorm = jnp.linalg.norm(residuals, axis=0)

            if jnp.all(rnorm < self.residual_tol):
                return evals, Umat

            # Safe-guard for small denominators
            eps = 1e-12
            safe_denom = jnp.where(jnp.abs(denom) < eps, jnp.sign(denom) * eps, denom)
            corrections = residuals / safe_denom

            Vmat = jnp.hstack((Vmat, corrections))

        print("Exiting Davidson diagonalisation due to max iterations reached")
        print("Use resulting eigenvectors at your own peril. :)")
        return evals, Umat

    @staticmethod
    def _apply_h_vec_prod(H_vec_prod, vectors):
        """Apply a vector-only H_vec_prod to one or many vectors.

        Args:
            H_vec_prod: Callable mapping shape (dim,) -> (dim,)
            vectors: Either shape (dim,) or (dim, nvec)
        """
        if vectors.ndim == 1:
            return H_vec_prod(vectors)

        cols = [H_vec_prod(vectors[:, i]) for i in range(vectors.shape[1])]
        return jnp.column_stack(cols)
