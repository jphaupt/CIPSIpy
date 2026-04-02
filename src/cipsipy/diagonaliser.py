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
                 max_macro_iterations: int = 50,
                 max_subspace: int = None) -> None:
        """
        H_diag: diagonal of the matrix/Hamiltonian (1D array)
        nstate: number of states to solve for
        residual_tol: convergence tolerance
        max_micro_iterations: maximum number of micro (inner) iterations
        max_macro_iterations: maximum number of macro (outer) iterations
        max_subspace: maximum number of subspace vectors before a thick restart.
            Acts as the padding size for the preallocated subspace buffers, so
            the shape seen by H_vec_prod is fixed across all iterations and JAX
            only compiles it once.  Defaults to
            ``(max_micro_iterations + 1) * nstate``, capped at ``dim``.
        """
        self.nstate = nstate
        self.residual_tol = residual_tol
        self.max_micro_iterations = max_micro_iterations
        self.max_macro_iterations = max_macro_iterations
        if self.nstate < 1:
            raise ValueError("Cannot have fewer than one state to solve.")
        self.H_diag = H_diag
        dim = H_diag.shape[0]
        if max_subspace is None:
            max_subspace = (max_micro_iterations + 1) * nstate
        # Ensure at least two blocks of nstate vectors and no larger than dim
        self.max_subspace = min(max(max_subspace, 2 * nstate), dim)

    def davidson(self, H_vec_prod, initial_guess=None):
        """Solve for the lowest ``nstate`` eigenvalues/vectors via Davidson.

        Args:
            H_vec_prod: Callable mapping a 1-D vector of shape ``(dim,)`` to
                ``H @ v`` of the same shape.
            initial_guess: unused (reserved for future use).

        The subspace is stored in two preallocated buffers of fixed shape
        ``(dim, max_subspace)``.  At each macro-iteration only the *new*
        correction vectors (at most ``nstate`` columns) are passed to
        ``H_vec_prod``, so that callable sees at most two distinct input shapes
        and is compiled at most twice by JAX rather than once per iteration as
        with the original growing-hstack approach.  When the buffer is full a
        *thick restart* resets the subspace to the current Ritz vectors,
        keeping the buffers' shape constant throughout.
        """
        dim = self.H_diag.shape[0]
        nstate = self.nstate
        max_subspace = self.max_subspace

        # Preallocated fixed-shape subspace buffers.  Their shape never changes,
        # so the vmap'd H_vec_prod call sees only a small number of distinct
        # shapes instead of a new shape on every iteration.
        Vmat = jnp.zeros((dim, max_subspace))
        Wmat = jnp.zeros((dim, max_subspace))

        # Seed the first nstate columns with unit vectors (already orthonormal)
        Vmat = Vmat.at[:, :nstate].set(jnp.eye(dim, nstate))
        # Apply H to the initial columns
        Wmat = Wmat.at[:, :nstate].set(
            self._apply_h_vec_prod(H_vec_prod, Vmat[:, :nstate])
        )
        m = nstate  # number of active (filled) columns

        evals = jnp.zeros(nstate)
        Umat = Vmat[:, :nstate]

        for _ in range(self.max_macro_iterations):
            # Subspace solve on the m active columns only
            V_m = Vmat[:, :m]
            W_m = Wmat[:, :m]
            Tmat = V_m.T @ W_m          # (m, m)
            evals_m, Smat = jnp.linalg.eigh(Tmat)
            evals = evals_m[:nstate]
            Smat = Smat[:, :nstate]

            # Ritz vectors and projected H application
            Umat  = V_m @ Smat          # (dim, nstate)
            HUmat = W_m @ Smat          # (dim, nstate)

            denom     = self.H_diag[:, jnp.newaxis] - evals[jnp.newaxis, :]
            residuals = HUmat - Umat * evals[jnp.newaxis, :]
            rnorm     = jnp.linalg.norm(residuals, axis=0)

            if jnp.all(rnorm < self.residual_tol):
                return evals, Umat

            # Safe-guard for small denominators (DPR correction).
            # sign(0.0) == 0 in IEEE 754, so using sign(denom)*eps as the
            # fallback would leave a zero denominator.  Instead, zero out the
            # correction for any component where the denominator is negligible.
            eps = 1e-12
            denom_is_small = jnp.abs(denom) < eps
            safe_denom  = jnp.where(denom_is_small, eps, denom)
            corrections = jnp.where(denom_is_small, 0.0, residuals / safe_denom)

            # Thick restart: when the buffer is full, reset the subspace to the
            # current Ritz vectors so the fixed-size buffers can be reused.
            # The restart is placed *after* the convergence check so the full
            # subspace is always tested before discarding any vectors.
            if m >= max_subspace:
                Vmat = Vmat.at[:, :nstate].set(Umat)
                Wmat = Wmat.at[:, :nstate].set(HUmat)
                Vmat = Vmat.at[:, nstate:].set(0.0)
                Wmat = Wmat.at[:, nstate:].set(0.0)
                m = nstate

            # Orthogonalise correction vectors against the current subspace
            # (modified Gram-Schmidt) so they contribute genuinely new directions.
            new_vecs = corrections
            for k in range(m):
                vk = Vmat[:, k:k + 1]
                new_vecs = new_vecs - vk * (vk.T @ new_vecs)

            # Mutual MGS: orthogonalise the nstate correction columns against
            # each other so they span independent directions.  When two or more
            # DPR corrections point in the same direction (common when nstate>1
            # and the residuals all lie in a single orthogonal complement
            # direction), later columns become near-zero after this step and
            # must be discarded rather than added as rank-deficient columns.
            # NOTE: the `if nj > eps` checks below use concrete JAX scalar
            # comparisons (valid in eager / interpreted mode).  davidson() is
            # intentionally not jax.jit-compiled itself; the performance gain
            # comes from H_vec_prod always seeing a fixed input shape.
            for j in range(nstate - 1):
                nj = jnp.linalg.norm(new_vecs[:, j])
                if nj > eps:
                    vj = new_vecs[:, j:j + 1] / nj
                    new_vecs = new_vecs.at[:, j + 1:].set(
                        new_vecs[:, j + 1:] - vj * (vj.T @ new_vecs[:, j + 1:])
                    )

            # Append only columns with sufficient norm; skip near-zero ones to
            # avoid introducing rank-deficient columns into the subspace.
            m_new = m
            for j in range(nstate):
                if m_new >= max_subspace:
                    break
                nj = jnp.linalg.norm(new_vecs[:, j])
                if nj > eps:
                    Vmat = Vmat.at[:, m_new].set(new_vecs[:, j] / nj)
                    m_new += 1

            # Apply H only to the new columns.  The slice Vmat[:, m:m_new] has
            # at most nstate columns, so H_vec_prod sees at most two distinct
            # shapes across all iterations and is compiled at most twice.
            if m_new > m:
                Wmat = Wmat.at[:, m:m_new].set(
                    self._apply_h_vec_prod(H_vec_prod, Vmat[:, m:m_new])
                )
            m = m_new

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

        # vmap over the column dimension (in_axes=1, out_axes=1)
        return jax.vmap(H_vec_prod, in_axes=1, out_axes=1)(vectors)
