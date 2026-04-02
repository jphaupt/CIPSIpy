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

        The subspace grows by at most ``nstate`` columns per macro-iteration.
        At each iteration ``H_vec_prod`` is applied only to the *new* columns
        (at most ``nstate``), so for expensive H the number of H applications
        is O(nstate × n_iters) rather than O(nstate × n_iters²) as in the
        original growing-hstack approach.  The subspace is orthogonalised via a
        single batched matrix projection (not an O(m) Python loop), then at most
        ``nstate`` new columns are appended by concatenation (which only copies
        the new data, unlike the padded-buffer scatter-update approach).  When
        the subspace reaches ``max_subspace`` columns a *thick restart* resets
        it to the ``n_keep = max_subspace - nstate`` best Ritz vectors.
        """
        dim = self.H_diag.shape[0]
        nstate = self.nstate
        max_subspace = self.max_subspace
        # On restart, keep as many Ritz vectors as the buffer allows minus the
        # nstate slots needed for the next batch of corrections.
        n_keep = max_subspace - nstate
        eps = 1e-12

        # Growing subspace: Vmat has shape (dim, m), starting from nstate columns.
        # Using growing concatenation (not preallocated scatter) keeps each
        # iteration's work proportional to the actual subspace size.
        Vmat = jnp.eye(dim, nstate)
        Wmat = self._apply_h_vec_prod(H_vec_prod, Vmat)
        m = nstate  # current number of subspace columns

        evals = jnp.zeros(nstate)
        Umat = Vmat

        for _ in range(self.max_macro_iterations):
            # Subspace solve: Rayleigh–Ritz on the full active subspace
            Tmat = Vmat.T @ Wmat                    # (m, m)
            evals_full, Smat_full = jnp.linalg.eigh(Tmat)
            evals = evals_full[:nstate]
            Smat  = Smat_full[:, :nstate]

            # Ritz vectors and their H images
            Umat  = Vmat @ Smat                     # (dim, nstate)
            HUmat = Wmat @ Smat                     # (dim, nstate)

            denom     = self.H_diag[:, jnp.newaxis] - evals[jnp.newaxis, :]
            residuals = HUmat - Umat * evals[jnp.newaxis, :]
            rnorm     = jnp.linalg.norm(residuals, axis=0)

            if jnp.all(rnorm < self.residual_tol):
                return evals, Umat

            # Safe-guard for small denominators (DPR correction).
            # sign(0.0) == 0 in IEEE 754, so using sign(denom)*eps as the
            # fallback would leave a zero denominator.  Instead, zero out the
            # correction for any component where the denominator is negligible.
            denom_is_small = jnp.abs(denom) < eps
            safe_denom  = jnp.where(denom_is_small, eps, denom)
            corrections = jnp.where(denom_is_small, 0.0, residuals / safe_denom)

            # Thick restart: when the subspace reaches max_subspace columns,
            # reset to the n_keep best Ritz vectors before adding corrections.
            # Placed *after* the convergence check so the full subspace is
            # always tested first.  Keeping n_keep = max_subspace - nstate
            # vectors preserves more context than keeping only nstate.
            if m >= max_subspace:
                Vmat = Vmat @ Smat_full[:, :n_keep]  # (dim, n_keep)
                Wmat = Wmat @ Smat_full[:, :n_keep]  # (dim, n_keep)
                m = n_keep

            # Orthogonalise all correction vectors against the current subspace
            # in a single batched projection (two matrix multiplications).
            # This replaces the original O(m) Python loop that dispatched one
            # small JAX kernel per subspace column.
            new_vecs = corrections - Vmat @ (Vmat.T @ corrections)

            # Append new column(s) via concatenation so each copy is
            # proportional to the data added, not the full padded buffer.
            # The outer loop runs at most nstate times (O(1) in practice).
            # A brief mutual-MGS step guards against near-duplicate directions
            # within the same batch (common when nstate > 1 and both DPR
            # corrections lie in a one-dimensional orthogonal complement).
            m_start = m  # index of the first new column in this batch
            for j in range(nstate):
                if m >= max_subspace:
                    break
                col = new_vecs[:, j]
                # Mutual MGS: project out columns added earlier in this batch
                # (at most nstate-1 steps, so always O(1))
                for k_idx in range(m_start, m):
                    c_k = Vmat[:, k_idx]
                    col = col - c_k * (c_k @ col)
                nj = jnp.linalg.norm(col)
                # For j=0, always add (residuals are non-zero → corrections are
                # non-trivially orthogonal to V_m when not converged).
                # Normalise via jnp.where to avoid a forced Python–JAX sync in
                # the common nstate=1 case; the `and` short-circuit prevents
                # JAX scalar materialisation for j=0.
                # For j>0, a near-zero column means it is parallel to an earlier
                # correction; skip it to prevent a rank-deficient subspace.
                if j > 0 and nj <= eps:
                    continue
                normed = col / jnp.where(nj > eps, nj, 1.0)
                Vmat = jnp.column_stack([Vmat, normed])
                m += 1

            # Apply H only to the newly appended columns.  This slice always has
            # at most nstate columns and in the common nstate=1 case has exactly
            # one column, so H_vec_prod sees at most two distinct input shapes
            # (one for the initial seed, one for all subsequent corrections) and
            # is compiled at most twice rather than once per iteration.
            n_add = m - m_start
            if n_add > 0:
                Wmat = jnp.column_stack([
                    Wmat,
                    self._apply_h_vec_prod(H_vec_prod, Vmat[:, m_start:m]),
                ])

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
