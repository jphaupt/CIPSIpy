from cipsipy.determinants import Wavefunction, clear_orbital_bit, get_det_subset_size
from cipsipy.hamiltonian import Hamiltonian, hamiltonian_vector_product
from cipsipy.diagonaliser import Diagonaliser
from typing import Dict, Iterable, List, Optional, Tuple
from cipsipy.fcidump import read_fcidump
from cipsipy.determinants import is_orbital_occupied, spatorb2spinorb_det, \
    get_creation_pair, get_creation_pairs, create, spinorb2spatorb_det, get_occupied_indices
import jax.numpy as jnp

class CIPSISolver:
    def __init__(self, n_g=0.99, n_s=0.999, fcidump_filename='FCIDUMP'):
        """
        n_g is the weight for choosing the number of generator determinants
        ```math
        \sum_{I\leq N_{gen}} c_I^2 \leq n_g
        ```

        Similarly, n_s gives the number of selectors
        ```math
        \sum_{I\leq N_{sel}} c_I^2 \leq n_s
        ```
        which allows us to approximate
        ```math
        \bra{\Psi}\hat H\ket\alpha \approx \sum_{I\leq N_{sel}} c_I \bra{D_I}\hat H\ket\alpha
        ```
        i.e. it (substantially) reduces the cost of e_α from PT

        TODO target multiple states
        TODO allow non-HF start
        """
        # for now, just get all your information from the FCIDUMP
        # TODO you will want to add optional (type-hinted) keyword arguments so
        #   that you can (a) check inputs/sanity check and (b) target other states
        # nelec: Tuple[int, int], norb,
        # self.nelec = nelec
        # n_alpha, n_beta = nelec

        # TODO we have to reorder according to coeffs every CIPSI iteration to get Ngen and Nsel
        #   isn't this very inefficient? There was a reason we used radix for the det sort

        self.fcidump_filename = fcidump_filename
        nelec_tot, norb, ms2, h_core, eri, e_nuc = read_fcidump(fcidump_filename)
        n_alpha = (nelec_tot + ms2) // 2
        n_beta = (nelec_tot - ms2) // 2
        if n_alpha + n_beta != nelec_tot:
            raise ValueError(f"ms2={ms2} and nelec={nelec_tot} inconsistent.")
        self.nelec = (n_alpha, n_beta)

        HF_alpha = (1 << n_alpha) - 1 # e.g. if n_alpha = 3, 1 << 3 is 0b1000, subtract 1 to get 0b111
        HF_beta = (1 << n_beta) - 1

        self.wfn = Wavefunction(
            coeffs=jnp.array([1.0]),
            dets_alpha=jnp.array([HF_alpha]),
            dets_beta=jnp.array([HF_beta]),
            norb=norb,
        )
        self.ham = Hamiltonian(norb, h_core, eri, e_nuc)

        self.n_g = n_g
        self.n_s = n_s

    def _is_target_spin_sector(self, det_alpha: int, det_beta: int) -> bool:
        """Return True if determinant matches the target (n_alpha, n_beta)."""
        return (int(det_alpha).bit_count() == self.nelec[0]) and (
            int(det_beta).bit_count() == self.nelec[1]
        )

    def run_unfiltered_selection(self, Evar):
        # FIXME move all the heavy calculations outside this class so it can be JIT'd
        # Evar is the current iteration's variational energy
        # FIXME there is probably a more intelligent way to go about ensuring the coeffs are sorted,
        # otherwise we would not be using radix sort elsewhere (see thesis sec 4.2)
        self.wfn, _ = self.wfn.coeff_sorted()
        N_gen, N_sel = get_det_subset_size(self.wfn.coeffs, self.n_g, self.n_s)

        # get slice views
        det_alpha_gen = self.wfn.dets_alpha[:N_gen]
        det_beta_gen = self.wfn.dets_beta[:N_gen]
        det_alpha_sel = self.wfn.dets_alpha[:N_sel]
        det_beta_sel = self.wfn.dets_beta[:N_sel]

        dets_ext_alpha = []
        dets_ext_beta  = []
        epsilon_ext = []
        # FIXME instead of looping over spinorbitals, loop over spatorbs for each spin case (aa, bb,ab)
        #   this should be more efficient and you only need to store NxN for Pmat instead of (2N)x(2N)
        #   Even better: use JAX vectorisation wherever possible
        for ps in range(2*self.ham.norb):
            for qs in range(ps+1, 2*self.ham.norb):
                for i_gen in range(N_gen):
                    Gdet_alpha = det_alpha_gen[i_gen]
                    Gdet_beta  = det_beta_gen[i_gen]
                    Gdet = spatorb2spinorb_det(Gdet_alpha, Gdet_beta, self.ham.norb)
                    # We cannot use annihilate's 0 sentinel here because a valid
                    # doubly-ionised determinant can also be represented by 0.
                    if (not is_orbital_occupied(Gdet, ps)) or (not is_orbital_occupied(Gdet, qs)):
                        continue # a_P a_Q G = 0
                    G_pq = clear_orbital_bit(clear_orbital_bit(Gdet, ps), qs)
                    Pmat_pq = jnp.zeros((2*self.ham.norb, 2*self.ham.norb))

                    # create masks (tagging): impose physicality + uniqueness
                    tagmask = apply_epv_and_single_tagging(ps, qs, Gdet, G_pq, self.ham.norb)
                    for j_sel in range(N_sel): # FIXME this loop can surely be replaced with matrix manipulation
                        Sdet_alpha = det_alpha_sel[j_sel]
                        Sdet_beta  = det_beta_sel[j_sel]
                        Sdet = spatorb2spinorb_det(Sdet_alpha, Sdet_beta, self.ham.norb)
                        # if ∃ (r,s) s.t. S=G_pq^rs, tag
                        r_s_pair = get_creation_pair(G_pq, Sdet, self.ham.norb)
                        if r_s_pair is not None:
                            tagmask = tagmask.at[r_s_pair[0], r_s_pair[1]].set(False)
                            tagmask = tagmask.at[r_s_pair[1], r_s_pair[0]].set(False)
                        # get excited pairs rs, ss s.t. G_pq^rs connects to Sdet
                        excited_pairs = get_creation_pairs(G_pq, Sdet, self.ham.norb)
                        if j_sel < i_gen:
                            # this means that G_pq^rs is generated by |D_J⟩
                            for rs, ss in excited_pairs:
                                tagmask = tagmask.at[rs, ss].set(False)
                                tagmask = tagmask.at[ss, rs].set(False)
                        else:
                            # increment all untagged (i.e. where tagmask is True)
                            # P_rs(G_pq) elements by c_J⟨S|H|G_pq^rs⟩
                            # P_rs(G_pq) = Pmat_pq.at[rs,ss]
                            # c_J = self.wfn.coeffs[j_sel]
                            # S = Sdet
                            # G_pq^rs = create(create(G_pq, rs), ss) (for rs, ss in excited_pairs)
                            # ⟨D1|H|D2⟩ = self.ham.element(...)
                            c_J = self.wfn.coeffs[j_sel]
                            for rs, ss in excited_pairs:
                               # TODO this is wildly inefficient and I'll want to implement a JAX-friendly version at some point
                                if not tagmask[rs, ss]:
                                    continue

                                G_pq_rs = create(create(G_pq, rs), ss)
                                if G_pq_rs == 0:
                                    continue

                                G_pq_rs_alpha, G_pq_rs_beta = spinorb2spatorb_det(
                                    G_pq_rs,
                                    self.ham.norb,
                                )
                                if not self._is_target_spin_sector(G_pq_rs_alpha, G_pq_rs_beta):
                                    continue

                                h_elem = self.ham.element(
                                    Sdet_alpha,
                                    Sdet_beta,
                                    G_pq_rs_alpha,
                                    G_pq_rs_beta,
                                )
                                Pmat_pq = Pmat_pq.at[rs, ss].add(c_J * h_elem)
                                Pmat_pq = Pmat_pq.at[ss, rs].add(c_J * h_elem)
                    # all untagged cells (tagmask == True) are associated with a unique det, |G_pq^rs⟩=|α⟩
                    # can now compute
                    # e_α = \frac{P_rs(G_pq)^2}{Evar - ⟨α|H|α⟩}
                    for rs in range(2*self.ham.norb):
                        for ss in range(rs + 1, 2*self.ham.norb):
                            if not tagmask[rs, ss]:
                                continue

                            det_external_spinorb = create(create(G_pq, rs), ss)
                            if det_external_spinorb == 0:
                                continue

                            det_alpha, det_beta = spinorb2spatorb_det(
                                det_external_spinorb,
                                self.ham.norb,
                            )
                            if not self._is_target_spin_sector(det_alpha, det_beta):
                                continue
                            # TODO optimise this with a determinant->diagonal lookup table and
                            # memoisation once the algorithm is validated end-to-end with
                            # integration/system tests against FCI references. For now we keep
                            # this explicit/direct path to minimise moving parts while debugging
                            # correctness.
                            Haa = self.ham.element(
                                det_alpha,
                                det_beta,
                                det_alpha,
                                det_beta,
                            )
                            denom = Evar - Haa
                            if jnp.isclose(denom, 0.0):
                                continue

                            P_rs = Pmat_pq[rs, ss]
                            epsilon = (P_rs * P_rs) / denom

                            dets_ext_alpha.append(det_alpha)
                            dets_ext_beta.append(det_beta)
                            epsilon_ext.append(epsilon)
        return dets_ext_alpha, dets_ext_beta, epsilon_ext

    def _diagonalise_variational_space(self) -> Tuple[float, jnp.ndarray]:
        """Diagonalise current internal space and return (E_var, ground coeffs)."""
        if self.wfn.coeffs.shape[0] == 1:
            e0 = self.ham.element(
                int(self.wfn.dets_alpha[0]),
                int(self.wfn.dets_beta[0]),
                int(self.wfn.dets_alpha[0]),
                int(self.wfn.dets_beta[0]),
            )
            return float(e0), jnp.array([1.0], dtype=self.wfn.coeffs.dtype)

        h_diag = self.ham.diagonal(
            self.wfn.coeffs,
            self.wfn.dets_alpha,
            self.wfn.dets_beta,
        )
        diag = Diagonaliser(H_diag=h_diag, nstate=1)
        evals, vecs = diag.davidson(
            lambda v: hamiltonian_vector_product(
                v,
                self.wfn.dets_alpha,
                self.wfn.dets_beta,
                h_diag,
                self.ham.norb,
                self.ham.h_core,
                self.ham.eri,
            )
        )
        e_var = float(evals[0])
        coeffs = vecs[:, 0]
        norm = jnp.linalg.norm(coeffs)
        if not jnp.isclose(norm, 0.0):
            coeffs = coeffs / norm
        return e_var, coeffs

    @staticmethod
    def _aggregate_external_contributions(
        da_ext: Iterable[int],
        db_ext: Iterable[int],
        epsilon_ext: Iterable[float],
    ) -> Dict[Tuple[int, int], float]:
        """Aggregate PT2 contributions for identical external determinants."""
        contribs: Dict[Tuple[int, int], float] = {}
        for da, db, eps in zip(da_ext, db_ext, epsilon_ext):
            key = (int(da), int(db))
            contribs[key] = contribs.get(key, 0.0) + float(eps)
        return contribs

    def _select_external_determinants(
        self,
        contribs: Dict[Tuple[int, int], float],
        selection_fraction: float,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Select truly external determinants and return their filtered PT2 map."""
        if not contribs:
            return [], {}

        # TODO performance: this set is rebuilt from the full variational
        # space every CIPSI iteration. Maintain it incrementally alongside
        # self.wfn when determinants are added to avoid the repeated O(N_det)
        # rebuild during duplicate filtering.
        current = {
            (int(da), int(db))
            for da, db in zip(self.wfn.dets_alpha, self.wfn.dets_beta)
        }
        external_contribs = {
            (int(da), int(db)): float(eps)
            for (da, db), eps in contribs.items()
            if (int(da), int(db)) not in current
        }
        if not external_contribs:
            return [], {}

        items = sorted(external_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        total_abs = sum(abs(eps) for _, eps in items)
        if jnp.isclose(total_abs, 0.0):
            return [], external_contribs

        target = selection_fraction * total_abs
        cumulative = 0.0
        selected: List[Tuple[int, int]] = []
        for (da, db), eps in items:
            selected.append((da, db))
            cumulative += abs(eps)
            if cumulative >= target:
                break
        return selected, external_contribs

    def _print_final_cipsi_summary(self, Evar_el: float) -> Tuple[float, float]:
        """Print final determinant/energy summary and return (E_var, E_est)."""
        da_ext, db_ext, epsilon_ext = self.run_unfiltered_selection(Evar_el)
        contribs = self._aggregate_external_contributions(da_ext, db_ext, epsilon_ext)
        _, external_contribs = self._select_external_determinants(contribs, selection_fraction=1.0)

        E_var = Evar_el + self.ham.e_nuc
        E_pt2 = float(sum(external_contribs.values()))
        E_est = E_var + E_pt2

        wfn_sorted, _ = self.wfn.coeff_sorted()
        n_det = len(wfn_sorted.coeffs)
        n_show = min(100, n_det)

        print("=" * 72)
        print("Final CIPSI wave function")
        print("=" * 72)
        print(f"N_det(final): {n_det}")
        print("Top determinants (ci alpha_bitstring beta_bitstring):")
        for i in range(n_show):
            ci = float(wfn_sorted.coeffs[i])
            alpha_bits = format(int(wfn_sorted.dets_alpha[i]), f"0{self.ham.norb}b")
            beta_bits = format(int(wfn_sorted.dets_beta[i]), f"0{self.ham.norb}b")
            print(f"{ci: .12e} {alpha_bits} {beta_bits}")

        print("=" * 72)
        print(f"E_var(final):  {E_var: .12f} a.u.")
        print(f"E_PT2(final):  {E_pt2: .12e} a.u.")
        print(f"E_est(final):  {E_est: .12f} a.u.")
        return E_var, E_est

    def run_cipsi(
        self,
        max_iterations: int = 50,
        pt2_tol: float = 1e-8,
        selection_fraction: float = 0.99,
        max_dets: Optional[int] = None,
    ):
        """Run iterative CIPSI determinant growth until convergence or limits."""
        print("Setting up CIPSI calculation")
        print(f"FCIDUMP: {self.fcidump_filename}")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (0.0 < selection_fraction <= 1.0):
            raise ValueError("selection_fraction must be in (0, 1]")

        final_Evar_el: Optional[float] = None
        needs_rediag = False
        stop_reason = "max_iterations reached"

        for iteration in range(max_iterations):
            # 1) Diagonalise the current variational subspace.
            Evar_el, coeffs = self._diagonalise_variational_space()
            self.wfn = self.wfn.with_coeffs(coeffs)
            final_Evar_el = Evar_el
            needs_rediag = False

            # 2) Compute unfiltered external PT2 contributions.
            da_ext, db_ext, epsilon_ext = self.run_unfiltered_selection(Evar_el)
            contribs = self._aggregate_external_contributions(da_ext, db_ext, epsilon_ext)
            selected, external_contribs = self._select_external_determinants(
                contribs,
                selection_fraction,
            )
            Ept2 = float(sum(external_contribs.values()))
            Evar = Evar_el + self.ham.e_nuc
            Efci_est = Evar + Ept2

            print(
                f"iter={iteration:3d}  N_det={len(self.wfn.coeffs):7d}  "
                f"E_var={Evar: .12f}  E_PT2={Ept2: .6e}  E_est={Efci_est: .12f}"
            )

            # 3) Convergence/saturation checks.
            if abs(Ept2) <= pt2_tol:
                stop_reason = f"E_PT2 converged (|E_PT2|={abs(Ept2):.3e} <= tol={pt2_tol:.3e})"
                break

            if not selected:
                stop_reason = "no external determinants selected"
                break
            selected = [
                (da, db)
                for da, db in selected
                if self._is_target_spin_sector(da, db)
            ]
            if not selected:
                stop_reason = "no external determinants in target spin sector"
                break


            if max_dets is not None:
                remaining = max_dets - len(self.wfn.coeffs)
                if remaining <= 0:
                    stop_reason = f"max_dets={max_dets} reached"
                    break
                selected = selected[:remaining]
                if not selected:
                    stop_reason = f"max_dets={max_dets} reached"
                    break

            add_alpha = jnp.array([da for da, _ in selected], dtype=self.wfn.dets_alpha.dtype)
            add_beta = jnp.array([db for _, db in selected], dtype=self.wfn.dets_beta.dtype)
            zeros = jnp.zeros((len(selected),), dtype=self.wfn.coeffs.dtype)

            self.wfn = Wavefunction(
                coeffs=jnp.concatenate((self.wfn.coeffs, zeros)),
                dets_alpha=jnp.concatenate((self.wfn.dets_alpha, add_alpha)),
                dets_beta=jnp.concatenate((self.wfn.dets_beta, add_beta)),
                norb=self.wfn.norb,
            )
            needs_rediag = True

            if max_dets is not None and len(self.wfn.coeffs) >= max_dets:
                # Re-diagonalise once on the final truncated space before returning.
                Evar_el, coeffs = self._diagonalise_variational_space()
                self.wfn = self.wfn.with_coeffs(coeffs)
                final_Evar_el = Evar_el
                needs_rediag = False
                stop_reason = f"max_dets={max_dets} reached"
                break

        print(f"CIPSI converged: {stop_reason}")

        if final_Evar_el is None or needs_rediag:
            Evar_el, coeffs = self._diagonalise_variational_space()
            self.wfn = self.wfn.with_coeffs(coeffs)
            final_Evar_el = Evar_el

        E_var_final, E_est_final = self._print_final_cipsi_summary(final_Evar_el)
        return E_var_final, E_est_final

# full algorithm:
# 1. we have internal dets D_I^(n), diagonalise this subspace for H to get E_var^(n) and coeffs for Ψ^(n)
# 2. for all external dets α, compute e_α (Epstein-Nesbet) <-- the really difficult part
# 3. We have E_PT2^(n) = \sum_α e_α and E_FCI^(n) ≈ E_var^(n) + E_PT2^(n)
# 4. Extract α*, those with the largest contributions and add to internal space
# 5. Go to iteration n+1 or exit based on convergence criteria (numdets, small E_PT2^(n), small ΔE_FCI, ...)

def apply_epv_and_single_tagging(
        ps: int,     # spin-orbital index 1
        qs: int,     # spin-orbital index 2
        Gdet: int,   # bitstring of original generator
        G_pq: int,   # bitstring of Gdet with ps, qs annihilated
        norb: int,   # number of spatial orbitals
    ):
    """
    NOTE: this works in spin-orbitals

    algorithm 15 of the thesis: apply tagging to exclusion-principle-violating
    and single-excitation determinants

    due to untagging logic, the extra row/column in the thesis is NOT just
    for efficiency
    -- otherwise we might untag "zero-body" excitations. We could implement
    this without the extra row/column though, but we'd have to do the actual
    tagging right at the very end
    I think that is what I will do
    """
    is_p_alpha = ps < norb
    is_q_alpha = qs < norb
    n = 2 * norb
    idx = jnp.arange(n)

    # initialise entirely untagged (True)
    Bmat = jnp.ones((n, n), dtype=bool)

    # to tag:
    # - diagonal of G_pq
    # - occupied in G_pq
    # - p and q
    # rows_to_tag[i] = True  ->  row/col i is eligible (virtual, not ps/qs)
    # rows_to_tag[i] = False ->  row/col i is excluded (occupied or ps/qs)
    occupied_mask = get_occupied_indices(G_pq, n).astype(bool)
    rows_to_tag = ~occupied_mask
    rows_to_tag = rows_to_tag.at[ps].set(False)
    rows_to_tag = rows_to_tag.at[qs].set(False)

    # Apply base tagging: zero diagonal, then mask tagged rows/columns.
    Bmat = Bmat.at[idx, idx].set(False)
    Bmat = Bmat & rows_to_tag[:, None] & rows_to_tag[None, :]

    # Single-excitation handling (Garniron 5.4.3):
    # Singles are formally present in G_pq^rs when one created orbital restores
    # one annihilated orbital. We selectively untag exactly those entries so
    # each single excitation is generated once, even with ordered ps < qs loops.
    if is_p_alpha != is_q_alpha:
        if is_p_alpha:
            alpha_ann = ps
            beta_ann = qs
        else:
            alpha_ann = qs
            beta_ann = ps

        alpha_occ = get_occupied_indices(Gdet, norb)
        has_alpha = bool(jnp.any(alpha_occ > 0))
        lowest_occ_alpha = int(jnp.argmax(alpha_occ)) if has_alpha else None

        beta_occ = get_occupied_indices(Gdet, n)[norb:]
        has_beta = bool(jnp.any(beta_occ > 0))
        lowest_occ_beta = int(jnp.argmax(beta_occ) + norb) if has_beta else None

        # Untag alpha-spin singles a -> s exactly once by fixing the created
        # beta orbital to the chosen lowest occupied beta in |G>.
        if (lowest_occ_beta is not None) and (beta_ann == lowest_occ_beta):
            s_range = jnp.arange(norb)
            occ_gpq_alpha = get_occupied_indices(G_pq, norb).astype(bool)
            valid_s = (s_range != alpha_ann) & ~occ_gpq_alpha
            Bmat = Bmat.at[beta_ann, s_range].set(
                jnp.where(valid_s, True, Bmat[beta_ann, s_range])
            )
            Bmat = Bmat.at[s_range, beta_ann].set(
                jnp.where(valid_s, True, Bmat[s_range, beta_ann])
            )

        # Untag beta-spin singles b -> r exactly once by fixing the created
        # alpha orbital to the chosen lowest occupied alpha in |G>.
        if (lowest_occ_alpha is not None) and (alpha_ann == lowest_occ_alpha):
            r_range = jnp.arange(norb, n)
            occ_gpq_beta = get_occupied_indices(G_pq, n)[norb:].astype(bool)
            valid_r = (r_range != beta_ann) & ~occ_gpq_beta
            Bmat = Bmat.at[alpha_ann, r_range].set(
                jnp.where(valid_r, True, Bmat[alpha_ann, r_range])
            )
            Bmat = Bmat.at[r_range, alpha_ann].set(
                jnp.where(valid_r, True, Bmat[r_range, alpha_ann])
            )

    return Bmat
