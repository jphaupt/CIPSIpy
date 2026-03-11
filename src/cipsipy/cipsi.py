from cipsipy.determinants import Wavefunction, clear_orbital_bit, get_det_subset_size
from cipsipy.hamiltonian import Hamiltonian
from typing import Tuple, Optional
from cipsipy.fcidump import read_fcidump
from cipsipy.determinants import is_orbital_occupied, spatorb2spinorb_det, annihilate
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

        nelec_tot, norb, ms2, h_core, eri, e_nuc = read_fcidump(fcidump_filename)
        n_alpha = (nelec_tot + ms2) // 2
        n_beta = (nelec_tot - ms2) // 2
        if n_alpha + n_beta != nelec_tot:
            raise ValueError(f"ms2={ms2} and nelec={nelec_tot} inconsistent.")
        self.nelec = (n_alpha, n_beta)

        HF_alpha = (1 << n_alpha) - 1 # e.g. if n_alpha = 3, 1 << 3 is 0b1000, subtract 1 to get 0b111
        HF_beta = (1 << n_beta) - 1

        self.wfn = Wavefunction([1.0], [HF_alpha], [HF_beta], norb)
        self.ham = Hamiltonian(norb, h_core, eri, e_nuc)

        self.n_g = n_g
        self.N_gen = 1 # number of generators
        self.n_s = n_s
        self.N_sel = 1 # number of selectors
        # N_det = len(self.wfn.coeffs)

        # TODO always make sure determinants/coeffs are sorted s.t. c_I^2 >= C_{I+1}^2
        #   sort at the start of every cipsi iteration?

    def run_unfiltered_selection(self, Evar):
        # loop over generators G
        # loop over batch (nonzero doubly-ionised generator G_pq)
        # Init tagging/mask array and Pmat
        # EPV and single-excit tagging
        # loop over selectors
        # more tagging
        # calculate e_α for untagged |α⟩=|G_pq^rs⟩
        # Evar is the current iteration's variational energy
        exit("STUB with outline")
        # outline:
        self.wfn = self.wfn.coeff_sorted()
        N_gen, N_sel = get_det_subset_size(self.wfn.coeffs, self.n_g, self.n_s)

        # get slice views
        det_alpha_gen = self.wfn.dets_alpha[:N_gen]
        det_beta_gen = self.wfn.dets_beta[:N_gen]
        # also for beta, coeffs
        # also for selectors

        dets_ext = []
        epsilon_ext = []
        # TODO instead of looping over spinorbitals, loop over spatorbs for each spin case (aa, bb,ab)
        #   this should be more efficient and you only need to store NxN for Pmat instead of (2N)x(2N)
        # TODO or maybe not? See algorithm 14 of thesis
        # loop generators
        # first loop over spin orbitals
        for ps in range(2*self.ham.norb):
            for qs in range(ps+1, 2*self.ham.norb):
                for i_gen in range(N_gen):
                    Gdet_alpha = det_alpha_gen[i_gen]
                    Gdet_beta  = det_beta_gen[i_gen]
                    Gdet = spatorb2spinorb_det(Gdet_alpha, Gdet_beta, self.ham.norb)
                    # same for beta
                    # coeff
                    # make sure orbitals are occupied
                    G_pq = annihilate(annihilate(Gdet, ps), qs)
                    if not G_pq == 0:
                        continue # a_P a_Q G = 0

                    # create masks (tagging): impose physicality + uniqueness
                    tagmask = apply_epv_and_single_tagging(ps, qs, Gdet, G_pq, self.ham.norb)
                    for j_sel in range(N_sel): # TODO this loop can surely be replaced with matrix manipulation
                        Sdet_alpha = det_alpha_gen[j_sel]
                        Sdet_beta  = det_beta_gen[j_sel]
                        Sdet = spatorb2spinorb_det(Sdet_alpha, Sdet_beta, self.ham.norb)
                        # Check if there exists (r,s) such that G_pq^rs = S
                        r_s_pair = get_creation_pair(G_pq, Sdet, self.ham.norb)
                        if r_s_pair is not None:
                            tagmask = tagmask.at[r_s_pair[0], r_s_pair[1]].set(False)
                            tagmask = tagmask.at[r_s_pair[1], r_s_pair[0]].set(False)
                        # if ∃ (r,s) s.t. S=G_pq^rs, tag

                        #

                    # selector loop -- calculate P_rs(G_pq) perturbation matrix
                    # internally this function will do extra masking as well
                    # TODO really really hard to work out unit tests for this
                    # TODO maybe split this into even more loops -- horrible for performance but we can optimise after we know we can recreate FCI
                    # TODO just create a function for each steps 6-11 in the thesis? Should be doable and easier to test/implement with TDD, though horribly slow

                    Pmat, tagmask = compute_Pmat_batch(Gdet_alpha, Gdet_beta, ps, qs, # TODO implement -- might need to change arguments
                                              tagmask, self.ham.eri, N_sel)
                    det, epsilon = get_external_determinants_weight(Pmat, tagmask, Evar) # TODO implement
                    dets_ext.append(det)
                    epsilon_ext.append(epsilon)
        return dets_ext, epsilon_ext


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
    # initialise entirely untagged (True)
    Bmat = jnp.ones((2*norb, 2*norb), dtype=bool)
    # to tag:
    # - diagonal of G_pq
    # - occupied in G_pq
    # - p and q
    # to untag:
    # q is alpha and p is lowest occupied beta in G (not G_pq) -> untag p
    # p is beta and q is lowest occupied alpha in G (not G_pq) -> untag q
    rows_to_tag = jnp.ones((2*norb,), dtype=bool)
    for rs in range(2*norb):
        if is_orbital_occupied(G_pq, rs):
            rows_to_tag = rows_to_tag.at[rs].set(False)
    rows_to_tag = rows_to_tag.at[ps].set(False)
    rows_to_tag = rows_to_tag.at[qs].set(False)
    if is_q_alpha:
        for rs in range(norb, 2*norb): # beta orbitals
            if is_orbital_occupied(Gdet, rs):
                if rs == ps:
                    rows_to_tag = rows_to_tag.at[ps].set(True)
                else:
                    break
    if not is_p_alpha:
        for rs in range(norb): # 0 to norb-1 are alpha orbitals
            if is_orbital_occupied(Gdet, rs):
                if rs == qs:
                    rows_to_tag = rows_to_tag.at[qs].set(True)
                else:
                    break
    # now do the actual tagging
    for i in range(2*norb):
        Bmat = Bmat.at[i, i].set(False)
        if not rows_to_tag[i]:
            Bmat = Bmat.at[i, :].set(False)
            Bmat = Bmat.at[:, i].set(False)

    return Bmat
