from cipsipy.determinants import Wavefunction, clear_orbital_bit
from cipsipy.hamiltonian import Hamiltonian
from typing import Tuple
from cipsipy.fcidump import read_fcidump
from cipsipy.determinants import is_orbital_occupied
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
        N_gen, N_sel = get_det_subset_size(self.wfn.coeffs, self.n_g, self.n_s)

        # get slice views
        det_alpha_gen = self.wfn.dets_alpha[:N_gen]
        # also for beta, coeffs
        # also for selectors

        dets_ext = []
        epsilon_ext = []
        # TODO instead of looping over spinorbitals, loop over spatorbs for each spin case (aa, bb,ab)
        #   this should be more efficient and you only need to store NxN for Pmat instead of (2N)x(2N)
        # TODO or maybe not? See algorithm 14 of thesis
        # loop generators
        # first loop over spin orbitals
        for p_so in range(2*self.ham.norb):
            for q_so in range(p_so+1, 2*self.ham.norb):
                p = spinorb2spatorb(p_so, self.ham.norb) # TODO implement
                # TODO might also want to a converter for (det_alpha,det_beta) <-> det_spinorb
                q = spinorb2spatorb(q_so, self.ham.norb)
                for i in range(N_gen):
                    Gdet_alpha = det_alpha_gen[i]
                    # same for beta
                    # coeff
                    # make sure orbitals are occupied
                    if not is_spinorb_occupied(Gdet_alpha, Gdet_beta, p_so, q_so): # TODO implement
                        continue # a_P a_Q G = 0

                    # create masks (tagging): impose physicality + uniqueness
                    physicality_mask = get_physical_mask(Gdet_alpha, Gdet_beta, p_so, q_so) # TODO implement

                    # selector loop -- calculate P_rs(G_pq) perturbation matrix
                    # internally this function will do extra masking as well
                    Pmat, tagmask = compute_Pmat_batch(Gdet_alpha, Gdet_beta, p_so, q_so, # TODO implement -- might need to change arguments
                                              physicality_mask, self.ham.eri, N_sel)
                    det, epsilon = get_external_determinants_weight(Pmat, tagmask, Evar) # TODO implement
                    dets_ext.append(det)
                    epsilon_ext.append(epsilon)
        return dets_ext, epsilon_ext

    def get_det_subset_size(coeffs, n):
        """
        get the number of determinants N such that
        ```math
        \sum_{I\leq N} c_I^2 \leq n
        ```
        """
        exit("STUB")


# full algorithm:
# 1. we have internal dets D_I^(n), diagonalise this subspace for H to get E_var^(n) and coeffs for Ψ^(n)
# 2. for all external dets α, compute e_α (Epstein-Nesbet) <-- the really difficult part
# 3. We have E_PT2^(n) = \sum_α e_α and E_FCI^(n) ≈ E_var^(n) + E_PT2^(n)
# 4. Extract α*, those with the largest contributions and add to internal space
# 5. Go to iteration n+1 or exit based on convergence criteria (numdets, small E_PT2^(n), small ΔE_FCI, ...)

def apply_epv_and_single_tagging(
        Bmat,          # Array of shape (2*norb + 1, 2*norb + 1)
        ps: int,          # spin-orbital index 1
        qs: int,          # spin-orbital index 2
        Gdet: int,          # bitstring of original generator
        norb: int,       # number of spatial orbitals
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
    G_pq = clear_orbital_bit(clear_orbital_bit(Gdet, ps), qs)
    is_p_alpha = ps < norb
    is_q_alpha = ps < norb
    # initialise entirely untagged (True for us)
    Bmat = jnp.ones((2*norb, 2*norb), dtype=bool)
    # to tag:
    # - diagonal of G_pq
    # - occupied in G_pq
    # - p and q
    # to untag:
    # q is alpha and p is lowest occupied beta in G (not G_pq) -> untag p
    # p is beta and q is lowest occupied alpha in G (not G_pq) -> untag q
    rows_to_tag = jnp.ones((2*norb,), dtype=bool)
    for rs in range(norb):
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
        for rs in range(norb):
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
