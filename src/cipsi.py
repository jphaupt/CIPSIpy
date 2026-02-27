from cipsipy.determinants import Wavefunction
from cipsipy.hamiltonian import Hamiltonian
from typing import Tuple
from cipsipy.fcidump import read_fcidump

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

        # TODO always make sure determinants/coeffs are sorted s.t. c_I^2 >= C_{I+1}^2
        #   sort at the start of every cipsi iteration?
