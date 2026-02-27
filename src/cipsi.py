from cipsipy.determinants import Wavefunction
from cipsipy.hamiltonian import Hamiltonian
from typing import Tuple
from cipsipy.fcidump import read_fcidump

class CIPSISolver:
    def __init__(self, fcidump_filename='FCIDUMP'):
        # for now, just get all your information from the FCIDUMP
        # TODO you will want to add optional (type-hinted) keyword arguments so
        #   that you can (a) check inputs/sanity check and (b) target other states
        # nelec: Tuple[int, int], norb,
        # self.nelec = nelec
        # n_alpha, n_beta = nelec

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
