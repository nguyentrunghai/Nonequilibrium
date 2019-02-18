"""
Define functions to estimate free energy differences and PMFs
"""

import numpy as np

from estimators import uni_df_t, uni_pmf
from estimators import bi_df_t, bi_pmf
from estimators import sym_est_df_t_v1, sym_est_pmf_v1
from estimators import sym_est_df_t_v2, sym_est_pmf_v2


def unidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseires_indices,
                      nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param timeseires_indices: list of 1d ndarray of int
    :param nbootstraps: int, number of bootstrap samples

    :return: free_energies, dict
    """
    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested  > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["timeseires_indices"] = timeseires_indices
    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]

    free_energies["lambdas"] = pulling_data["lambda_F"][timeseires_indices]
    w_t = pulling_data["wF_t"][:total_ntrajs_requested, timeseires_indices]

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        free_energies["main_estimates"]["block_%d"%block] = uni_df_t(w_t[left_bound : right_bound])

    for bootstrap in range(nbootstraps):
        free_energies["bootstrap_%d"%bootstrap] = {}
        for block in range(nblocks):
            traj_indices = np.random.choice(total_ntrajs_requested, size=ntrajs_per_block, replace=True)
            free_energies["bootstrap_%d" % bootstrap]["block_%d"%block] = uni_df_t(w_t[traj_indices])

    return free_energies

