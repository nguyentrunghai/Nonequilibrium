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
    :param timeseires_indices: list or 1d ndarray of int
    :param nbootstraps: int, number of bootstrap samples
    :return: free_energies, dict
    """
    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested  > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
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
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block
            free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = uni_df_t(
                w_t_bootstrap[left_bound : right_bound])

    return free_energies


def unidirectional_pmf(pulling_data,
                       nblocks, ntrajs_per_block,
                       pmf_bin_edges, V,
                       nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param pmf_bin_edges: ndarray
    :param V: python function, pulling harmonic potential
    :param nblocks: int, number of trajectories per block
    :param ntrajs_per_block: int, number of trajectories per block
    :param nbootstraps: int, number of bootstrap samples
    :return: pmfs, dict
    """
    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    lambda_F = pulling_data["lambda_F"]
    ks = pulling_data["ks"]
    w_t = pulling_data["wF_t"][:total_ntrajs_requested]
    z_t = pulling_data["zF_t"][:total_ntrajs_requested]

    pmfs = {}
    pmfs["nblocks"] = nblocks
    pmfs["ntrajs_per_block"] = ntrajs_per_block
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["ks"] = ks

    pmfs["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        _, pmfs["main_estimates"]["block_%d" % block] = uni_pmf(z_t[left_bound: right_bound],
                                                             w_t[left_bound: right_bound],
                                                             lambda_F, V, ks, pmf_bin_edges)

    for bootstrap in range(nbootstraps):
        pmfs["bootstrap_%d" % bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]
        z_t_bootstrap = z_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block

            _, pmfs["bootstrap_%d" % bootstrap]["block_%d" % block] = uni_pmf(
                z_t_bootstrap[left_bound: right_bound],
                w_t_bootstrap[left_bound: right_bound],
                lambda_F, V, ks, pmf_bin_edges)

    return pmfs


def bidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseires_indices,
                      nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: dict returned by _IO.load_1d_sim_results()
    :param ntrajs_per_block: int, number of trajectories per block
    :param timeseires_indices: list or 1d ndarray of int
    :param nbootstraps: int, number of bootstrap samples
    :return: free_energies, dict
    """