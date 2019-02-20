"""
Define functions to estimate free energy differences and PMFs
"""

import numpy as np

from estimators import uni_df_t, uni_pmf
from estimators import bi_df_t, bi_pmf
from estimators import sym_est_df_t_v1, sym_est_pmf_v1
#from estimators import sym_est_df_t_v2, sym_est_pmf_v2


def unidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseries_indices,
                      which_data,
                      nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param timeseries_indices: list or 1d ndarray of int
    :param which_data: str
    :param nbootstraps: int, number of bootstrap samples
    :return: free_energies, dict
    """
    if which_data not in ["F", "R"]:
        raise ValueError("Unknown which_data")

    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    if which_data == "F":
        total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    elif which_data == "R":
        total_ntrajs_in_data = pulling_data["wR_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested  > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]

    if which_data == "F":
        w_t = pulling_data["wF_t"][: total_ntrajs_requested, timeseries_indices]
        lambdas = pulling_data["lambda_F"][timeseries_indices]
    elif which_data == "R":
        w_t = pulling_data["wR_t"][: total_ntrajs_requested, timeseries_indices]
        lambdas = pulling_data["lambda_R"][timeseries_indices]

    free_energies["lambdas"] = lambdas

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
                       which_data,
                       pmf_bin_edges, V,
                       nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param which_data: str
    :param pmf_bin_edges: ndarray, bin edges
    :param V: python function, pulling harmonic potential
    :param nbootstraps: int, number of bootstrap samples
    :return: pmfs, dict
    """
    if which_data not in ["F", "R"]:
        raise ValueError("Unknown which_data")

    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    if which_data == "F":
        total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
    elif which_data == "R":
        total_ntrajs_in_data = pulling_data["wR_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    ks = pulling_data["ks"]

    if which_data == "F":
        lambdas = pulling_data["lambda_F"]
        w_t = pulling_data["wF_t"][: total_ntrajs_requested]
        z_t = pulling_data["zF_t"][: total_ntrajs_requested]
    elif which_data == "R":
        lambdas = pulling_data["lambda_R"]
        w_t = pulling_data["wR_t"][: total_ntrajs_requested]
        z_t = pulling_data["zR_t"][: total_ntrajs_requested]

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
                                                             lambdas, V, ks, pmf_bin_edges)

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
                lambdas, V, ks, pmf_bin_edges)

    return pmfs


def bidirectional_fe(pulling_data, nblocks, ntrajs_per_block,
                    timeseries_indices,
                    nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param timeseries_indices: list or 1d ndarray of int
    :param nbootstraps: int, number of bootstrap samples
    :return: free_energies, dict
    """
    if pulling_data["wF_t"].shape[0] != pulling_data["wR_t"].shape[0]:
        raise ValueError("Forward and reverse must have the same number of trajectories")

    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    total_ntrajs_in_data = pulling_data["wF_t"].shape[0] + pulling_data["wR_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block

    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]
    free_energies["lambdas"] = pulling_data["lambda_F"][timeseries_indices]

    wF_t = pulling_data["wF_t"][: total_ntrajs_requested // 2, timeseries_indices]
    wR_t = pulling_data["wR_t"][: total_ntrajs_requested // 2, timeseries_indices]

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * (ntrajs_per_block // 2)
        right_bound = (block + 1) * (ntrajs_per_block // 2)
        free_energies["main_estimates"]["block_%d" % block] = bi_df_t(wF_t[left_bound : right_bound],
                                                                      wR_t[left_bound : right_bound])

        for bootstrap in range(nbootstraps):
            free_energies["bootstrap_%d" % bootstrap] = {}
            traj_indices = np.random.choice(total_ntrajs_requested // 2,
                                            size=total_ntrajs_requested // 2, replace=True)
            wF_t_bootstrap = wF_t[traj_indices]
            wR_t_bootstrap = wR_t[traj_indices]

            for block in range(nblocks):
                left_bound = block * (ntrajs_per_block // 2)
                right_bound = (block + 1) * (ntrajs_per_block // 2)
                free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = bi_df_t(
                    wF_t_bootstrap[left_bound : right_bound],
                    wR_t_bootstrap[left_bound : right_bound])

    return free_energies


def bidirectional_pmf(pulling_data,
                       nblocks, ntrajs_per_block,
                       pmf_bin_edges, V,
                       nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param pmf_bin_edges: ndarray, bin edges
    :param V: python function, pulling harmonic potential
    :param nbootstraps: int, number of bootstrap samples
    :return: pmfs, dict
    """
    if pulling_data["wF_t"].shape[0] != pulling_data["wR_t"].shape[0]:
        raise ValueError("Forward and reverse must have the same number of trajectories")

    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    total_ntrajs_in_data = pulling_data["wF_t"].shape[0] + pulling_data["wR_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block

    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    ks = pulling_data["ks"]
    lambdas = pulling_data["lambda_F"]

    wF_t = pulling_data["wF_t"][: total_ntrajs_requested // 2]
    zF_t = pulling_data["zF_t"][: total_ntrajs_requested // 2]

    wR_t = pulling_data["wR_t"][: total_ntrajs_requested // 2]
    zR_t = pulling_data["zR_t"][: total_ntrajs_requested // 2]

    pmfs = {}
    pmfs["nblocks"] = nblocks
    pmfs["ntrajs_per_block"] = ntrajs_per_block
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["ks"] = ks

    pmfs["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * (ntrajs_per_block // 2)
        right_bound = (block + 1) * (ntrajs_per_block // 2)
        _, pmfs["main_estimates"]["block_%d" % block] = bi_pmf(zF_t[left_bound : right_bound],
                                                              wF_t[left_bound : right_bound],
                                                              zR_t[left_bound : right_bound],
                                                              wR_t[left_bound : right_bound],
                                                              lambdas, V, ks, pmf_bin_edges)

    for bootstrap in range(nbootstraps):
        pmfs["bootstrap_%d" % bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested // 2,
                                        size=total_ntrajs_requested // 2, replace=True)
        wF_t_bootstrap = wF_t[traj_indices]
        zF_t_bootstrap = zF_t[traj_indices]
        wR_t_bootstrap = wR_t[traj_indices]
        zR_t_bootstrap = zR_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * (ntrajs_per_block // 2)
            right_bound = (block + 1) * (ntrajs_per_block // 2)

            _, pmfs["bootstrap_%d" % bootstrap]["block_%d" % block] = bi_pmf(zF_t_bootstrap[left_bound : right_bound],
                                                                             wF_t_bootstrap[left_bound : right_bound],
                                                                             zR_t_bootstrap[left_bound : right_bound],
                                                                             wR_t_bootstrap[left_bound : right_bound],
                                                                             lambdas, V, ks, pmf_bin_edges)

    return pmfs


def symmetric_fe(pulling_data, nblocks, ntrajs_per_block,
                      timeseries_indices,
                      nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param timeseries_indices: list or 1d ndarray of int
    :param nbootstraps: int, number of bootstrap samples
    :return: free_energies, dict
    """
    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    free_energies = {}
    free_energies["nblocks"] = nblocks
    free_energies["ntrajs_per_block"] = ntrajs_per_block
    free_energies["timeseries_indices"] = timeseries_indices

    free_energies["ks"] = pulling_data["ks"]
    free_energies["dt"] = pulling_data["dt"]

    w_t = pulling_data["wF_t"][: total_ntrajs_requested, timeseries_indices]
    lambdas = pulling_data["lambda_F"][timeseries_indices]
    free_energies["lambdas"] = lambdas

    free_energies["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        free_energies["main_estimates"]["block_%d" % block] = sym_est_df_t_v1(w_t[left_bound: right_bound])

    for bootstrap in range(nbootstraps):
        free_energies["bootstrap_%d"%bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block
            free_energies["bootstrap_%d" % bootstrap]["block_%d" % block] = sym_est_df_t_v1(
                w_t_bootstrap[left_bound : right_bound])

    return free_energies


def symmetric_pmf(pulling_data,
                  nblocks, ntrajs_per_block,
                  pmf_bin_edges, V,
                  symmetrize_pmf,
                  nbootstraps=0):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param nblocks: int, number of blocks of trajectories
    :param ntrajs_per_block: int, number of trajectories per block
    :param pmf_bin_edges: ndarray, bin edges
    :param V: python function, pulling harmonic potential
    :param symmetrize_pmf: bool
    :param nbootstraps: int, number of bootstrap samples
    :return: pmfs, dict
    """
    if ntrajs_per_block % 2 != 0:
        raise ValueError("Number of trajs per block must be even")

    total_ntrajs_in_data = pulling_data["wF_t"].shape[0]

    total_ntrajs_requested = nblocks * ntrajs_per_block
    if total_ntrajs_requested > total_ntrajs_in_data:
        raise ValueError("Number of trajs requested is too large")

    ks = pulling_data["ks"]
    lambdas = pulling_data["lambda_F"]
    w_t = pulling_data["wF_t"][: total_ntrajs_requested]
    z_t = pulling_data["zF_t"][: total_ntrajs_requested]

    pmfs = {}
    pmfs["nblocks"] = nblocks
    pmfs["ntrajs_per_block"] = ntrajs_per_block
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["ks"] = ks

    pmfs["main_estimates"] = {}
    for block in range(nblocks):
        left_bound = block * ntrajs_per_block
        right_bound = (block + 1) * ntrajs_per_block
        _, pmfs["main_estimates"]["block_%d" % block] = sym_est_pmf_v1(z_t[left_bound: right_bound],
                                                                       w_t[left_bound: right_bound],
                                                                       lambdas, V, ks, pmf_bin_edges,
                                                                       symmetrize_pmf)

    for bootstrap in range(nbootstraps):
        pmfs["bootstrap_%d" % bootstrap] = {}
        traj_indices = np.random.choice(total_ntrajs_requested, size=total_ntrajs_requested, replace=True)
        w_t_bootstrap = w_t[traj_indices]
        z_t_bootstrap = z_t[traj_indices]

        for block in range(nblocks):
            left_bound = block * ntrajs_per_block
            right_bound = (block + 1) * ntrajs_per_block

            _, pmfs["bootstrap_%d" % bootstrap]["block_%d" % block] = sym_est_pmf_v1(
                z_t_bootstrap[left_bound: right_bound],
                w_t_bootstrap[left_bound: right_bound],
                lambdas, V, ks, pmf_bin_edges,
                symmetrize_pmf)

    return pmfs
