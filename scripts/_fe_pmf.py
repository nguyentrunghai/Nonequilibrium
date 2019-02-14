"""

"""

import numpy as np

from estimators import uni_df_t, uni_pmf
from estimators import bi_df_t, bi_pmf
from estimators import sym_est_df_t_v1, sym_est_pmf_v1
from estimators import sym_est_df_t_v2, sym_est_pmf_v2


def _use_unidirectional(pulling_data, pmf_bin_edges, V, which_data):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param pmf_bin_edges: ndarray
    :param V: harmonic potential function
    :param which_data:  list of str

    :return: (free_energies, pmfs), both are dict
    """
    print("use unidirectional estimators")
    assert which_data in ["f", "r"], "unknown which_data"

    ks = pulling_data["ks"]
    dt = pulling_data["dt"]

    if which_data == "f":
        lambdas = pulling_data["lambda_F"]
        w_t = pulling_data["wF_t"]
        z_t = pulling_data["zF_t"]

    elif which_data == "r":
        lambdas = pulling_data["lambda_R"]
        w_t = pulling_data["wR_t"]
        z_t = pulling_data["zR_t"]

    repeats, nsamples, times = w_t.shape

    free_energies = {}
    free_energies["lambdas"] = lambdas

    if "pulling_times" in pulling_data:
        print("pulling_times is available")
        free_energies["pulling_times"] = pulling_data["pulling_times"]
    else:
        print("pulling_times calculated from dt")
        free_energies["pulling_times"] = np.arange(len(lambdas)) * dt

    free_energies["fes"] = {}
    free_energies["ks"] = ks

    pmfs = {}
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["pmfs"] = {}
    pmfs["ks"] = ks

    for repeat in range(repeats):
        free_energies["fes"][str(repeat)] = uni_df_t(w_t[repeat])

        _, pmfs["pmfs"][str(repeat)] = uni_pmf(z_t[repeat], w_t[repeat], lambdas, V, ks, pmf_bin_edges)

    free_energies["mean"] = np.stack(free_energies["fes"].values()).mean(axis=0)
    free_energies["std"] = np.stack(free_energies["fes"].values()).std(axis=0)

    pmfs["mean"] = np.stack(pmfs["pmfs"].values()).mean(axis=0)
    pmfs["std"] = np.stack(pmfs["pmfs"].values()).std(axis=0)

    return free_energies, pmfs


def _use_bidirectional(pulling_data, pmf_bin_edges, V):
    """
    :param pulling_data: dict returned by _IO.load_1d_sim_results()
    :param pmf_bin_edges: ndarray
    :param V: harmonic potential function
    :return: list of str
    """
    print("use bidirectional estimators")

    ks = pulling_data["ks"]
    dt = pulling_data["dt"]
    lambda_F = pulling_data["lambda_F"]

    wF_t = pulling_data["wF_t"]

    repeats, nsamples, times = wF_t.shape
    half_nsamples = int(nsamples / 2)

    wF_t = wF_t[:, :half_nsamples, :]
    zF_t = pulling_data["zF_t"][:, :half_nsamples, :]

    wR_t = pulling_data["wR_t"][:, :half_nsamples, :]
    zR_t = pulling_data["zR_t"][:, :half_nsamples, :]

    free_energies = {}
    free_energies["lambdas"] = lambda_F

    if "pulling_times" in pulling_data:
        print("pulling_times is available")
        free_energies["pulling_times"] = pulling_data["pulling_times"]
    else:
        print("pulling_times calculated from dt")
        free_energies["pulling_times"] = np.arange(len(lambda_F)) * dt

    free_energies["fes"] = {}
    free_energies["ks"] = ks

    pmfs = {}
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["pmfs"] = {}
    pmfs["ks"] = ks

    for repeat in range(repeats):
        free_energies["fes"][str(repeat)] = bi_df_t(wF_t[repeat], wR_t[repeat])

        _, pmfs["pmfs"][str(repeat)] = bi_pmf(zF_t[repeat], wF_t[repeat], zR_t[repeat], wR_t[repeat],
                                              lambda_F, V, ks, pmf_bin_edges)

    free_energies["mean"] = np.stack(free_energies["fes"].values()).mean(axis=0)
    free_energies["std"] = np.stack(free_energies["fes"].values()).std(axis=0)

    pmfs["mean"] = np.stack(pmfs["pmfs"].values()).mean(axis=0)
    pmfs["std"] = np.stack(pmfs["pmfs"].values()).std(axis=0)

    return free_energies, pmfs


def _use_symmetric(pulling_data, pmf_bin_edges, version, symmetrize_pmf, V):
    """
    version :   int, either 1 or 2
    is_system_symmetric :   bool,
    """
    assert version in [1, 2], "version must be either 1 or 2"

    if version == 1:
        fe_estimator = sym_est_df_t_v1
        pmf_estimator = sym_est_pmf_v1
        print("use symmetric_v1 estimators")
    else:
        fe_estimator = sym_est_df_t_v2
        pmf_estimator = sym_est_pmf_v2
        print("use symmetric_v2 estimators")

    ks = pulling_data["ks"]
    dt = pulling_data["dt"]
    lambda_F = pulling_data["lambda_F"]
    wF_t = pulling_data["wF_t"]
    zF_t = pulling_data["zF_t"]

    repeats, nsamples, times = wF_t.shape

    free_energies = {}
    free_energies["lambdas"] = lambda_F

    if "pulling_times" in pulling_data:
        print("pulling_times is available")
        free_energies["pulling_times"] = pulling_data["pulling_times"]
    else:
        print("pulling_times calculated from dt")
        free_energies["pulling_times"] = np.arange(len(lambda_F)) * dt

    free_energies["fes"] = {}
    free_energies["ks"] = ks

    pmfs = {}
    pmfs["pmf_bin_edges"] = pmf_bin_edges
    pmfs["pmfs"] = {}
    pmfs["ks"] = ks

    for repeat in range(repeats):
        free_energies["fes"][str(repeat)] = fe_estimator(wF_t[repeat])

        _, pmfs["pmfs"][str(repeat)] = pmf_estimator(zF_t[repeat], wF_t[repeat], lambda_F, V, ks,
                                                     pmf_bin_edges, symmetrize_pmf)

    free_energies["mean"] = np.stack(free_energies["fes"].values()).mean(axis=0)
    free_energies["std"] = np.stack(free_energies["fes"].values()).std(axis=0)

    pmfs["mean"] = np.stack(pmfs["pmfs"].values()).mean(axis=0)
    pmfs["std"] = np.stack(pmfs["pmfs"].values()).std(axis=0)

    return free_energies, pmfs


def pull_fe_pmf(which_estimator, pulling_data, pmf_bin_edges, symmetrize_pmf, V):
    assert which_estimator in ["uf", "ur", "b", "s1", "s2"], "which_estimator must be one of ['uf', 'ur', 'b', 's1', 's2']"

    if which_estimator == "uf":
        free_energies, pmfs = _use_unidirectional(pulling_data, pmf_bin_edges, V, "f")

    if which_estimator == "ur":
        free_energies, pmfs = _use_unidirectional(pulling_data, pmf_bin_edges, V, "r")

    elif which_estimator == "b":
        free_energies, pmfs = _use_bidirectional(pulling_data, pmf_bin_edges, V)

    elif which_estimator == "s1":
        free_energies, pmfs = _use_symmetric(pulling_data, pmf_bin_edges, 1, symmetrize_pmf, V)

    elif which_estimator == "s2":
        free_energies, pmfs = _use_symmetric(pulling_data, pmf_bin_edges, 2, symmetrize_pmf, V)

    return free_energies, pmfs


def equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data) - 1, nsamples, dtype=int)
    return data[new_index]
