
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

try:
    import pymbar
except ImportError as e:
    print("pymbar can be obtained from https://github.com/choderalab/pymbar.git")
    raise e


def bennett(w_F, w_R):
    """
    Bennett Acceptance Ratio
    C. Bennett. Efficient Estimation of Free Energy Differences from Monte Carlo Data.
                                    Journal of Computational Physics 22, 245-268 (1976).
    G. Crooks. Path-ensemble averages in systems driven far from equilibrium. Physical Review E 61, 2361-2366 (2000).
    M. Shirts, E. Bair, G. Hooker, and V. Pande.
                    Equilibrium Free Energies from Nonequilibrium Measurements Using Maximum-Likelihood Methods.
                    Physical Review Letters 91, 140601 (2003).

    need pymmbar (https://github.com/choderalab/pymbar):

    :param w_F:   ndarray with shape (NF,)
            works done in forward direction starting from the initial (A) equilibrium ensemble, in unit kT

    :param w_R:   ndarray with shape (NR,)
            works done in forward direction starting from the initial (A) equilibrium ensemble, in unit of kT

    :return:
            df_AB   :   float
                        free energy difference between states A and B (df_AB = f_B - f_A), in unit of kT
    """

    assert w_F.ndim  == w_R.ndim == 1, "w_F, w_R must be 1d arrays"
    df_AB, ddf = pymbar.BAR(w_F, w_R, relative_tolerance=0.000001, verbose=False, compute_uncertainty=True)

    return df_AB


def time_reversal_of_work(w_t):
    """
    :param w_t: ndarray of shape (N, nr_time_steps)
                work done

    :return:    wr_t, ndarray of shape (N, nr_time_steps) time reversal of w_t
    """
    assert w_t.ndim == 2, "w_t must be a 2d array"
    assert np.all(w_t[:, 0] == 0), "all works at t=0 must be zero"

    N = w_t.shape[0]

    dw = w_t[:, 1:] - w_t[:, :-1]
    dw = dw[:, ::-1]                # reverse order
    dw = -dw
    wr_t = np.append(np.zeros([N, 1]), np.cumsum(dw, axis=1), axis=1)
    return wr_t


def time_reversal_of_trajectory(z_t):
    """
    :param z_t: ndarray of shape (N, nr_time_steps)
                None-equilibrium trajectory of the pulling coordinate

    :return: zR_t, ndarray of shape (N, nr_time_steps) time reversal of z_t
    """
    assert z_t.ndim == 2, "w_t must be a 2d array"
    return z_t[:, ::-1]


def hist_counts(data, bin_edges, weights):
    """
    :param data:    ndarray with shape (N, number_of_steps)

    :param bin_edges:   ndarray of shape ( nbins+1 )

    :param weights: ndarray of shape (N, number_of_steps), unnormalized weights

    :return: histograms, np.array of shape (number_of_steps, nbins)
    """
    assert data.ndim == 2, "data must be 2d array"
    assert weights.ndim == 2, "weights must be 2d array"
    assert bin_edges.ndim == 1, "bin_edges must be 1d array"
    assert data.shape == weights.shape, "data and weights must have the same shape"

    times = data.shape[1]
    nbins = bin_edges.shape[0] - 1

    histograms = np.zeros([times, nbins], dtype=float)
    for time in range(times):
        histograms[time, :], e = np.histogram(data[:, time], bins=bin_edges, weights=weights[:, time], density=False)

    return histograms


def bin_centers(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2.


def center_reflection(value, center):
    return -value + 2 * center


def get_bin_indexes(data, bin_edges):
    """
    :param data:    ndarray, float, any shape

    :param bin_edges: ndarray, float, shape (nbins+1), the bin edges

    :return: (bin_indexes, _bin_edges)
        bin_indexes : ndarray, int, same shape as data
        _bin_edges : ndarray, float, shape = (nbins+1, )
    """
    assert isinstance(bin_edges, np.ndarray), "bin_edges must be a ndarray"

    cats, _bin_edges = pd.cut(data.ravel(), bin_edges, retbins=True)
    bin_indexes = cats.codes.reshape(data.shape)

    return bin_indexes, _bin_edges


def equal_spaced_bins(list_of_data, nbins, symmetric_center=None):
    """
    :param list_of_data: list of np.ndarray or nc._netCDF4.Variable
    :param nbins:   int, number of bins

    :param symmetric_center: None or float

    :return: bin_edges, ndarray, float, shape = (nbins+1, )
    """
    assert isinstance(list_of_data, list), "list_of_data must be a list"
    if symmetric_center is not None:
        assert nbins % 2 == 0, "When symmetric_center is not None, nbins must be even"
        # which mean that the symmetric center is the bin edge right in the middle

    mins = []
    maxs = []
    stds = []

    for data in list_of_data:
        load_data = data[:]
        mins.append(load_data.min())
        maxs.append(load_data.max())
        stds.append(load_data.std())

    min_x = np.min(mins)
    max_x = np.max(maxs)
    std_x = np.min(stds)

    lower = min_x - 0.0000001 * std_x
    upper = max_x + 0.0000001 * std_x

    if symmetric_center is not None:
        assert lower < symmetric_center < upper, "symmetric_center is not in between [min, max]"

        left_interval = symmetric_center - lower
        right_interval = upper - symmetric_center

        interval = np.max([left_interval, right_interval])

        lower = symmetric_center - interval
        upper = symmetric_center + interval

    bin_edges = np.linspace(lower, upper, nbins + 1)

    return bin_edges


def equal_sample_bins(list_of_data, nbins):
    """
    :param list_of_data: list of np.ndarray or nc._netCDF4.Variable
    :param nbins: int, number of bins

    :return: bin_edges, ndarray, float, shape = (nbins+1, )
    """
    assert isinstance(list_of_data, list), "list_of_data must be a list"

    all_data = np.concatenate([data[:].ravel() for data in list_of_data])
    percents = np.linspace(0, 100., nbins + 1)
    bin_edges = np.percentile(all_data, percents)

    std_x = all_data.std()

    bin_edges[0] = bin_edges[0] - 0.00001 * std_x
    bin_edges[-1] = bin_edges[-1] + 0.00001 * std_x
    return bin_edges


def right_wrap(z, symm_center):
    """
    :param z: ndarray
    :param symm_center: float
    :return: new_z, ndarray
    """
    new_z = np.copy(z)
    where_to_apply = (new_z > symm_center)
    new_z[where_to_apply] = 2*symm_center - new_z[where_to_apply]
    return new_z


def left_wrap(z, symm_center):
    """
    :param z: ndarray
    :param symm_center: float
    :return: z, ndarray
    """
    new_z = np.copy(z)
    where_to_apply = (new_z < symm_center)
    new_z[where_to_apply] = 2*symm_center - new_z[where_to_apply]
    return new_z


def right_replicate_fe(first_half):
    """
    In : np.array([-5, -4, -3, -2, -1, 0])
    Out: array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    :param first_half:
    :return:
    """
    center = first_half[-1]
    second_half = 2*center - first_half[:-1]
    second_half = second_half[::-1]
    return np.hstack([first_half, second_half])


def left_replicate_fe(first_half):
    """
    In : array([0, 1, 2, 3, 4, 5])
    Out: array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    :param first_half:
    :return:
    """
    center = first_half[0]
    second_half = 2*center - first_half[1:]
    second_half = second_half[::-1]
    return np.hstack([second_half, first_half])


right_replicate_bin_edges = right_replicate_fe
left_replicate_bin_edges = left_replicate_fe


def right_replicate_pmf(first_half):
    """
    In : np.array([-5, -4, -3, -2, -1, 0])
    Out: array([-5, -4, -3, -2, -1,  0,  0,  1,  2,  3,  4,  5])
    :param first_half:
    :return:
    """
    center = first_half[-1]
    second_half = 2*center - first_half
    second_half = second_half[::-1]
    return np.hstack([first_half, second_half])


def left_replicate_pmf(first_half):
    """
    In : array([0, 1, 2, 3, 4, 5])
    Out: array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
    :param first_half:
    :return:
    """
    center = first_half[0]
    second_half = 2*center - first_half
    second_half = second_half[::-1]
    return np.hstack([second_half, first_half])


def stride_index(lambda_F, lambda_R, n):
    """
    calculate indices_F and indices_R such that lambda_F[indices_F] == lambda_R[indices_R][::-1]
    or lambda_F[indices_F][::-1] == lambda_R[indices_R]

    :param lambda_F: 1d ndarray of float
    :param lambda_R: 1d ndarray of float
    :param n: int
    :return (indices_F, indices_R): each is 1d ndarray of int
    """
    indices_F = np.linspace(0, lambda_F.shape[0] - 1, n)
    indices_F = np.round(indices_F)
    indices_F = indices_F.astype(np.int)

    indices_R = lambda_R.shape[0] - 1 - indices_F
    indices_R = indices_R[::-1]

    if not np.allclose(lambda_F[indices_F], lambda_R[indices_R][::-1]):
        raise IndexError("The condition lambda_F[indices_F] == lambda_R[indices_R][::-1] is not satisfied.")
    return indices_F, indices_R


