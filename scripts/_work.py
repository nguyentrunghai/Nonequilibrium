"""
"""
from __future__ import print_function

import numpy as np
import netCDF4 as nc

from models_1d import V_MMTK

TEMPERATURE = 300.
KB = 0.0083144621  # kJ/mol
BETA = 1./KB/TEMPERATURE


def _work_integrate(lambda_t, z_t, k):
    """
    lambda_t    :   np.array of shape (times), restrained center
    z_t         :   np.array of shape (times), restrained coordinate
    k           :   float, harmonic force constant
    """
    assert lambda_t.ndim == z_t.ndim == 1, "lambda_t and z_t must be 1d"
    assert lambda_t.shape == z_t.shape, "lambda_t and z_t must have the same shape"

    times = lambda_t.shape[0]
    w_t = np.zeros(times, dtype=float)

    # eq (3.4) in Christopher Jarzynski, Prog. Theor. Phys. Suppl 2006, 165, 1
    # 2 because MMTK use k*(x-l)**2 (without a factor 1/2)
    #w_t[1:] = - 2 * k * np.cumsum( (z_t[:-1] - lambda_t[:-1]) * (lambda_t[1:] - lambda_t[:-1]) )

    # in D. Minh and J. Chodera J Chem Phys 2009, 131, 134110. in Potential of mean force section
    # this is consistent with 1d model
    w_t[1:] = V_MMTK(z_t[1:], k, lambda_t[1:]) - V_MMTK(z_t[1:], k, lambda_t[:-1])
    w_t = np.cumsum(w_t)

    return w_t


def _stride_subdata(total_nsamples, stride):
    indices = np.arange(0, total_nsamples, stride, dtype=int)
    if indices[-1] != total_nsamples-1:
        indices = np.append(indices, [ total_nsamples-1 ])
    return indices


def _extract_nc(nc_file, stride):
    """
    """
    nc_handle = nc.Dataset(nc_file, "r")
    k = nc_handle.variables["lambda_k"][0]

    k *= BETA

    lambda_t = nc_handle.variables["lambda"]
    z_t = nc_handle.variables["restrained_coordinate"]
    w_t = _work_integrate(lambda_t, z_t, k)
    assert w_t.shape == lambda_t.shape == z_t.shape, "w_t, lambda_t and z_t have different shape " + str(
        w_t.shape) + str(lambda_t.shape) + str(z_t.shape)

    total_nsamples = w_t.shape[0]
    indices = _stride_subdata(total_nsamples, stride)

    w_t = w_t[indices]
    lambda_t = lambda_t[indices]
    z_t = z_t[indices]
    nc_handle.close()

    return k, lambda_t, z_t, w_t, indices


def extract_multiple_nc(nc_files, time_stride, nrepeats, ntrajs):
    """
    nc_files    :   list of str, nc file paths
    time_stride :   int, stride of time dimension
    nrepeats    :   int, number of repeats
    ntrajs      :   int, number of trajectories per repeat
    """
    assert nrepeats * ntrajs == len(nc_files), "nrepeats * ntrajs  not the same as len(nc_files)"
    k, _lambda_t, _z_t, _w_t, indices = _extract_nc(nc_files[0], time_stride)

    times = _lambda_t.shape[0]
    z_t = np.zeros([nrepeats, ntrajs, times], dtype=float)
    w_t = np.zeros([nrepeats, ntrajs, times], dtype=float)

    file_count = -1
    for repeat in range(nrepeats):
        for traj in range(ntrajs):
            file_count += 1
            file = nc_files[file_count]
            print("extracting " + file)
            k, lambda_t, z_t[repeat, traj, :], w_t[repeat, traj, :], indices = _extract_nc(file, time_stride)
            assert lambda_t.shape == _lambda_t.shape, file + " do not have the same data size as " + nc_files[0]
    return k, _lambda_t, z_t, w_t, indices


