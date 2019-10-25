"""
write nc file which contains
'zF_t', 'wF_t', 'zR_t', 'wR_t', 'ks',
'lambda_F', 'lambda_R', 'pulling_times', 'dt'
"""
from __future__ import print_function
from __future__ import division

import os
import argparse
import copy

import numpy as np
import netCDF4 as nc

from models_1d import V
from _IO import save_to_nc

parser = argparse.ArgumentParser()

parser.add_argument("--forward_pull_dir", type=str, default="forward")
parser.add_argument("--backward_pull_dir", type=str, default="backward")

parser.add_argument("--forward_force_file", type=str, default="forward.force")
parser.add_argument("--backward_force_file", type=str, default="backward.force")

parser.add_argument("--protocol", type=str, default="symmetric")

parser.add_argument("--range", type=str,  default="0 200")

parser.add_argument("--exclude", type=str,  default=" ")

parser.add_argument("--pulling_speed", type=float,  default=0.01) # Angstrom per ps = speed in A per step / (2*10**(-3))
parser.add_argument("--force_constant", type=float,  default=7.2) # kcal/mol/A^2
parser.add_argument("--lambda_range", type=str,  default="13. 33.")

parser.add_argument("--out", type=str,  default="pull_data.nc")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE


def _lambda_t(pulling_times, pulling_speed, lambda_0):
    """
    :param pulling_times: ndarray of float, ps
    :param pulling_speed: float, Angstrom per ps
    :param lambda_0: float, Angstrom
    :return: lambda_t, ndarray of float
    """
    lambda_t = pulling_times*pulling_speed + lambda_0
    return lambda_t


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
    w_t[1:] = V(z_t[1:], k, lambda_t[1:]) - V(z_t[1:], k, lambda_t[:-1])
    w_t = np.cumsum(w_t)

    return w_t


def _time_z_work(tcl_force_out_file, pulling_speed, lambda_0, k):
    """
    :param tcl_force_out_file: str, file name
    :param pulling_speed: float, Angstrom per ps
    :return: (pulling_times, z_t, w_t) in (ps, Angstrom, kcal/mol)
    """
    data = np.loadtxt(tcl_force_out_file)

    pulling_times = data[:, 0]
    z_t = data[:, 1]
    #forces = data[:, 2]

    #nsteps = len(pulling_times)

    #w_t = np.zeros([nsteps], dtype=float)
    #dts = pulling_times[1:] - pulling_times[:-1]

    #w_t[1:] = pulling_speed * forces[:-1] * dts
    #w_t = np.cumsum(w_t)

    lambda_t = _lambda_t(pulling_times, pulling_speed, lambda_0)
    w_t = _work_integrate(lambda_t, z_t, k)

    return pulling_times, z_t, w_t



def _combine_forward_backward(forward_force_file, backward_force_file,
                              pulling_speed,
                              lambda_min, lambda_max, k):
    t_F, zF_t, wF_t = _time_z_work(forward_force_file, pulling_speed, lambda_min, k)
    t_R, zR_t, wR_t = _time_z_work(backward_force_file, -pulling_speed, lambda_max, k)

    l_F = _lambda_t(t_F, pulling_speed, lambda_min)
    l_R = _lambda_t(t_R, -pulling_speed, lambda_max)
    lambda_F = np.concatenate((l_F, l_R[1:]))
    lambda_R = copy.deepcopy(lambda_F)

    pulling_times = np.concatenate( (t_F, t_R[1:] + t_F[-1]) )
    z_t = np.concatenate( (zF_t, zR_t[1:]) )
    w_t = np.concatenate( (wF_t, wR_t[1:] + wF_t[-1]) )
    return pulling_times, lambda_F, lambda_R, z_t, w_t


def _take_only_forward(forward_force_file, pulling_speed, lambda_min):
    t_F, zF_t, wF_t = _time_z_work(forward_force_file, pulling_speed)
    lambda_F = _lambda_t(t_F, pulling_speed, lambda_min)
    pulling_times = t_F
    return pulling_times, lambda_F, zF_t, wF_t


def _take_only_backward(backward_force_file, pulling_speed, lambda_max):
    t_R, zR_t, wR_t = _time_z_work(backward_force_file, -pulling_speed)
    lambda_R = _lambda_t(t_R, -pulling_speed, lambda_max)
    pulling_times = t_R
    return pulling_times, lambda_R, zR_t, wR_t

# -----------

assert args.protocol in ["symmetric", "asymmetric"], "Unrecognized protocol"

ks = 100. * BETA * args.force_constant
lambda_min = float(args.lambda_range.split()[0])
lambda_max = float(args.lambda_range.split()[1])

start = int(args.range.split()[0])
end = int(args.range.split()[1])

exclude = [int(s) for s in args.exclude.split()]
print("exclude", exclude)

indices_to_collect = [i for i in range(start, end) if i not in exclude]
forward_files = [os.path.join(args.forward_pull_dir, "%d"%i, args.forward_force_file)
                 for i in indices_to_collect]
backward_files = [os.path.join(args.backward_pull_dir, "%d"%i, args.backward_force_file)
                  for i in indices_to_collect]

if args.protocol == "symmetric":
    pulling_times, lambda_F, lambda_R, _, _ = _combine_forward_backward(forward_files[0], backward_files[0],
                                                                  args.pulling_speed, lambda_min, lambda_max)
else:
    pulling_times, lambda_F, _, _ = _take_only_forward(forward_files[0], args.pulling_speed, lambda_min)
    _, lambda_R, _, _ = _take_only_backward(backward_files[0], args.pulling_speed, lambda_max)


dt = pulling_times[1] - pulling_times[0]
nsteps = pulling_times.shape[0]

if args.protocol == "symmetric":
    ntrajs = len(forward_files)
else:
    ntrajs = len(forward_files) + len(backward_files)

assert ntrajs % 2 == 0, "ntrajs must be even"

half_ntrajs = ntrajs // 2

z_ts = np.zeros([ntrajs, nsteps], dtype=float)
w_ts = np.zeros([ntrajs, nsteps], dtype=float)

for i, (f_file, b_file) in enumerate(zip(forward_files, backward_files)):
    print("loading:", f_file)
    print("loading:", b_file)
    if args.protocol == "symmetric":
        _, _, _, z_ts[i, :], w_ts[i, :] = _combine_forward_backward(f_file, b_file, args.pulling_speed,
                                                                    lambda_min, lambda_max)
    else:
        _, _, z_ts[i, :], w_ts[i, :] = _take_only_forward(f_file, args.pulling_speed, lambda_min)
        _, _, z_ts[i + half_ntrajs, :], w_ts[i + half_ntrajs, :] = _take_only_backward(b_file,
                                                                                    args.pulling_speed, lambda_max)

lambda_F /= 10.            # to nm
lambda_R /= 10.            # to nm
z_ts /= 10.            # to nm
w_ts *= BETA       # kcal/mol to KT/mol

data = {"dt" : np.array([dt], dtype=float),
        "pulling_times" : pulling_times,
        "ks" : np.array([ks]),
        "lambda_F" : lambda_F,

        "wF_t" : w_ts[ : half_ntrajs, :],
        "zF_t" : z_ts[ : half_ntrajs, :],

        "lambda_R": lambda_R,
        "wR_t" : w_ts[half_ntrajs :, :],
        "zR_t" : z_ts[half_ntrajs :, :],
        }

out_nc_handle = nc.Dataset(args.out, "w", format="NETCDF4")
save_to_nc(data, out_nc_handle)
out_nc_handle.close()

print("DONE")
