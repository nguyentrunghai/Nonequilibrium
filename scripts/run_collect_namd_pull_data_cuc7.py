"""
write nc file which contains
'zF_t', 'wF_t', 'zR_t', 'wR_t', 'ks',
'lambda_F', 'lambda_R', 'pulling_times', 'dt'
"""

import os
import argparse

import numpy as np
import netCDF4 as nc

from _IO import save_to_nc

parser = argparse.ArgumentParser()

parser.add_argument("--forward_pull_dir", type=str, default="forward")
parser.add_argument("--backward_pull_dir", type=str, default="backward")

parser.add_argument("--range", type=str,  default="0 10")

parser.add_argument("--exclude", type=str,  default=" ")

parser.add_argument("--pulling_speed", type=float,  default=0.1)  # Angstrom per ps = speed in A per step / (2*10**(-3))
parser.add_argument("--force_constant", type=float,  default=7.2) # kcal/mol/A^2
parser.add_argument("--lambda_range", type=str,  default="-20. 20.") # Angstrom

parser.add_argument("--ntrajs_per_block", type=int,  default=10)
parser.add_argument("--out", type=str,  default="pull_data.nc")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE

FORCE_FILE = "cuc7.force"


def _time_z_work(tcl_force_out_file, pulling_speed):
    """
    :param tcl_force_out_file: str, file name
    :param pulling_speed: float, Angstrom per ps
    :return: (pulling_times, z_t, w_t) in (ps, Angstrom, kcal/mol)
    """
    data = np.loadtxt(tcl_force_out_file)

    pulling_times = data[:, 0]
    z_t = data[:, 1]
    forces = data[:, 2]

    nsteps = len(pulling_times)

    w_t = np.zeros([nsteps], dtype=float)
    dts = pulling_times[1:] - pulling_times[:-1]

    w_t[1:] = pulling_speed * forces[:-1] * dts
    w_t = np.cumsum(w_t)

    return pulling_times, z_t, w_t


def _lambda_t(pulling_times, pulling_speed, z0):
    """
    :param pulling_times: ndarray of float, ps
    :param pulling_speed: float, Angstrom per ps
    :param z0: float, Angstrom
    :return: lambda_t, ndarray of float
    """
    lambda_t = pulling_times*pulling_speed + z0
    return lambda_t

# -----------

ks = 100. * BETA * args.force_constant             # kT per nm^2
lambda_min = float(args.lambda_range.split()[0])
lambda_max = float(args.lambda_range.split()[1])

start = int(args.range.split()[0])
end = int(args.range.split()[1])

exclude = [int(s) for s in args.exclude.split()]
print("exclude", exclude)

indices_to_collect = [i for i in range(start, end) if i not in exclude]

forward_files  = [os.path.join(args.forward_pull_dir, "%d"%i, FORCE_FILE ) for i in indices_to_collect]
backward_files = [os.path.join(args.backward_pull_dir, "%d"%i, FORCE_FILE ) for i in indices_to_collect]

for f in forward_files:
    assert os.path.exists(f), f + " does not exit."
for f in backward_files:
    assert os.path.exists(f), f + " does not exit."

pulling_times_F, _, _ = _time_z_work(forward_files[0], args.pulling_speed)
lambda_F = _lambda_t(pulling_times_F, args.pulling_speed, lambda_min)

pulling_times_R, _, _ = _time_z_work(backward_files[0], -args.pulling_speed)
lambda_R = _lambda_t(pulling_times_R, -args.pulling_speed, lambda_max)

dt = pulling_times_F[1] - pulling_times_F[0]
nsteps = pulling_times_F.shape[0]
assert nsteps == pulling_times_R.shape[0]

assert len(forward_files) %  args.ntrajs_per_block == 0, "total number of data files must be multiple of ntrajs_per_block"
nrepeats = len(forward_files) / args.ntrajs_per_block


zF_ts = np.zeros([nrepeats, args.ntrajs_per_block, nsteps], dtype=float)
wF_ts = np.zeros([nrepeats, args.ntrajs_per_block, nsteps], dtype=float)

zR_ts = np.zeros([nrepeats, args.ntrajs_per_block, nsteps], dtype=float)
wR_ts = np.zeros([nrepeats, args.ntrajs_per_block, nsteps], dtype=float)

icount = -1
for repeat in range(nrepeats):
    for traj in range(args.ntrajs_per_block):
        icount += 1
        print "loading ", icount

        _, zF_t, wF_t = _time_z_work(forward_files[icount], args.pulling_speed)
        assert zF_t.shape[0] == nsteps, forward_files[icount] + " do not have correct timesteps"
        zF_ts[repeat, traj, :] = zF_t
        wF_ts[repeat, traj, :] = wF_t

        _, zR_t, wR_t = _time_z_work(backward_files[icount], -args.pulling_speed)
        assert zR_t.shape[0] == nsteps, backward_files[icount] + " do not have correct timesteps"
        zR_ts[repeat, traj, :] = zR_t
        wR_ts[repeat, traj, :] = wR_t

lambda_F /= 10.     # to nm
lambda_R /= 10.     # to nm
zF_ts /= 10.        # to nm
wF_ts *= BETA       # kcal/mol to KT/mol

zR_ts /= 10.        # to nm
wR_ts *= BETA       # kcal/mol to KT/mol

out_nc_handle = nc.Dataset(args.out, "w", format="NETCDF4")

data = {"dt" : np.array([dt], dtype=float),
        "pulling_times" : pulling_times_F,
        "ks" : np.array([ks]),
        "lambda_F" : lambda_F,
        "wF_t" : wF_ts,
        "zF_t" : zF_ts,

        "lambda_R": lambda_R,
        "wR_t" : wR_ts,
        "zR_t" : zR_ts,
        }

save_to_nc(data, out_nc_handle)
out_nc_handle.close()

print("DONE")
