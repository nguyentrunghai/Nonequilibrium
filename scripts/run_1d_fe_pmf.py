"""
"""
from __future__ import print_function
from __future__ import division

import argparse

from _IO import load_1d_sim_results
from utils import stride
from utils import equal_spaced_bins

parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file", type=str, default="work.nc")

# to determine pmf bin
parser.add_argument( "--other_pulling_data_nc_files", type=str, default="other_work_files.nc")

# number of data points to take for free energy calculations
parser.add_argument( "--nfe_points", type=int, default=11)

# number of bins for the PMF
parser.add_argument( "--pmf_nbins", type=int, default=10)

# some number or -999 (means None)
parser.add_argument( "--symmetric_center",   type=float, default=-999)

# which side to wrap z, left, right or none
parser.add_argument( "--side_to_wrap_z",   type=str, default="none")

# symmetric or asymmetric
parser.add_argument( "--system_type", type=str, default="symmetric")

# for symmetric systems, asymmetric protocol means pulling only one half of the pmf
# symmetric or asymmetric
parser.add_argument( "--protocol_type", type=str, default="symmetric")

# include uf, ur, b, s
parser.add_argument( "--estimators",  type=str, default="uf b s")

# number of blocks of trajectories
parser.add_argument( "--nblocks",  type=int, default=10)

# numbers of trajectories
parser.add_argument( "--ntrajs_per_block",  type=str, default="10 20 30")

args = parser.parse_args()


def _pmf_bin_edges(pulling_files, nbins, symmetric_center):
    trajs = []
    for f in pulling_files:
        data = nc.Dataset(f, "r")
        zF_t = data.variables["zF_t"]
        zR_t = data.variables["zR_t"]
        trajs.append(zF_t)
        trajs.append(zR_t)
    return equal_spaced_bins(trajs, nbins, symmetric_center=symmetric_center)


if args.system_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized system_type: " + args.system_type)

if args.protocol_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized protocol_type: " + args.protocol_type)

if args.side_to_wrap_z not in ["none", "left", "right"]:
    raise ValueError("Unrecognized side_to_wrap_z: " + args.side_to_wrap_z)

estimators = args.estimators.split()
if not all(e in ["uf", "ur", "b", "s"] for e in estimators):
    raise ValueError("unrecognized estimator(s)")

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)

timeseries_indices = stride(pulling_data["lambda_F"].shape[0], args.nfe_points)
print("timeseries_indices", timeseries_indices)

if args.symmetric_center == -999:
    symmetric_center = None
else:
    symmetric_center = args.symmetric_center

print("symmetric_center", symmetric_center)

pulling_files = [args.pulling_data_nc_file] + [f for f in args.other_pulling_data_nc_files.split()]
print("Data Files to consider for calculating for PMF bins:")
print("\n".join(pulling_files))

pmf_bin_edges = _pmf_bin_edges(pulling_files, args.pmf_nbins, symmetric_center)
if args.system_type == "symmetric" and args.protocol_type == "asymmetric":
    # mean
print("pmf_bin_edges", pmf_bin_edges)
