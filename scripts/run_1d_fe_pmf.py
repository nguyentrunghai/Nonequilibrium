"""
"""
from __future__ import print_function
from __future__ import division

import argparse

from _IO import load_1d_sim_results
from utils import stride

parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file", type=str, default="work.nc")

# to determine pmf bin
parser.add_argument( "--other_pulling_data_nc_files", type=str, default="other_work_files.nc")

# number of data points to take for free energy calculations
parser.add_argument( "--nfe_points", type=int, default=11)

# number of bins for the PMF
parser.add_argument( "--pmf_nbins", type=int, default=10)

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

if args.system_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized system_type")

if args.protocol_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized protocol_type")

estimators = args.estimators.split()
if not all(e in ["uf", "ur", "b", "s"] for e in estimators):
    raise ValueError("unrecognized estimator(s)")

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)
timeseries_indices = stride(pulling_data["lambda_F"].shape[0], args.nfe_points)