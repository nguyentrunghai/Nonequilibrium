"""
calculate pmf for gA
"""

from __future__ import print_function

import argparse

import numpy as np
import netCDF4 as nc

from models_1d import V
from estimators import uni_pmf, bi_pmf, sym_est_pmf_v1

parser = argparse.ArgumentParser()

parser.add_argument("--work_data_file", type=str, default="work.nc")

parser.add_argument( "--pmf_lower_edge",            type=float, default=-13.0)
parser.add_argument( "--pmf_upper_edge",            type=float, default=13.0)

# in the work data min = -13.4, max=13.3
parser.add_argument( "--max_abs_z",            type=float, default=13.4)

# number of bins for the PMF
parser.add_argument( "--pmf_nbins", type=int, default=20)

# some number or -999 (means None)
parser.add_argument( "--symmetric_center",   type=float, default=-999)

args = parser.parse_args()


def _pmf_bin_edges(lower, upper, nbins, symmetric_center):

    if symmetric_center is not None:
        assert lower < symmetric_center < upper, "symmetric_center is not in between [min, max]"

        left_interval = symmetric_center - lower
        right_interval = upper - symmetric_center

        interval = np.max([left_interval, right_interval])

        lower = symmetric_center - interval
        upper = symmetric_center + interval

    bin_edges = np.linspace(lower, upper, nbins + 1)

    return bin_edges


if args.symmetric_center == -999:
    symmetric_center = None
else:
    symmetric_center = args.symmetric_center
bin_edges = _pmf_bin_edges(args.pmf_lower_edge, args.pmf_upper_edge, args.pmf_nbins, symmetric_center)

bin_edges = np.array([-args.max_abs_z] + list(bin_edges) + [args.max_abs_z])
print("bin_edges", bin_edges)


with nc.Dataset(args.work_data_file, "r") as handle:
    data = {key: handle.variables[key][:] for key in handle.variables.keys()}

zF_t = data["zF_t"][:]
wF_t = data["wF_t"][:]
lambda_F = data["lambda_F"][:]

zR_t = data["zR_t"][:]
wR_t = data["wR_t"][:]
lambda_R = data["lambda_R"][:]

ks = data["ks"][0]

centers_uni, pmf_uni = uni_pmf(zF_t, wF_t, lambda_F, V, ks, bin_edges)
