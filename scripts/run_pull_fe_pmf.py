"""
estimate free energy differences and PMF using unidirectional estimator for pulling simulation data
"""
from __future__ import print_function

import argparse
import pickle

import numpy as np
import netCDF4 as nc

from _IO import load_1d_sim_results

from models_1d import V
from utils import equal_spaced_bins
from _fe_pmf import pull_fe_pmf

parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file",      type=str, default="pull_data.nc")

parser.add_argument( "--pmf_lower_edge",            type=float, default=1)
parser.add_argument( "--pmf_upper_edge",            type=float, default=2)
parser.add_argument( "--pmf_nbins",                 type=int, default=25)

parser.add_argument( "--estimators",                type=str, default="u b s1 s2")
parser.add_argument( "--is_system_symmetric",       action="store_true", default=False)

parser.add_argument( "--fe_out_prefix",             type=str, default="fe")
parser.add_argument( "--pmf_out_prefix",            type=str, default="pmf")

args = parser.parse_args()

assert args.pmf_lower_edge < args.pmf_upper_edge, "lower_edge must be less than upper_edge"

print("pulling_data_nc_file", args.pulling_data_nc_file)
print("pmf_lower_edge", args.pmf_lower_edge)
print("pmf_upper_edge", args.pmf_upper_edge)
print("pmf_nbins", args.pmf_nbins)
print("estimators", args.estimators)
print("is_system_symmetric", args.is_system_symmetric)

# work is in unit of kT, distance in nm, force constant ks is in kT/mol/nm*2
pulling_data = load_1d_sim_results(args.pulling_data_nc_file)

pmf_bin_edges = np.linspace(args.pmf_lower_edge, args.pmf_upper_edge, args.pmf_nbins+1)
print("pmf_bin_edges", pmf_bin_edges)

estimators = args.estimators.split()

for estimator in estimators:
    free_energies, pmfs = pull_fe_pmf(estimator, pulling_data, pmf_bin_edges, args.is_system_symmetric, V)

    pickle.dump(free_energies, open(args.fe_out_prefix + "_" + estimator + ".pkl", "w"))
    pickle.dump(pmfs, open(args.pmf_out_prefix + "_" + estimator + ".pkl", "w"))

print("DONE")

